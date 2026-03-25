# =============================================================================
#  src/train.py  -  Entrainement des Modeles ML
#  Auteur : Meriem | ECE Paris - Data & AI B3
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

import xgboost as xgb
import lightgbm as lgb

GOLD  = '#d4af37'
RED   = '#ff6b6b'
GREEN = '#4ade80'

MODELS_DIR  = 'models'
FIGURES_DIR = 'outputs/figures'
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# 1. DEFINITION DES MODELES
# =============================================================================

def get_all_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Retourne tous les modeles a entrainer avec leurs configurations.

    Args:
        scale_pos_weight : ratio normal/fraude (pour XGBoost)

    Returns:
        dict {nom_modele: instance_modele}
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=500,
            C=0.01,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,        # reduit pour aller plus vite
            max_depth=8,            # limite la profondeur
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,       # reduit
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            use_label_encoder=False
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,       # reduit
            learning_rate=0.1,
            max_depth=5,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        ),
    }


# =============================================================================
# 2. ENTRAINEMENT D'UN MODELE
# =============================================================================

def train_single_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Entraine un seul modele et calcule toutes ses metriques.

    Returns:
        dict avec model, predictions, toutes les metriques, temps d'entrainement
    """
    print(f"\n  Entrainement : {name}...")
    start_time = time.time()

    model.fit(X_train, y_train)

    train_time   = time.time() - start_time
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    result = {
        'model'        : model,
        'name'         : name,
        'y_pred'       : y_pred,
        'y_pred_proba' : y_pred_proba,
        'train_time'   : round(train_time, 2),
        'roc_auc'      : roc_auc_score(y_test, y_pred_proba),
        'pr_auc'       : average_precision_score(y_test, y_pred_proba),
        'f1'           : f1_score(y_test, y_pred),
        'precision'    : precision_score(y_test, y_pred),
        'recall'       : recall_score(y_test, y_pred),
    }

    print(f"    Temps d entrainement : {train_time:.1f}s")
    print(f"    AUC-ROC   : {result['roc_auc']:.4f}")
    print(f"    AUC-PR    : {result['pr_auc']:.4f}")
    print(f"    F1-Score  : {result['f1']:.4f}")
    print(f"    Precision : {result['precision']:.4f}")
    print(f"    Recall    : {result['recall']:.4f}")

    return result


# =============================================================================
# 3. ISOLATION FOREST
# =============================================================================

def train_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination: float = 0.002
) -> dict:
    """
    Entraine le modele Isolation Forest (detection d'anomalies non supervise).
    Ne necessite pas les labels y_train.

    Args:
        contamination : proportion estimee de fraudes dans le dataset

    Returns:
        dict avec model, predictions et metriques
    """
    print(f"\n  Entrainement : Isolation Forest (non supervise)...")
    print(f"    contamination = {contamination}")

    start_time = time.time()
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    train_time = time.time() - start_time

    # Les scores negatifs = plus isolees = plus suspectes
    iso_scores = -iso.score_samples(X_test)
    # predict renvoie -1 pour anomalie, 1 pour normal
    iso_pred   = (iso.predict(X_test) == -1).astype(int)

    result = {
        'model'        : iso,
        'name'         : 'Isolation Forest',
        'y_pred'       : iso_pred,
        'y_pred_proba' : iso_scores,
        'train_time'   : round(train_time, 2),
        'roc_auc'      : roc_auc_score(y_test, iso_scores),
        'pr_auc'       : average_precision_score(y_test, iso_scores),
        'f1'           : f1_score(y_test, iso_pred),
        'precision'    : precision_score(y_test, iso_pred, zero_division=0),
        'recall'       : recall_score(y_test, iso_pred, zero_division=0),
    }

    print(f"    Temps d entrainement : {train_time:.1f}s")
    print(f"    AUC-ROC   : {result['roc_auc']:.4f}")
    print(f"    F1-Score  : {result['f1']:.4f}")

    return result


# =============================================================================
# 4. VALIDATION CROISEE STRATIFIEE
# =============================================================================

def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> dict:
    """
    Effectue une validation croisee stratifiee sur un modele.

    Returns:
        dict avec scores moyens et ecart-types
    """
    print(f"\n  Validation croisee ({n_splits} folds) : {name}...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores_roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc',   n_jobs=-1)
    scores_f1  = cross_val_score(model, X, y, cv=cv, scoring='f1',        n_jobs=-1)
    scores_pr  = cross_val_score(model, X, y, cv=cv, scoring='average_precision', n_jobs=-1)

    result = {
        'roc_auc_mean' : scores_roc.mean(),
        'roc_auc_std'  : scores_roc.std(),
        'f1_mean'      : scores_f1.mean(),
        'f1_std'       : scores_f1.std(),
        'pr_auc_mean'  : scores_pr.mean(),
        'pr_auc_std'   : scores_pr.std(),
    }

    print(f"    AUC-ROC : {result['roc_auc_mean']:.4f} (+/- {result['roc_auc_std']:.4f})")
    print(f"    F1      : {result['f1_mean']:.4f} (+/- {result['f1_std']:.4f})")
    print(f"    AUC-PR  : {result['pr_auc_mean']:.4f} (+/- {result['pr_auc_std']:.4f})")

    return result


# =============================================================================
# 5. OPTIMISATION HYPERPARAMETRES
# =============================================================================

def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale_pos_weight: float = 1.0,
    search_type: str = 'grid'
) -> dict:
    """
    Optimise les hyperparametres de XGBoost.

    Args:
        search_type : 'grid' (GridSearchCV) ou 'random' (RandomizedSearchCV)

    Returns:
        dict avec meilleur modele et metriques
    """
    print(f"\n  Optimisation XGBoost [{search_type.upper()}SearchCV]...")

    param_grid = {
        'n_estimators'   : [100, 200, 300],
        'max_depth'      : [4, 6, 8],
        'learning_rate'  : [0.01, 0.05, 0.1],
        'subsample'      : [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        use_label_encoder=False
    )

    if search_type == 'random':
        search = RandomizedSearchCV(
            xgb_base, param_grid,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # GridSearch sur une grille reduite
        param_grid_small = {
            'n_estimators'  : [100, 200],
            'max_depth'     : [4, 6],
            'learning_rate' : [0.05, 0.1],
            'subsample'     : [0.8, 1.0],
        }
        search = GridSearchCV(
            xgb_base, param_grid_small,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

    search.fit(X_train, y_train)

    print(f"  Meilleurs hyperparametres : {search.best_params_}")
    print(f"  Meilleur score CV AUC-ROC : {search.best_score_:.4f}")

    best_model   = search.best_estimator_
    y_pred       = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    result = {
        'model'        : best_model,
        'name'         : 'XGBoost Optimise',
        'best_params'  : search.best_params_,
        'cv_score'     : search.best_score_,
        'y_pred'       : y_pred,
        'y_pred_proba' : y_pred_proba,
        'train_time'   : 0,
        'roc_auc'      : roc_auc_score(y_test, y_pred_proba),
        'pr_auc'       : average_precision_score(y_test, y_pred_proba),
        'f1'           : f1_score(y_test, y_pred),
        'precision'    : precision_score(y_test, y_pred),
        'recall'       : recall_score(y_test, y_pred),
    }

    print(f"\n  Resultats apres optimisation :")
    print(f"    AUC-ROC   : {result['roc_auc']:.4f}")
    print(f"    F1-Score  : {result['f1']:.4f}")
    print(f"    Precision : {result['precision']:.4f}")
    print(f"    Recall    : {result['recall']:.4f}")

    return result


# =============================================================================
# 6. MLFLOW LOGGING
# =============================================================================

def log_to_mlflow(name: str, result: dict, extra_params: dict = None) -> None:
    """Enregistre un modele, ses parametres et ses metriques dans MLflow."""
    mlflow.set_experiment("fraud_detection")

    with mlflow.start_run(run_name=name):
        # Parametres
        mlflow.log_param("model_type",   name)
        mlflow.log_param("resampling",   "SMOTE")
        mlflow.log_param("test_size",    0.2)
        mlflow.log_param("train_time_s", result.get('train_time', 0))

        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

        # Metriques
        mlflow.log_metric("roc_auc",   result['roc_auc'])
        mlflow.log_metric("pr_auc",    result['pr_auc'])
        mlflow.log_metric("f1_score",  result['f1'])
        mlflow.log_metric("precision", result['precision'])
        mlflow.log_metric("recall",    result['recall'])

        # Modele
        try:
            mlflow.sklearn.log_model(result['model'], name.replace(' ', '_').lower())
        except Exception:
            pass

    print(f"  MLflow run enregistre : {name}")


# =============================================================================
# 7. SAUVEGARDE
# =============================================================================

def save_model(model, filename: str) -> None:
    """Sauvegarde un modele avec joblib."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"  Sauvegarde : {path}")


def plot_training_summary(results: dict) -> None:
    """Graphique recapitulatif des performances de tous les modeles."""
    names   = list(results.keys())
    metrics = {
        'AUC-ROC'  : [results[n]['roc_auc']   for n in names],
        'F1-Score' : [results[n]['f1']         for n in names],
        'Precision': [results[n]['precision']  for n in names],
        'Recall'   : [results[n]['recall']     for n in names],
    }

    x = np.arange(len(names))
    width = 0.18
    colors = [GOLD, GREEN, '#60a5fa', '#c084fc']

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle('Comparaison des Performances - Tous les Modeles', color=GOLD, fontsize=14, fontweight='bold')

    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.65, 1.08)
    ax.set_ylabel('Score')
    ax.legend(loc='lower right')
    ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.15)
    ax.axhline(y=0.9, color=GOLD,   linestyle='--', alpha=0.15)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/09_training_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# PIPELINE COMPLET D'ENTRAINEMENT
# =============================================================================

def train_all_models(data: dict, optimize: bool = False, cross_validate: bool = False) -> dict:
    """
    Pipeline complet : entraine tous les modeles, log dans MLflow, sauvegarde le meilleur.

    Args:
        data            : dictionnaire retourne par preprocess_data()
        optimize        : si True, lance GridSearchCV sur XGBoost (plus lent)
        cross_validate  : si True, effectue une validation croisee

    Returns:
        dict de tous les resultats par modele
    """
    print("\n" + "=" * 60)
    print("  ENTRAINEMENT DES MODELES")
    print("=" * 60)

    X_train_res = data['X_train_res']
    y_train_res = data['y_train_res']
    X_test      = data['X_test']
    y_test      = data['y_test']
    X_train     = data['X_train']
    y_train     = data['y_train']

    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"\nScale pos weight : {scale_pos_weight:.1f}")

    models  = get_all_models(scale_pos_weight=scale_pos_weight)
    results = {}

    # --- Modeles supervises ---
    print(f"\n[ETAPE 1/3] Entrainement des modeles supervises ({len(models)} modeles)...")
    for name, model in models.items():
        results[name] = train_single_model(name, model, X_train_res, y_train_res, X_test, y_test)
        log_to_mlflow(name, results[name])

        if cross_validate:
            cv_res = cross_validate_model(name, model, X_train_res, y_train_res)
            results[name]['cv'] = cv_res

    # --- Isolation Forest ---
    print(f"\n[ETAPE 2/3] Isolation Forest (non supervise)...")
    contamination = y_train.mean()
    results['Isolation Forest'] = train_isolation_forest(X_train, X_test, y_test, contamination=contamination)
    log_to_mlflow('Isolation Forest', results['Isolation Forest'])

    # --- Optimisation XGBoost ---
    if optimize:
        print(f"\n[ETAPE 3/3] Optimisation hyperparametres XGBoost...")
        results['XGBoost Optimise'] = optimize_xgboost(
            X_train_res, y_train_res, X_test, y_test,
            scale_pos_weight=scale_pos_weight,
            search_type='grid'
        )
        log_to_mlflow('XGBoost Optimise', results['XGBoost Optimise'],
                      extra_params=results['XGBoost Optimise']['best_params'])
    else:
        print(f"\n[ETAPE 3/3] Optimisation ignoree (optimize=False)")

    # --- Meilleur modele ---
    best_name  = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = results[best_name]['model']

    print(f"\nMeilleur modele : {best_name}")
    print(f"  AUC-ROC   : {results[best_name]['roc_auc']:.4f}")
    print(f"  F1-Score  : {results[best_name]['f1']:.4f}")
    print(f"  Precision : {results[best_name]['precision']:.4f}")
    print(f"  Recall    : {results[best_name]['recall']:.4f}")

    # --- Sauvegarde ---
    print("\nSauvegarde du meilleur modele et du scaler...")
    save_model(best_model,     'best_fraud_model.pkl')
    save_model(data['scaler'], 'scaler.pkl')

    # --- Graphique recapitulatif ---
    print("\nGraphique recapitulatif des performances...")
    plot_training_summary(results)

    print("\nEntrainement termine.")
    return results
