# =============================================================================
#  src/evaluate.py  -  Evaluation Complète des Performances
#  Auteur : Meriem | ECE Paris - Data & AI B3
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import os

GOLD   = '#d4af37'
RED    = '#ff6b6b'
GREEN  = '#4ade80'
BLUE   = '#60a5fa'
PURPLE = '#c084fc'
ORANGE = '#fb923c'
COLORS = [GOLD, GREEN, BLUE, PURPLE, ORANGE, RED]

FIGURES_DIR = 'outputs/figures'
REPORTS_DIR = 'outputs/reports'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =============================================================================
# 1. TABLEAU RECAPITULATIF
# =============================================================================

def print_summary_table(results: dict) -> pd.DataFrame:
    """Affiche et sauvegarde un tableau comparatif de tous les modeles."""
    rows = []
    for name, v in results.items():
        rows.append({
            'Modele'   : name,
            'AUC-ROC'  : round(v['roc_auc'],  4),
            'AUC-PR'   : round(v['pr_auc'],    4),
            'F1-Score' : round(v['f1'],        4),
            'Precision': round(v['precision'], 4),
            'Recall'   : round(v['recall'],    4),
            'Temps (s)': v.get('train_time', 0),
        })

    df_res = pd.DataFrame(rows).set_index('Modele').sort_values('AUC-ROC', ascending=False)

    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF DES MODELES")
    print("=" * 70)
    print(df_res.to_string())
    print("=" * 70)

    path = f'{REPORTS_DIR}/model_comparison.csv'
    df_res.to_csv(path)
    print(f"Tableau sauvegarde : {path}")

    return df_res


# =============================================================================
# 2. COURBES ROC
# =============================================================================

def plot_roc_curves(results: dict, y_test: pd.Series) -> None:
    """Trace les courbes ROC pour tous les modeles sur le meme graphique."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Courbes ROC - Comparaison des Modeles', color=GOLD, fontsize=14, fontweight='bold')

    for i, (name, res) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        fpr, tpr, thresholds = roc_curve(y_test, res['y_pred_proba'])
        auc = res['roc_auc']
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name}  (AUC = {auc:.4f})", alpha=0.9)

    # Ligne de reference (random)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random classifier (AUC = 0.5)')
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color='white')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.set_title('Courbe ROC - Receiver Operating Characteristic', color=GOLD, fontsize=11, pad=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/10_roc_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 3. COURBES PRECISION-RECALL
# =============================================================================

def plot_precision_recall_curves(results: dict, y_test: pd.Series) -> None:
    """Trace les courbes Precision-Recall (plus adaptees aux classes desequilibrees)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Courbes Precision-Recall', color=GOLD, fontsize=14, fontweight='bold')

    baseline = y_test.mean()

    for i, (name, res) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        precision, recall, _ = precision_recall_curve(y_test, res['y_pred_proba'])
        ap = res['pr_auc']
        ax.plot(recall, precision, color=color, lw=2.5, label=f"{name}  (AP = {ap:.4f})", alpha=0.9)

    ax.axhline(y=baseline, color='white', linestyle='--', lw=1.5, alpha=0.5,
               label=f'Baseline (prevalence = {baseline:.4f})')

    ax.set_xlim([0.0, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/11_precision_recall_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 4. MATRICES DE CONFUSION
# =============================================================================

def plot_confusion_matrices(results: dict, y_test: pd.Series) -> None:
    """Trace les matrices de confusion pour chaque modele."""
    n     = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle('Matrices de Confusion - Tous les Modeles', color=GOLD, fontsize=14, fontweight='bold')
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        annot = np.array([
            [f'TN\n{tn:,}', f'FP\n{fp:,}'],
            [f'FN\n{fn:,}', f'TP\n{tp:,}']
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            ax=axes_flat[i],
            cmap='YlOrBr',
            linewidths=1.5,
            linecolor='#1a1a1a',
            xticklabels=['Predit Normal', 'Predit Fraude'],
            yticklabels=['Reel Normal',  'Reel Fraude'],
            cbar=True
        )
        axes_flat[i].set_title(
            f'{name}\nAUC={res["roc_auc"]:.3f} | F1={res["f1"]:.3f}',
            color=GOLD, fontsize=10
        )
        axes_flat[i].set_xlabel('Prediction', fontsize=9)
        axes_flat[i].set_ylabel('Realite',    fontsize=9)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/12_confusion_matrices.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 5. COMPARAISON DES METRIQUES
# =============================================================================

def plot_metrics_comparison(results: dict) -> None:
    """Graphique en barres groupees comparant toutes les metriques."""
    names   = list(results.keys())
    metrics = {
        'AUC-ROC'  : [results[n]['roc_auc']   for n in names],
        'F1-Score' : [results[n]['f1']         for n in names],
        'Precision': [results[n]['precision']  for n in names],
        'Recall'   : [results[n]['recall']     for n in names],
        'AUC-PR'   : [results[n]['pr_auc']     for n in names],
    }

    x     = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.suptitle('Comparaison Globale des Metriques', color=GOLD, fontsize=14, fontweight='bold')

    colors_list = [GOLD, GREEN, BLUE, PURPLE, ORANGE]
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric,
                      color=colors_list[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, color='white', fontweight='bold')

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0.6, 1.1)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.15, lw=1)
    ax.axhline(y=0.9, color=GOLD,   linestyle='--', alpha=0.2,  lw=1)
    ax.grid(True, axis='y', alpha=0.15)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/13_metrics_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")


# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_names: list, model_name: str = '', top_n: int = 20) -> None:
    """
    Affiche les features les plus importantes.
    Compatible avec XGBoost, LightGBM, Random Forest.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"  {model_name} ne supporte pas feature_importances_")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    colors = [RED if v > top.mean() else GOLD for v in top.values]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'Feature Importance - {model_name} (Top {top_n})', color=GOLD, fontsize=14, fontweight='bold')

    bars = ax.barh(top.index, top.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axvline(top.mean(), color=GREEN, linestyle='--', linewidth=1.5, label=f'Moyenne: {top.mean():.4f}')
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Top {top_n} features les plus discriminantes', color=GOLD, fontsize=10)
    ax.legend()

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', fontsize=8, color='white')

    plt.tight_layout()
    path = f'{FIGURES_DIR}/14_feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")

    print(f"\nTop 10 features importantes ({model_name}) :")
    print(importances.sort_values(ascending=False).head(10).to_string())


# =============================================================================
# 7. SEUIL DE DECISION
# =============================================================================

def plot_threshold_analysis(results: dict, y_test: pd.Series, model_name: str = None) -> None:
    """
    Analyse l'impact du seuil de decision sur les metriques.
    Permet de trouver le seuil optimal selon l'objectif.
    """
    if model_name is None:
        model_name = max(results, key=lambda k: results[k]['roc_auc'])

    res = results[model_name]
    y_proba = res['y_pred_proba']

    thresholds  = np.linspace(0.01, 0.99, 100)
    precisions  = []
    recalls     = []
    f1_scores   = []
    fpr_list    = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(   recall_score(   y_test, y_pred_t, zero_division=0))
        f1_scores.append( f1_score(       y_test, y_pred_t, zero_division=0))
        tn = ((y_pred_t == 0) & (y_test == 0)).sum()
        fp = ((y_pred_t == 1) & (y_test == 0)).sum()
        fpr_list.append(fp / (fp + tn + 1e-9))

    best_f1_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_f1_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Analyse du Seuil de Decision - {model_name}', color=GOLD, fontsize=13, fontweight='bold')

    axes[0].plot(thresholds, precisions, color=GOLD,  lw=2, label='Precision')
    axes[0].plot(thresholds, recalls,    color=RED,   lw=2, label='Recall')
    axes[0].plot(thresholds, f1_scores,  color=GREEN, lw=2, label='F1-Score')
    axes[0].axvline(best_thresh, color='white', linestyle='--', lw=2,
                    label=f'Seuil optimal F1 : {best_thresh:.2f}')
    axes[0].set_xlabel('Seuil de decision')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision / Recall / F1 selon le seuil', color=GOLD)
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(thresholds, fpr_list, color=RED, lw=2, label='Taux Faux Positifs')
    axes[1].axvline(best_thresh, color='white', linestyle='--', lw=2,
                    label=f'Seuil optimal : {best_thresh:.2f}')
    axes[1].set_xlabel('Seuil de decision')
    axes[1].set_ylabel('Taux de Faux Positifs')
    axes[1].set_title('Taux de Faux Positifs selon le seuil', color=GOLD)
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/15_threshold_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure sauvegardee : {path}")
    print(f"\nSeuil optimal (meilleur F1) : {best_thresh:.3f}")
    print(f"  Precision : {precisions[best_f1_idx]:.4f}")
    print(f"  Recall    : {recalls[best_f1_idx]:.4f}")
    print(f"  F1-Score  : {f1_scores[best_f1_idx]:.4f}")


# =============================================================================
# 8. RAPPORT DU MEILLEUR MODELE
# =============================================================================

def print_best_model_report(results: dict, y_test: pd.Series) -> str:
    """Affiche et sauvegarde le rapport detaille du meilleur modele."""
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_res  = results[best_name]

    print(f"\n{'='*60}")
    print(f"MEILLEUR MODELE : {best_name}")
    print(f"{'='*60}")
    print(f"  AUC-ROC   : {best_res['roc_auc']:.4f}")
    print(f"  AUC-PR    : {best_res['pr_auc']:.4f}")
    print(f"  F1-Score  : {best_res['f1']:.4f}")
    print(f"  Precision : {best_res['precision']:.4f}")
    print(f"  Recall    : {best_res['recall']:.4f}")
    print(f"  Temps     : {best_res.get('train_time', 'N/A')}s")
    print(f"\nClassification Report complet :")
    report_str = classification_report(y_test, best_res['y_pred'], target_names=['Normal', 'Fraude'])
    print(report_str)

    report_dict = classification_report(y_test, best_res['y_pred'],
                                        target_names=['Normal', 'Fraude'], output_dict=True)
    path = f'{REPORTS_DIR}/best_model_classification_report.csv'
    pd.DataFrame(report_dict).transpose().to_csv(path)
    print(f"Rapport sauvegarde : {path}")

    return best_name


# =============================================================================
# PIPELINE COMPLET D'EVALUATION
# =============================================================================

def evaluate_models(results: dict, data: dict) -> None:
    """
    Pipeline complet d'evaluation de tous les modeles.

    Args:
        results : dict retourne par train_all_models()
        data    : dict retourne par preprocess_data()
    """
    print("\n" + "=" * 60)
    print("  EVALUATION DES PERFORMANCES")
    print("=" * 60)

    y_test        = data['y_test']
    X_test        = data['X_test']
    feature_names = data.get('feature_names', list(X_test.columns))

    print("\n[1/7] Tableau comparatif...")
    print_summary_table(results)

    print("\n[2/7] Courbes ROC...")
    plot_roc_curves(results, y_test)

    print("\n[3/7] Courbes Precision-Recall...")
    plot_precision_recall_curves(results, y_test)

    print("\n[4/7] Matrices de confusion...")
    plot_confusion_matrices(results, y_test)

    print("\n[5/7] Comparaison des metriques...")
    plot_metrics_comparison(results)

    print("\n[6/7] Analyse du seuil de decision...")
    plot_threshold_analysis(results, y_test)

    print("\n[7/7] Rapport du meilleur modele...")
    best_name = print_best_model_report(results, y_test)

    # Feature importance du meilleur modele
    best_model = results[best_name]['model']
    print(f"\nFeature importance ({best_name})...")
    plot_feature_importance(best_model, feature_names, model_name=best_name)

    print(f"\nEvaluation terminee.")
    print(f"  Graphiques : {FIGURES_DIR}/")
    print(f"  Rapports   : {REPORTS_DIR}/")
