# =============================================================================
#  src/preprocessing.py  -  Pretraitement des Donnees
#  Auteur : Meriem | ECE Paris - Data & AI B3
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
import os

GOLD  = '#d4af37'
RED   = '#ff6b6b'
GREEN = '#4ade80'

FIGURES_DIR = 'outputs/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# 1. NORMALISATION DES FEATURES
# =============================================================================

def scale_features(df: pd.DataFrame, scaler_type: str = 'standard') -> tuple:
    """
    Normalise les colonnes Amount et Time.

    Args:
        df           : DataFrame original
        scaler_type  : 'standard' (StandardScaler) ou 'robust' (RobustScaler)

    Returns:
        df_scaled : DataFrame avec Amount_scaled, Time_scaled (sans Amount et Time originaux)
        scaler    : objet scaler fitte
    """
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    df = df.copy()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Amount', 'Time'])

    print(f"Normalisation [{scaler_type.upper()}] :")
    print(f"  Amount_scaled -> moyenne : {df['Amount_scaled'].mean():.4f}, ecart-type : {df['Amount_scaled'].std():.4f}")
    print(f"  Time_scaled   -> moyenne : {df['Time_scaled'].mean():.4f}, ecart-type : {df['Time_scaled'].std():.4f}")

    return df, scaler


# =============================================================================
# 2. SPLIT TRAIN / TEST
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = 'Class',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Separe X et y puis split train/test de facon stratifiee.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # preserve la proportion de fraudes
    )

    print(f"\nSplit stratifie ({int((1-test_size)*100)}/{int(test_size*100)}) :")
    print(f"  Train : {X_train.shape[0]:,} samples")
    print(f"    - Normal  : {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
    print(f"    - Fraude  : {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
    print(f"  Test  : {X_test.shape[0]:,} samples")
    print(f"    - Normal  : {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
    print(f"    - Fraude  : {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 3. REEQUILIBRAGE DES CLASSES
# =============================================================================

def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'smote',
    random_state: int = 42
) -> tuple:
    """
    Applique une technique de reequilibrage sur les donnees d'entrainement.
    IMPORTANT : ne jamais appliquer sur les donnees de test.

    Args:
        method : 'smote' | 'adasyn' | 'undersampling' | 'smotetomek' | 'tomeklinks'

    Returns:
        X_resampled, y_resampled
    """
    methods_available = {
        'smote'        : SMOTE(random_state=random_state, k_neighbors=5),
        'adasyn'       : ADASYN(random_state=random_state),
        'undersampling': RandomUnderSampler(random_state=random_state),
        'smotetomek'   : SMOTETomek(random_state=random_state),
        'tomeklinks'   : TomekLinks(),
    }

    if method not in methods_available:
        raise ValueError(f"Methode inconnue '{method}'. Choisir parmi : {list(methods_available.keys())}")

    sampler = methods_available[method]

    print(f"\nReequilibrage [{method.upper()}] :")
    print(f"  Avant -> {len(y_train):,} samples | Normal : {(y_train==0).sum():,} | Fraude : {(y_train==1).sum():,}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)

    print(f"  Apres -> {len(y_res):,} samples | Normal : {(y_res==0).sum():,} | Fraude : {(y_res==1).sum():,}")
    print(f"  Ratio final : {(y_res==0).sum() / (y_res==1).sum():.2f} normal pour 1 fraude")

    return X_res, y_res


def compare_resampling_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> None:
    """Compare visuellement les differentes methodes de reequilibrage."""
    methods = ['smote', 'adasyn', 'undersampling', 'smotetomek']
    results = {}

    for method in methods:
        try:
            X_res, y_res = apply_resampling(X_train, y_train, method=method)
            results[method] = {
                'n_total'  : len(y_res),
                'n_normal' : (y_res == 0).sum(),
                'n_fraud'  : (y_res == 1).sum(),
            }
        except Exception as e:
            print(f"Erreur avec {method} : {e}")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparaison des Methodes de Reequilibrage', color=GOLD, fontsize=14, fontweight='bold')

    names = list(results.keys())
    n_normals = [results[m]['n_normal'] for m in names]
    n_frauds  = [results[m]['n_fraud']  for m in names]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, n_normals, w, label='Normal', color=GOLD, alpha=0.8)
    axes[0].bar(x + w/2, n_frauds,  w, label='Fraude', color=RED,  alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel('Nombre de samples')
    axes[0].set_title('Distribution apres reequilibrage', color=GOLD)
    axes[0].legend()

    totals = [results[m]['n_total'] for m in names]
    axes[1].bar(names, totals, color=GOLD, alpha=0.8, edgecolor='black')
    for i, t in enumerate(totals):
        axes[1].text(i, t + 100, f'{t:,}', ha='center', color=GOLD, fontweight='bold', fontsize=9)
    axes[1].set_title('Total de samples par methode', color=GOLD)
    axes[1].set_ylabel('Total')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = f'{FIGURES_DIR}/08_resampling_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparaison sauvegardee : {path}")


# =============================================================================
# PIPELINE COMPLET DE PRETRAITEMENT
# =============================================================================

def preprocess_data(
    df: pd.DataFrame,
    resampling_method: str = 'smote',
    scaler_type: str = 'standard',
    test_size: float = 0.2,
    random_state: int = 42,
    compare_methods: bool = False
) -> dict:
    """
    Pipeline complet de pretraitement des donnees.

    Args:
        df                : DataFrame brut du dataset creditcard
        resampling_method : methode de reequilibrage ('smote', 'adasyn', etc.)
        scaler_type       : type de scaler ('standard' ou 'robust')
        test_size         : proportion du jeu de test (0.2 = 20%)
        random_state      : graine aleatoire pour la reproductibilite
        compare_methods   : si True, compare toutes les methodes de reequilibrage

    Returns:
        dict contenant :
            X_train, X_test, y_train, y_test     : donnees originales splitees
            X_train_res, y_train_res              : donnees train reequilibrees
            scaler                                : scaler fitte
            feature_names                         : liste des noms de features
    """
    print("\n" + "=" * 60)
    print("  PRETRAITEMENT DES DONNEES")
    print("=" * 60)

    print(f"\nConfiguration :")
    print(f"  Scaler            : {scaler_type}")
    print(f"  Reequilibrage     : {resampling_method}")
    print(f"  Ratio train/test  : {int((1-test_size)*100)}/{int(test_size*100)}")
    print(f"  Random state      : {random_state}")

    # Etape 1 : Normalisation
    print("\n[1/3] Normalisation de Amount et Time...")
    df_scaled, scaler = scale_features(df, scaler_type=scaler_type)

    # Etape 2 : Split
    print("\n[2/3] Split train/test stratifie...")
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=test_size, random_state=random_state
    )

    # Comparaison optionnelle des methodes
    if compare_methods:
        print("\n[Optionnel] Comparaison des methodes de reequilibrage...")
        compare_resampling_methods(X_train, y_train)

    # Etape 3 : Reequilibrage
    print("\n[3/3] Reequilibrage des classes sur le train uniquement...")
    X_train_res, y_train_res = apply_resampling(
        X_train, y_train,
        method=resampling_method,
        random_state=random_state
    )

    feature_names = list(X_train.columns)

    print(f"\nPretraitement termine.")
    print(f"  Features utilisees : {len(feature_names)}")
    print(f"  X_train_res shape  : {X_train_res.shape}")
    print(f"  X_test shape       : {X_test.shape}")

    return {
        'X_train'      : X_train,
        'X_test'       : X_test,
        'y_train'      : y_train,
        'y_test'       : y_test,
        'X_train_res'  : X_train_res,
        'y_train_res'  : y_train_res,
        'scaler'       : scaler,
        'feature_names': feature_names,
    }
