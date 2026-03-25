# =============================================================================
#  fraud_detection.py  -  Pipeline Principal
#  Auteur : Meriem | ECE Paris - Data & AI B3
#
#  UTILISATION :
#    1. Mettre creditcard.csv dans le dossier data/
#    2. Lancer : python fraud_detection.py
#    3. Voir MLflow : mlflow ui
# =============================================================================

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.eda           import run_eda
from src.preprocessing import preprocess_data
from src.train         import train_all_models
from src.evaluate      import evaluate_models
from src.predict       import predict_transaction


# =============================================================================
# CONFIGURATION DU PIPELINE
# =============================================================================

CONFIG = {
    'data_path'         : 'data/creditcard.csv',
    'resampling_method' : 'smote',    # smote | adasyn | undersampling | smotetomek
    'scaler_type'       : 'standard', # standard | robust
    'test_size'         : 0.2,        # 20% pour le test
    'random_state'      : 42,
    'run_eda'           : True,       # False pour sauter l EDA et aller plus vite
    'optimize_xgb'      : False,      # True pour lancer GridSearchCV (plus lent)
    'cross_validate'    : False,      # True pour validation croisee
    'compare_resampling': False,      # True pour comparer les methodes de reequilibrage
}


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "FRAUD DETECTION PIPELINE" + " " * 24 + "#")
    print("#" + " " * 20 + "ECE Paris - Data & AI B3" + " " * 24 + "#")
    print("#" * 70 + "\n")

    # ------------------------------------------------------------------
    # ETAPE 1 : Chargement du dataset
    # ------------------------------------------------------------------
    print("=" * 70)
    print("ETAPE 1 : CHARGEMENT DES DONNEES")
    print("=" * 70)

    print(f"\nFichier : {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG['data_path'])

    print(f"Dataset charge avec succes !")
    print(f"  Nombre de transactions : {df.shape[0]:,}")
    print(f"  Nombre de features     : {df.shape[1]}")
    print(f"  Fraudes                : {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")
    print(f"  Valeurs manquantes     : {df.isnull().sum().sum()}")
    print(f"\nApercu des 5 premieres lignes :")
    print(df.head().to_string())

    # ------------------------------------------------------------------
    # ETAPE 2 : EDA
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPE 2 : ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 70)

    if CONFIG['run_eda']:
        run_eda(df)
    else:
        print("EDA ignoree (run_eda=False dans CONFIG)")
        print("Pour l activer : mettre CONFIG['run_eda'] = True")

    # ------------------------------------------------------------------
    # ETAPE 3 : Pretraitement
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPE 3 : PRETRAITEMENT DES DONNEES")
    print("=" * 70)

    data = preprocess_data(
        df,
        resampling_method = CONFIG['resampling_method'],
        scaler_type       = CONFIG['scaler_type'],
        test_size         = CONFIG['test_size'],
        random_state      = CONFIG['random_state'],
        compare_methods   = CONFIG['compare_resampling'],
    )

    # ------------------------------------------------------------------
    # ETAPE 4 : Entrainement
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPE 4 : ENTRAINEMENT DES MODELES")
    print("=" * 70)

    results = train_all_models(
        data,
        optimize       = CONFIG['optimize_xgb'],
        cross_validate = CONFIG['cross_validate'],
    )

    # ------------------------------------------------------------------
    # ETAPE 5 : Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPE 5 : EVALUATION DES PERFORMANCES")
    print("=" * 70)

    evaluate_models(results, data)

    # ------------------------------------------------------------------
    # ETAPE 6 : Test de prediction en temps reel
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPE 6 : TEST DE PREDICTION EN TEMPS REEL")
    print("=" * 70)

    print("\nTest avec 2 transactions exemples...")

    # Transaction 1 : normale
    normal_tx = {
        'Time': 80000,
        'V1' :  0.23,  'V2' :  0.05,  'V3' :  0.22,  'V4' :  0.21,
        'V5' : -0.01,  'V6' :  0.12,  'V7' :  0.08,  'V8' :  0.03,
        'V9' : -0.02,  'V10':  0.01,  'V11':  0.04,  'V12': -0.03,
        'V13': -0.01,  'V14':  0.02,  'V15':  0.01,  'V16': -0.01,
        'V17': -0.01,  'V18':  0.00,  'V19':  0.01,  'V20':  0.00,
        'V21':  0.00,  'V22':  0.01,  'V23':  0.00,  'V24':  0.00,
        'V25':  0.01,  'V26':  0.00,  'V27':  0.00,  'V28':  0.00,
        'Amount': 45.50
    }

    # Transaction 2 : suspecte
    suspicious_tx = {
        'Time': 80000,
        'V1' : -3.04,  'V2' :  2.11,  'V3' : -3.58,  'V4' :  3.25,
        'V5' : -2.88,  'V6' : -1.59,  'V7' : -2.62,  'V8' :  0.84,
        'V9' : -1.57,  'V10': -4.46,  'V11':  3.15,  'V12': -7.48,
        'V13':  0.13,  'V14': -6.45,  'V15':  0.27,  'V16': -2.75,
        'V17': -4.61,  'V18': -1.25,  'V19': -1.09,  'V20': -0.37,
        'V21': -0.54,  'V22': -0.20,  'V23': -0.24,  'V24':  0.11,
        'V25': -0.23,  'V26':  0.30,  'V27': -0.01,  'V28':  0.01,
        'Amount': 2450.00
    }

    print("\nTransaction 1 (normale, montant : 45.50 EUR) :")
    r1 = predict_transaction(normal_tx)

    print("\nTransaction 2 (suspecte, montant : 2450.00 EUR) :")
    r2 = predict_transaction(suspicious_tx)

    # ------------------------------------------------------------------
    # RESUME FINAL
    # ------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("PIPELINE TERMINE AVEC SUCCES")
    print("#" * 70)
    print(f"\nFichiers generes :")
    print(f"  models/best_fraud_model.pkl     <- modele sauvegarde")
    print(f"  models/scaler.pkl               <- scaler sauvegarde")
    print(f"  outputs/figures/                <- tous les graphiques PNG")
    print(f"  outputs/reports/                <- tableaux CSV")
    print(f"  mlruns/                         <- experiences MLflow")
    print(f"\nCommandes utiles :")
    print(f"  mlflow ui                       <- visualiser les experiences")
    print(f"  jupyter notebook                <- ouvrir le notebook")
    print("#" * 70 + "\n")


if __name__ == '__main__':
    main()
