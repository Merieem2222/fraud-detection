# =============================================================================
#  src/predict.py  -  Prediction en Temps Reel et en Batch
#  Auteur : Meriem | ECE Paris - Data & AI B3
# =============================================================================

import numpy as np
import pandas as pd
import joblib
import os

MODELS_DIR = 'models'


# =============================================================================
# 1. CHARGEMENT DU MODELE ET DU SCALER
# =============================================================================

def load_model(model_path: str = 'models/best_fraud_model.pkl'):
    """Charge le modele depuis le fichier joblib."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}\n"
            f"Lance d'abord fraud_detection.py pour entrainer et sauvegarder le modele."
        )
    model = joblib.load(model_path)
    print(f"Modele charge : {model_path}")
    return model


def load_scaler(scaler_path: str = 'models/scaler.pkl'):
    """Charge le scaler depuis le fichier joblib."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler introuvable : {scaler_path}\n"
            f"Lance d'abord fraud_detection.py pour entrainer et sauvegarder le scaler."
        )
    scaler = joblib.load(scaler_path)
    print(f"Scaler charge : {scaler_path}")
    return scaler


# =============================================================================
# 2. PREPROCESSING D'UNE TRANSACTION
# =============================================================================

def validate_transaction(transaction: dict) -> None:
    """
    Verifie que la transaction contient toutes les cles necessaires.
    Leve une ValueError si une cle est manquante.
    """
    required_keys = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    missing = [k for k in required_keys if k not in transaction]
    if missing:
        raise ValueError(
            f"Cles manquantes dans la transaction : {missing}\n"
            f"Une transaction doit contenir : V1 a V28, Amount, Time"
        )


def preprocess_single_transaction(transaction: dict, scaler) -> pd.DataFrame:
    """
    Prepare une seule transaction pour la prediction.

    Args:
        transaction : dict avec V1-V28, Amount, Time
        scaler      : StandardScaler fitte sur les donnees d'entrainement

    Returns:
        DataFrame avec les bonnes colonnes pour predict_proba()
    """
    validate_transaction(transaction)

    df_tx = pd.DataFrame([transaction])

    # Normaliser Amount et Time comme pendant l'entrainement
    df_tx['Amount_scaled'] = scaler.transform(df_tx[['Amount']])
    df_tx['Time_scaled']   = scaler.transform(df_tx[['Time']])

    # Supprimer les colonnes originales (non normalisees)
    df_tx = df_tx.drop(columns=['Amount', 'Time'])

    return df_tx


# =============================================================================
# 3. NIVEAUX DE RISQUE
# =============================================================================

def get_risk_level(probability: float) -> str:
    """
    Convertit une probabilite de fraude en niveau de risque.

    Seuils :
        < 0.30  : LOW      (faible risque)
        < 0.50  : MEDIUM   (risque modere)
        < 0.75  : HIGH     (risque eleve)
        >= 0.75 : CRITICAL (risque critique)
    """
    if probability < 0.30:
        return 'LOW'
    elif probability < 0.50:
        return 'MEDIUM'
    elif probability < 0.75:
        return 'HIGH'
    else:
        return 'CRITICAL'


def get_recommendation(risk_level: str) -> str:
    """Retourne la recommandation metier selon le niveau de risque."""
    recs = {
        'LOW'     : 'Approuver automatiquement la transaction',
        'MEDIUM'  : 'Demander une validation supplementaire au client',
        'HIGH'    : 'Bloquer la transaction et alerter le client',
        'CRITICAL': 'Bloquer immediatement et escalader au service fraude',
    }
    return recs.get(risk_level, 'Analyser manuellement')


def get_risk_color(risk_level: str) -> str:
    """Retourne la couleur associee au niveau de risque (pour affichage)."""
    colors = {
        'LOW'     : 'vert',
        'MEDIUM'  : 'orange',
        'HIGH'    : 'rouge',
        'CRITICAL': 'rouge fonce',
    }
    return colors.get(risk_level, 'gris')


# =============================================================================
# 4. PREDICTION UNE TRANSACTION
# =============================================================================

def predict_transaction(
    transaction  : dict,
    model_path   : str   = 'models/best_fraud_model.pkl',
    scaler_path  : str   = 'models/scaler.pkl',
    threshold    : float = 0.5,
    verbose      : bool  = True
) -> dict:
    """
    Predit si une transaction est frauduleuse.

    Args:
        transaction  : dict contenant V1-V28, Amount, Time
        model_path   : chemin vers le modele sauvegarde (.pkl)
        scaler_path  : chemin vers le scaler sauvegarde (.pkl)
        threshold    : seuil de decision (defaut 0.5)
                       - Baisser le seuil = plus sensible (moins de faux negatifs)
                       - Augmenter le seuil = plus specifique (moins de faux positifs)
        verbose      : afficher le resultat dans la console

    Returns:
        dict avec :
            is_fraud           (bool)  : True si fraude detectee
            fraud_probability  (float) : probabilite entre 0 et 1
            risk_level         (str)   : LOW / MEDIUM / HIGH / CRITICAL
            recommendation     (str)   : action recommandee
            threshold_used     (float) : seuil utilise
    """
    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)

    df_tx    = preprocess_single_transaction(transaction, scaler)
    proba    = float(model.predict_proba(df_tx)[0][1])
    is_fraud = proba >= threshold
    risk     = get_risk_level(proba)

    result = {
        'is_fraud'          : is_fraud,
        'fraud_probability' : round(proba, 4),
        'risk_level'        : risk,
        'recommendation'    : get_recommendation(risk),
        'threshold_used'    : threshold,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("RESULTAT DE LA PREDICTION")
        print("=" * 50)
        print(f"  Montant         : {transaction.get('Amount', 'N/A')} EUR")
        print(f"  Fraude detectee : {'OUI' if is_fraud else 'NON'}")
        print(f"  Probabilite     : {proba:.4f} ({proba*100:.2f}%)")
        print(f"  Niveau de risque: {risk} ({get_risk_color(risk)})")
        print(f"  Recommandation  : {result['recommendation']}")
        print(f"  Seuil utilise   : {threshold}")
        print("=" * 50)

    return result


# =============================================================================
# 5. PREDICTION BATCH (PLUSIEURS TRANSACTIONS)
# =============================================================================

def predict_batch(
    df_transactions : pd.DataFrame,
    model_path      : str   = 'models/best_fraud_model.pkl',
    scaler_path     : str   = 'models/scaler.pkl',
    threshold       : float = 0.5,
    output_path     : str   = None
) -> pd.DataFrame:
    """
    Predit en batch sur un DataFrame de transactions.

    Args:
        df_transactions : DataFrame avec colonnes V1-V28, Amount, Time
        output_path     : si fourni, sauvegarde le resultat en CSV

    Returns:
        DataFrame original avec colonnes ajoutees :
            fraud_probability, is_fraud, risk_level, recommendation
    """
    print(f"\nPrediction batch : {len(df_transactions):,} transactions...")

    model  = load_model(model_path)
    scaler = load_scaler(scaler_path)

    df = df_transactions.copy()

    # Normaliser
    df['Amount_scaled'] = scaler.transform(df[['Amount']])
    df['Time_scaled']   = scaler.transform(df[['Time']])
    df_features = df.drop(columns=['Amount', 'Time', 'Class'], errors='ignore')

    # Prediction
    probas = model.predict_proba(df_features)[:, 1]

    # Ajouter les resultats
    df_result = df_transactions.copy()
    df_result['fraud_probability'] = probas.round(4)
    df_result['is_fraud']          = probas >= threshold
    df_result['risk_level']        = [get_risk_level(p)     for p in probas]
    df_result['recommendation']    = [get_recommendation(get_risk_level(p)) for p in probas]

    # Stats
    n_fraud     = df_result['is_fraud'].sum()
    n_critical  = (df_result['risk_level'] == 'CRITICAL').sum()
    n_high      = (df_result['risk_level'] == 'HIGH').sum()

    print(f"\nResultats batch :")
    print(f"  Total transactions : {len(df_result):,}")
    print(f"  Fraudes detectees  : {n_fraud:,} ({n_fraud/len(df_result)*100:.3f}%)")
    print(f"  Risque CRITICAL    : {n_critical:,}")
    print(f"  Risque HIGH        : {n_high:,}")
    print(f"  Risque MEDIUM      : {(df_result['risk_level']=='MEDIUM').sum():,}")
    print(f"  Risque LOW         : {(df_result['risk_level']=='LOW').sum():,}")

    if output_path:
        df_result.to_csv(output_path, index=False)
        print(f"\nResultats sauvegardes : {output_path}")

    return df_result


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == '__main__':

    # Transaction normale
    normal_transaction = {
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

    # Transaction suspecte
    suspicious_transaction = {
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

    print("\nTest 1 : Transaction normale")
    result1 = predict_transaction(normal_transaction)

    print("\nTest 2 : Transaction suspecte")
    result2 = predict_transaction(suspicious_transaction)
