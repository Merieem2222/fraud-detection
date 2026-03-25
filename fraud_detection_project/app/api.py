# =============================================================================
#  app/api.py  -  API FastAPI de Detection de Fraude
#  Auteur : Meriem | ECE Paris - Data & AI B3
#
#  LANCER L'API :
#    uvicorn app.api:app --reload --port 8000
#
#  TESTER L'API :
#    http://localhost:8000/docs   (interface Swagger automatique)
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


# =============================================================================
# INITIALISATION DE L'APP
# =============================================================================

app = FastAPI(
    title       = "FraudShield API",
    description = "API de detection de transactions bancaires frauduleuses - ECE Paris Data & AI",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Autoriser les requetes depuis Streamlit (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Chemins des modeles
MODEL_PATH  = "models/best_fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Chargement du modele au demarrage
model  = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Modele charge  : {MODEL_PATH}")
        print(f"Scaler charge  : {SCALER_PATH}")
    else:
        print("ATTENTION : Modele non trouve. Lance d'abord fraud_detection.py")


# =============================================================================
# SCHEMAS PYDANTIC (validation automatique des donnees)
# =============================================================================

class Transaction(BaseModel):
    """Schema d'une transaction bancaire."""
    Time   : float = Field(..., description="Temps en secondes depuis la premiere transaction")
    V1     : float = Field(..., description="Feature anonymisee V1 (PCA)")
    V2     : float = Field(..., description="Feature anonymisee V2 (PCA)")
    V3     : float = Field(..., description="Feature anonymisee V3 (PCA)")
    V4     : float = Field(..., description="Feature anonymisee V4 (PCA)")
    V5     : float = Field(..., description="Feature anonymisee V5 (PCA)")
    V6     : float = Field(..., description="Feature anonymisee V6 (PCA)")
    V7     : float = Field(..., description="Feature anonymisee V7 (PCA)")
    V8     : float = Field(..., description="Feature anonymisee V8 (PCA)")
    V9     : float = Field(..., description="Feature anonymisee V9 (PCA)")
    V10    : float = Field(..., description="Feature anonymisee V10 (PCA)")
    V11    : float = Field(..., description="Feature anonymisee V11 (PCA)")
    V12    : float = Field(..., description="Feature anonymisee V12 (PCA)")
    V13    : float = Field(..., description="Feature anonymisee V13 (PCA)")
    V14    : float = Field(..., description="Feature anonymisee V14 (PCA)")
    V15    : float = Field(..., description="Feature anonymisee V15 (PCA)")
    V16    : float = Field(..., description="Feature anonymisee V16 (PCA)")
    V17    : float = Field(..., description="Feature anonymisee V17 (PCA)")
    V18    : float = Field(..., description="Feature anonymisee V18 (PCA)")
    V19    : float = Field(..., description="Feature anonymisee V19 (PCA)")
    V20    : float = Field(..., description="Feature anonymisee V20 (PCA)")
    V21    : float = Field(..., description="Feature anonymisee V21 (PCA)")
    V22    : float = Field(..., description="Feature anonymisee V22 (PCA)")
    V23    : float = Field(..., description="Feature anonymisee V23 (PCA)")
    V24    : float = Field(..., description="Feature anonymisee V24 (PCA)")
    V25    : float = Field(..., description="Feature anonymisee V25 (PCA)")
    V26    : float = Field(..., description="Feature anonymisee V26 (PCA)")
    V27    : float = Field(..., description="Feature anonymisee V27 (PCA)")
    V28    : float = Field(..., description="Feature anonymisee V28 (PCA)")
    Amount : float = Field(..., ge=0, description="Montant de la transaction en EUR")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 80000,
                "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
                "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
                "V9": 0.36,  "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99,"V14": -0.31,"V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02,"V22": 0.28, "V23": -0.11,"V24": 0.07,
                "V25": 0.13, "V26": -0.19,"V27": 0.13, "V28": -0.02,
                "Amount": 149.62
            }
        }


class PredictionResult(BaseModel):
    """Schema de la reponse de prediction."""
    transaction_id    : str
    is_fraud          : bool
    fraud_probability : float
    risk_level        : str
    recommendation    : str
    threshold_used    : float
    timestamp         : str
    model_used        : str


class BatchTransaction(BaseModel):
    """Schema pour la prediction en batch."""
    transactions : list[Transaction]
    threshold    : Optional[float] = 0.5


class HealthResponse(BaseModel):
    """Schema de la reponse de sante de l'API."""
    status       : str
    model_loaded : bool
    model_type   : str
    timestamp    : str


class StatsResponse(BaseModel):
    """Schema des statistiques du dataset."""
    total_transactions : int
    total_frauds       : int
    fraud_rate_pct     : float
    avg_amount_normal  : float
    avg_amount_fraud   : float
    top_features       : list[str]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_risk_level(probability: float) -> str:
    if probability < 0.30:   return "LOW"
    elif probability < 0.50: return "MEDIUM"
    elif probability < 0.75: return "HIGH"
    else:                    return "CRITICAL"


def get_recommendation(risk_level: str) -> str:
    recs = {
        "LOW"     : "Approuver automatiquement la transaction",
        "MEDIUM"  : "Demander une validation supplementaire au client",
        "HIGH"    : "Bloquer la transaction et alerter le client",
        "CRITICAL": "Bloquer immediatement et escalader au service fraude",
    }
    return recs.get(risk_level, "Analyser manuellement")


def preprocess(transaction: dict) -> pd.DataFrame:
    """Prepare une transaction pour la prediction."""
    df_tx = pd.DataFrame([transaction])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df_tx['Amount_scaled'] = sc.fit_transform(df_tx[['Amount']])
    df_tx['Time_scaled']   = sc.fit_transform(df_tx[['Time']])
    df_tx = df_tx.drop(columns=['Amount', 'Time'])
    return df_tx


# =============================================================================
# ENDPOINTS DE L'API
# =============================================================================

@app.get("/", tags=["General"])
def root():
    """Page d'accueil de l'API."""
    return {
        "message"     : "FraudShield API - Detection de Fraude Bancaire",
        "version"     : "1.0.0",
        "auteur"      : "Meriem - ECE Paris Data & AI B3",
        "docs"        : "/docs",
        "endpoints"   : ["/health", "/predict", "/predict/batch", "/stats", "/docs"]
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Verifie que l'API et le modele sont operationnels."""
    return HealthResponse(
        status       = "ok" if model is not None else "modele non charge",
        model_loaded = model is not None,
        model_type   = type(model).__name__ if model else "None",
        timestamp    = datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(transaction: Transaction, threshold: float = 0.5):
    """
    Predit si une seule transaction est frauduleuse.

    - **threshold** : seuil de decision (defaut 0.5). Baisser = plus sensible aux fraudes.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge. Lance d'abord fraud_detection.py")

    try:
        tx_dict = transaction.dict()
        df_tx   = preprocess(tx_dict)
        proba   = float(model.predict_proba(df_tx)[0][1])
        risk    = get_risk_level(proba)

        return PredictionResult(
            transaction_id    = f"TX-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}",
            is_fraud          = proba >= threshold,
            fraud_probability = round(proba, 4),
            risk_level        = risk,
            recommendation    = get_recommendation(risk),
            threshold_used    = threshold,
            timestamp         = datetime.now().isoformat(),
            model_used        = type(model).__name__,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchTransaction):
    """
    Predit sur plusieurs transactions en une seule requete.
    Maximum 1000 transactions par requete.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge.")

    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 transactions par requete.")

    try:
        results      = []
        n_fraud      = 0
        threshold    = batch.threshold

        for tx in batch.transactions:
            tx_dict = tx.dict()
            df_tx   = preprocess(tx_dict)
            proba   = float(model.predict_proba(df_tx)[0][1])
            is_fraud = proba >= threshold
            if is_fraud:
                n_fraud += 1
            risk = get_risk_level(proba)

            results.append({
                "transaction_id"    : f"TX-{datetime.now().strftime('%f')}",
                "amount"            : tx.Amount,
                "is_fraud"          : is_fraud,
                "fraud_probability" : round(proba, 4),
                "risk_level"        : risk,
                "recommendation"    : get_recommendation(risk),
            })

        return {
            "total_transactions" : len(results),
            "total_frauds"       : n_fraud,
            "fraud_rate_pct"     : round(n_fraud / len(results) * 100, 3),
            "threshold_used"     : threshold,
            "results"            : results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")


@app.get("/stats", response_model=StatsResponse, tags=["Dataset"])
def get_dataset_stats():
    """Retourne les statistiques du dataset d'entrainement."""
    return StatsResponse(
        total_transactions = 284807,
        total_frauds       = 492,
        fraud_rate_pct     = 0.1727,
        avg_amount_normal  = 88.29,
        avg_amount_fraud   = 122.21,
        top_features       = ["V14", "V10", "V4", "V17", "V12", "V11", "V3", "V7", "V16", "V2"],
    )


@app.get("/predict/example", tags=["Prediction"])
def predict_example():
    """Retourne un exemple de transaction frauduleuse et une normale pour tester."""
    return {
        "transaction_normale": {
            "Time": 80000, "Amount": 45.50,
            "V1":0.23,"V2":0.05,"V3":0.22,"V4":0.21,"V5":-0.01,
            "V6":0.12,"V7":0.08,"V8":0.03,"V9":-0.02,"V10":0.01,
            "V11":0.04,"V12":-0.03,"V13":-0.01,"V14":0.02,"V15":0.01,
            "V16":-0.01,"V17":-0.01,"V18":0.00,"V19":0.01,"V20":0.00,
            "V21":0.00,"V22":0.01,"V23":0.00,"V24":0.00,"V25":0.01,
            "V26":0.00,"V27":0.00,"V28":0.00,
        },
        "transaction_suspecte": {
            "Time": 80000, "Amount": 2450.00,
            "V1":-3.04,"V2":2.11,"V3":-3.58,"V4":3.25,"V5":-2.88,
            "V6":-1.59,"V7":-2.62,"V8":0.84,"V9":-1.57,"V10":-4.46,
            "V11":3.15,"V12":-7.48,"V13":0.13,"V14":-6.45,"V15":0.27,
            "V16":-2.75,"V17":-4.61,"V18":-1.25,"V19":-1.09,"V20":-0.37,
            "V21":-0.54,"V22":-0.20,"V23":-0.24,"V24":0.11,"V25":-0.23,
            "V26":0.30,"V27":-0.01,"V28":0.01,
        }
    }
