#  FraudShield AI — Détection de Fraude Bancaire par Machine Learning

> Pipeline ML complet pour la détection de transactions bancaires frauduleuses : de l'analyse exploratoire au déploiement via API FastAPI et dashboard Streamlit, avec tracking MLflow.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-006400?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-02569B?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.7-0194E2?style=flat-square&logo=mlflow&logoColor=white)

---

##  Le dataset

**Kaggle Credit Card Fraud Detection** — un dataset réel de transactions bancaires européennes.

| Métrique | Valeur |
|----------|--------|
| Transactions totales | **284 807** |
| Fraudes | **492** (0.17%) |
| Features | **30** (V1-V28 via PCA + Time + Amount) |
| Déséquilibre | **578:1** (normal vs fraude) |

Le défi principal : détecter les 0.17% de fraudes dans un dataset massivement déséquilibré.


---

##  Modèles entraînés (5 modèles)

| Modèle | Type | Particularité |
|--------|------|---------------|
| **Logistic Regression** | Supervisé | Baseline avec `class_weight='balanced'` |
| **Random Forest** | Supervisé | Ensemble d'arbres, robuste au bruit |
| **XGBoost** | Supervisé | Gradient Boosting avec `scale_pos_weight` |
| **LightGBM** | Supervisé | Gradient Boosting rapide, `is_unbalance=True` |
| **Isolation Forest** | Non supervisé | Détection d'anomalies sans labels |

Optimisation optionnelle par **GridSearchCV / RandomizedSearchCV** sur XGBoost.

---

##  Architecture du projet

```
fraud_detection_project/
├── fraud_detection.py          # Pipeline principal (point d'entrée)
├── src/
│   ├── eda.py                  # Analyse exploratoire (8 visualisations)
│   ├── preprocessing.py        # Scaling, rééquilibrage (SMOTE/ADASYN)
│   ├── train.py                # Entraînement 5 modèles + MLflow + sauvegarde
│   ├── evaluate.py             # Métriques, courbes ROC/PR, SHAP, rapports
│   └── predict.py              # Prédiction temps réel sur nouvelles transactions
├── app/
│   ├── dashboard.py            # Dashboard Streamlit "FraudShield AI"
│   └── api.py                  # API REST FastAPI avec Swagger
├── models/
│   ├── best_fraud_model.pkl    # Meilleur modèle sauvegardé
│   └── scaler.pkl              # Scaler sauvegardé
├── notebooks/
│   └── fraud_detection.ipynb   # Notebook Jupyter complet
├── data/                       # creditcard.csv (à télécharger depuis Kaggle)
├── outputs/
│   ├── figures/                # Graphiques PNG générés
│   └── reports/                # Rapports CSV
├── mlruns/                     # Expériences MLflow
└── requirements.txt
```

---

##  Installation & utilisation

```bash
# Cloner le repo
git clone https://github.com/Merieem2222/fraud-detection-ml.git
cd fraud-detection-ml

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dépendances
pip install -r requirements.txt
```

### Télécharger le dataset
Téléchargez `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le dans `data/`.

### Lancer le pipeline complet
```bash
python fraud_detection.py
```

### Lancer le dashboard Streamlit
```bash
streamlit run app/dashboard.py
```

### Lancer l'API FastAPI
```bash
uvicorn app.api:app --reload --port 8000
# Documentation Swagger : http://localhost:8000/docs
```

### Visualiser les expériences MLflow
```bash
mlflow ui
# Interface : http://localhost:5000
```

---

## 🛠️ Stack technique

| Composant | Technologies |
|-----------|-------------|
| **ML / Data** | Scikit-learn, XGBoost, LightGBM, Pandas, NumPy |
| **Rééquilibrage** | SMOTE, ADASYN, Undersampling, SMOTETomek (imbalanced-learn) |
| **Explicabilité** | SHAP (feature importance) |
| **Tracking** | MLflow (expériences, métriques, modèles) |
| **Dashboard** | Streamlit (interface interactive) |
| **API** | FastAPI + Pydantic (REST API avec Swagger) |
| **Visualisation** | Matplotlib, Seaborn |
| **Serialisation** | Joblib (sauvegarde modèles) |

---

##  Configuration du pipeline

Le fichier `fraud_detection.py` contient un dictionnaire `CONFIG` pour personnaliser le pipeline :

```python
CONFIG = {
    'resampling_method' : 'smote',     # smote | adasyn | undersampling | smotetomek
    'scaler_type'       : 'standard',  # standard | robust
    'test_size'         : 0.2,
    'run_eda'           : True,        # False pour sauter l'EDA
    'optimize_xgb'      : False,       # True pour GridSearchCV (plus lent)
    'cross_validate'    : False,       # True pour validation croisée
    'compare_resampling': False,       # True pour comparer les méthodes
}
```

---

## Ce que j'ai appris

- Gérer un **déséquilibre extrême** (578:1) avec différentes stratégies de rééchantillonnage
- Comparer **5 algorithmes ML** (supervisés + non supervisé) sur un même problème
- Utiliser **MLflow** pour le suivi d'expériences et la reproductibilité
- Déployer un modèle ML via **API REST** (FastAPI) et **dashboard** (Streamlit)
- Interpréter les prédictions avec **SHAP** pour l'explicabilité
- Construire un **pipeline modulaire** et configurable de bout en bout

---

## Auteure

**Meriem DABDOUBI** — Étudiante B3 Data & IA @ ECE Paris  
[GitHub](https://github.com/Merieem2222)

---

*Projet réalisé dans le cadre de ma formation en Data & Intelligence Artificielle à l'ECE Paris — 2025*
