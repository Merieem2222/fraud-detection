# Fraud Detection - Detection des Transactions Frauduleuses

## Description
Projet de Machine Learning pour detecter les transactions bancaires frauduleuses.
Dataset : Kaggle Credit Card Fraud Detection (284 807 transactions, 492 fraudes).

## Installation
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Utilisation
```bash
# Mettre creditcard.csv dans data/
# Lancer le pipeline complet
python fraud_detection.py

# Voir MLflow
mlflow ui

# Notebook
jupyter notebook notebooks/fraud_detection.ipynb
```

## Structure
```
fraud_detection_project/
├── data/creditcard.csv
├── notebooks/fraud_detection.ipynb
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── outputs/
├── fraud_detection.py
└── requirements.txt
```

## Auteur
Meriem | ECE Paris - Data & AI B3
