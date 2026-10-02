# Fraud Detection & Risk Intelligence System

An explainable machine learning system for detecting potentially fraudulent financial transactions using supervised classification, anomaly detection, and SHAP-based explainability.

## Overview

This project implements a hybrid fraud detection pipeline combining supervised machine learning with unsupervised anomaly detection.

### Key Features

- Data preprocessing and exploratory analysis
- Feature engineering for transaction and behavioral signals
- Class imbalance handling using SMOTE and undersampling
- Fraud classification using Logistic Regression, Random Forest, and XGBoost
- Anomaly detection using Isolation Forest
- Hybrid transaction risk scoring
- SHAP-based model explainability
- FastAPI REST API for model inference
- Streamlit dashboard for analysis and visualization
- MLflow experiment tracking
- Docker-based deployment

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Imbalanced Data | imbalanced-learn, SMOTE |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Experiment Tracking | MLflow |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |

## Project Structure

```text
fraud-detection-risk-intelligence/
│
├── api/
│   └── main.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│
├── data/
│   └── ingest.py
│
├── experiments/
│   └── runner.py
│
├── models/
│   ├── trainer.py
│   ├── anomaly_detector.py
│   ├── hybrid_engine.py
│   └── explainer.py
│
├── pipelines/
│   └── feature_pipeline.py
│
├── reports/
│
├── tests/
│   └── test_system.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── metrics.py
│
├── train.py
├── requirements.txt
├── pyproject.toml
└── README.md

Installation
Clone the Repository
git clone https://github.com/ishitaxgupta07/fraud-detection-risk-intelligence.git
cd fraud-detection-risk-intelligence

Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate

Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Dataset
The system supports:
- Credit Card Fraud Detection Dataset
- PaySim Synthetic Financial Dataset
Place the required datasets inside the data/ directory.
data/
├── creditcard.csv
└── PS_20174392719_1491204439457_log.csv

Training
Train the fraud detection pipeline:
python train.py --dataset creditcard --imbalance smote

For a faster development run:
python train.py --dataset creditcard --skip-shap --skip-experiments

Machine Learning Pipeline
Transaction Data
       ↓
Data Ingestion
       ↓
Data Cleaning & EDA
       ↓
Feature Engineering
       ↓
Class Imbalance Handling
       ↓
Model Training
       ↓
XGBoost + Isolation Forest
       ↓
Hybrid Risk Score
       ↓
SHAP Explainability
       ↓
Fraud Risk Analysis

Model Components
Supervised Classification
The system evaluates:
- Logistic Regression
- Random Forest
- XGBoost
XGBoost is used as the primary classification model.
Anomaly Detection
Isolation Forest identifies transactions with unusual patterns that may indicate fraudulent behavior.
Hybrid Risk Score
The system combines the supervised fraud probability and anomaly score into a unified risk score for transaction-level risk assessment.
Explainability
SHAP provides:
- Global feature importance
- Local prediction explanations
- Feature-level contribution analysis
API
Start the FastAPI server:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

API documentation:
http://localhost:8000/docs

Endpoints
GET  /health
POST /predict
POST /explain
POST /risk-score

Dashboard
Start the Streamlit dashboard:
streamlit run dashboard/app.py

The dashboard provides:
- Fraud detection
- Transaction analytics
- Model comparison
- Risk score visualization
- SHAP explanations
- Experiment results
Experiment Tracking
MLflow is used to track model experiments and compare different configurations.
Start MLflow:
mlflow ui --port 5000

Open:
http://localhost:5000

Docker
Build the application:
docker build -f docker/Dockerfile -t fraud-detection .

Run the API:
docker run -p 8000:8000 fraud-detection

Run the complete stack:
cd docker
docker-compose up -d

Services:
API:       http://localhost:8000
Dashboard: http://localhost:8501
MLflow:    http://localhost:5000

Stop the services:
docker-compose down

Testing
Run the test suite:
pytest tests/ -v

Run tests with coverage:
pytest tests/ --cov=. --cov-report=html