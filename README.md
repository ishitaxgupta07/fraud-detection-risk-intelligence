# Fraud Detection & Risk Intelligence System

An explainable machine learning system for detecting potentially fraudulent financial transactions using supervised classification, anomaly detection, and SHAP-based explainability.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Components](#model-components)
- [API](#api)
- [Dashboard](#dashboard)
- [Experiment Tracking](#experiment-tracking)
- [Docker](#docker)
- [Testing](#testing)

## Overview

This project implements a hybrid fraud detection pipeline that combines supervised machine learning with unsupervised anomaly detection. Each transaction receives a unified risk score, along with SHAP explanations that show why it was flagged.

## Key Features

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

| Category            | Technologies                    |
| ------------------- | ------------------------------- |
| Language            | Python                          |
| Machine Learning    | Scikit-learn, XGBoost           |
| Anomaly Detection   | Isolation Forest                |
| Explainability      | SHAP                            |
| Data Processing     | Pandas, NumPy                   |
| Imbalanced Data     | imbalanced-learn, SMOTE         |
| API                 | FastAPI, Uvicorn, Pydantic      |
| Dashboard           | Streamlit, Plotly               |
| Experiment Tracking | MLflow                          |
| Deployment          | Docker, Docker Compose          |
| Testing             | Pytest                          |

## Project Structure

```text
fraud-detection-risk-intelligence/
│
├── api/
│   └── main.py                 # FastAPI application
│
├── dashboard/
│   ├── app.py                  # Streamlit entry point
│   └── pages/
│
├── data/
│   └── ingest.py               # Data loading and ingestion
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── experiments/
│   └── runner.py               # MLflow experiment runner
│
├── models/
│   ├── trainer.py              # Supervised model training
│   ├── anomaly_detector.py     # Isolation Forest
│   ├── hybrid_engine.py        # Hybrid risk scoring
│   └── explainer.py            # SHAP explanations
│
├── pipelines/
│   └── feature_pipeline.py     # Feature engineering
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
├── train.py                    # Training entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ishitaxgupta07/fraud-detection-risk-intelligence.git
cd fraud-detection-risk-intelligence
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

The system supports the following datasets:

- Credit Card Fraud Detection Dataset
- PaySim Synthetic Financial Dataset

Place the required datasets inside the `data/` directory:

```text
data/
├── creditcard.csv
└── PS_20174392719_1491204439457_log.csv
```

## Training

Train the fraud detection pipeline:

```bash
python train.py --dataset creditcard --imbalance smote
```

For a faster development run (skips SHAP and experiment tracking):

```bash
python train.py --dataset creditcard --skip-shap --skip-experiments
```

## Machine Learning Pipeline

```text
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
```

## Model Components

### Supervised Classification

The system evaluates three classifiers:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost is used as the primary classification model.

### Anomaly Detection

Isolation Forest identifies transactions with unusual patterns that may indicate fraudulent behavior.

### Hybrid Risk Score

The system combines the supervised fraud probability and the anomaly score into a unified risk score for transaction-level risk assessment.

### Explainability

SHAP provides:

- Global feature importance
- Local prediction explanations
- Feature-level contribution analysis

## API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at <http://localhost:8000/docs>.

### Endpoints

| Method | Endpoint      | Description                          |
| ------ | ------------- | ------------------------------------ |
| GET    | `/health`     | Service health check                 |
| POST   | `/predict`    | Predict fraud for a transaction      |
| POST   | `/explain`    | SHAP explanation for a prediction    |
| POST   | `/risk-score` | Hybrid risk score for a transaction  |

## Dashboard

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:

- Fraud detection
- Transaction analytics
- Model comparison
- Risk score visualization
- SHAP explanations
- Experiment results

## Experiment Tracking

MLflow is used to track model experiments and compare different configurations.

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Then open <http://localhost:5000>.

## Docker

Build the image:

```bash
docker build -f docker/Dockerfile -t fraud-detection .
```

Run the API container:

```bash
docker run -p 8000:8000 fraud-detection
```

Run the complete stack:

```bash
cd docker
docker-compose up -d
```

| Service   | URL                   |
| --------- | --------------------- |
| API       | http://localhost:8000 |
| Dashboard | http://localhost:8501 |
| MLflow    | http://localhost:5000 |

Stop the services:

```bash
docker-compose down
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```
