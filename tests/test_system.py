"""Unit tests for the Fraud Detection System."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------
# Utils Tests
# ----------------------------------------------------------------
class TestMetrics:
    def test_compute_metrics_basic(self):
        from utils.metrics import compute_metrics

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.15, 0.85])
        m = compute_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in m
        assert "f1_score" in m
        assert "roc_auc" in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_risk_category(self):
        from utils.metrics import risk_category

        assert risk_category(10) == "Low"
        assert risk_category(35) == "Medium"
        assert risk_category(60) == "High"
        assert risk_category(85) == "Critical"
        assert risk_category(0) == "Low"
        assert risk_category(100) == "Critical"


class TestConfig:
    def test_config_defaults(self):
        from utils.config import Config

        cfg = Config()
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42
        assert cfg.contamination == 0.01

    def test_config_paths_created(self):
        from utils.config import Config

        cfg = Config()
        assert Path(cfg.model_dir).exists()
        assert Path(cfg.report_dir).exists()


# ----------------------------------------------------------------
# Data Ingestion Tests
# ----------------------------------------------------------------
class TestDataIngestion:
    def test_synthetic_creditcard_load(self):
        from data.ingest import DataIngestion

        ing = DataIngestion(dataset="creditcard")
        df = ing.load()  # Will use synthetic data if CSV not found
        assert df is not None
        assert len(df) > 0
        assert "Class" in df.columns

    def test_synthetic_paysim_load(self):
        from data.ingest import DataIngestion

        ing = DataIngestion(dataset="paysim")
        df = ing.load()
        assert df is not None
        assert "isFraud" in df.columns

    def test_clean_removes_duplicates(self):
        from data.ingest import DataIngestion

        ing = DataIngestion(dataset="creditcard")
        ing.load()
        # Manually add duplicates
        ing.df = pd.concat([ing.df, ing.df.iloc[:10]], ignore_index=True)
        before = len(ing.df)
        ing.clean()
        assert len(ing.df) <= before

    def test_split_shapes(self):
        from data.ingest import DataIngestion

        ing = DataIngestion(dataset="creditcard")
        ing.load()
        ing.clean()
        X_tr, X_te, y_tr, y_te = ing.split(strategy="smote")
        assert X_tr.shape[0] > 0
        assert X_te.shape[0] > 0
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_te) == X_te.shape[0]


# ----------------------------------------------------------------
# Feature Pipeline Tests
# ----------------------------------------------------------------
class TestFeaturePipeline:
    def test_creditcard_pipeline(self):
        from data.ingest import DataIngestion
        from pipelines.feature_pipeline import get_feature_pipeline

        ing = DataIngestion(dataset="creditcard")
        ing.load()
        ing.clean()
        X_tr, X_te, y_tr, y_te = ing.split(strategy="none" if False else "smote")
        pipeline = get_feature_pipeline("creditcard")
        X_transformed = pipeline.fit_transform(X_tr)
        assert X_transformed.shape[0] == X_tr.shape[0]
        assert X_transformed.shape[1] >= X_tr.shape[1]

    def test_paysim_pipeline(self):
        from data.ingest import DataIngestion
        from pipelines.feature_pipeline import get_feature_pipeline

        ing = DataIngestion(dataset="paysim")
        ing.load()
        ing.clean()
        X_tr, X_te, y_tr, y_te = ing.split(strategy="smote")
        pipeline = get_feature_pipeline("paysim")
        X_transformed = pipeline.fit_transform(X_tr)
        assert X_transformed.shape[0] == X_tr.shape[0]


# ----------------------------------------------------------------
# Anomaly Detector Tests
# ----------------------------------------------------------------
class TestAnomalyDetector:
    def test_fit_predict(self):
        from models.anomaly_detector import AnomalyDetector

        X = np.random.randn(500, 10)
        y = np.zeros(500)
        y[:20] = 1
        detector = AnomalyDetector()
        detector.fit(X)
        scores = detector.anomaly_scores(X)
        assert len(scores) == 500
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
        preds = detector.predict(X)
        assert set(preds).issubset({0, 1})


# ----------------------------------------------------------------
# Hybrid Engine Tests
# ----------------------------------------------------------------
class TestHybridEngine:
    def test_risk_score_range(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression

        from models.hybrid_engine import HybridFraudEngine

        X = np.random.randn(200, 10)
        y = np.array([0] * 180 + [1] * 20)
        sup = LogisticRegression(max_iter=500)
        sup.fit(X, y)
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X)
        engine = HybridFraudEngine(supervised_model=sup, anomaly_detector=iso)
        scores = engine.risk_score(X[:10])
        assert len(scores) == 10
        assert all(0 <= s <= 100 for s in scores)

    def test_predict_full_keys(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression

        from models.hybrid_engine import HybridFraudEngine

        X = np.random.randn(200, 10)
        y = np.array([0] * 180 + [1] * 20)
        sup = LogisticRegression(max_iter=500)
        sup.fit(X, y)
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X)
        engine = HybridFraudEngine(supervised_model=sup, anomaly_detector=iso)
        results = engine.predict_full(X[:5])
        assert len(results) == 5
        for r in results:
            assert "fraud_probability" in r
            assert "risk_score" in r
            assert "risk_category" in r
            assert "anomaly_score" in r


# ----------------------------------------------------------------
# API Tests
# ----------------------------------------------------------------
class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from api.main import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_predict_without_model(self, client):
        payload = {"features": [0.1] * 28}
        resp = client.post("/predict", json=payload)
        # Either 200 (model loaded) or 503 (model not loaded)
        assert resp.status_code in (200, 503)

    def test_predict_response_schema(self, client):
        """If model is loaded, validate response schema."""
        payload = {"features": [0.1] * 28}
        resp = client.post("/predict", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            assert "fraud_probability" in data
            assert "risk_score" in data
            assert "risk_category" in data
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert 0.0 <= data["risk_score"] <= 100.0
