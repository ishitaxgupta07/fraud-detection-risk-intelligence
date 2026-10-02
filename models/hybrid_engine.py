"""Phase 5: Hybrid Fraud Engine combining XGBoost + Isolation Forest."""

import joblib
import numpy as np

from utils.config import Config
from utils.logger import get_logger
from utils.metrics import risk_category

logger = get_logger("hybrid_engine")
cfg = Config()


class HybridFraudEngine:
    """
    Combines supervised fraud probability (XGBoost) with
    unsupervised anomaly score (Isolation Forest) into a
    unified Risk Score [0-100].
    """

    def __init__(
        self,
        supervised_model=None,
        anomaly_detector=None,
        supervised_weight: float = 0.65,
        anomaly_weight: float = 0.35,
    ):
        self.supervised_model = supervised_model
        self.anomaly_detector = anomaly_detector
        self.supervised_weight = supervised_weight
        self.anomaly_weight = anomaly_weight

    def load_models(self):
        self.supervised_model = joblib.load(cfg.model_dir / "best_model.pkl")
        self.anomaly_detector = joblib.load(cfg.model_dir / "isolation_forest.pkl")
        logger.info("Hybrid engine models loaded.")
        return self

    def fraud_probability(self, X: np.ndarray) -> np.ndarray:
        return self.supervised_model.predict_proba(X)[:, 1]

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        raw = self.anomaly_detector.decision_function(X)
        scores = -raw
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros(len(scores))
        return (scores - min_s) / (max_s - min_s)

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        """Return risk score in [0, 100]."""
        fp = self.fraud_probability(X)
        an = self.anomaly_score(X)
        combined = self.supervised_weight * fp + self.anomaly_weight * an
        return np.clip(combined * 100, 0, 100)

    def predict_full(self, X: np.ndarray) -> list:
        """Return list of dicts with full prediction details."""
        fp = self.fraud_probability(X)
        an = self.anomaly_score(X)
        rs = self.risk_score(X)
        results = []
        for i in range(len(X)):
            results.append(
                {
                    "fraud_probability": round(float(fp[i]), 4),
                    "anomaly_score": round(float(an[i]), 4),
                    "risk_score": round(float(rs[i]), 2),
                    "risk_category": risk_category(float(rs[i])),
                    "is_fraud": int(fp[i] >= 0.5 or rs[i] >= 75),
                }
            )
        return results
