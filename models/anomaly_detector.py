"""Phase 4: Anomaly Detection with Isolation Forest."""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from utils.config import Config
from utils.logger import get_logger
from utils.metrics import compute_metrics

logger = get_logger("anomaly_detector")
cfg = Config()


class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators_if,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        self.threshold = None

    def fit(self, X_train: np.ndarray):
        logger.info("Fitting Isolation Forest...")
        self.model.fit(X_train)
        joblib.dump(self.model, cfg.model_dir / "isolation_forest.pkl")
        logger.info("Isolation Forest saved.")
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return normalized anomaly scores in [0, 1] where 1 = most anomalous."""
        raw = self.model.decision_function(X)  # higher = more normal
        # Invert and normalize to [0, 1]
        scores = -raw
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros(len(scores))
        return (scores - min_s) / (max_s - min_s)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1 = fraud/anomaly, 0 = normal."""
        raw = self.model.predict(X)  # -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        y_prob = self.anomaly_scores(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        logger.info(f"Isolation Forest | F1: {metrics['f1_score']} | ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        return metrics
