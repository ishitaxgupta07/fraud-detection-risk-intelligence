"""Phase 3: ML Model Training, Comparison, and Selection."""

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils.config import Config
from utils.logger import get_logger
from utils.metrics import compute_metrics

logger = get_logger("ml_trainer")
cfg = Config()


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.random_state),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=cfg.random_state, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=cfg.n_estimators_xgb,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        scale_pos_weight=cfg.scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=-1,
    ),
}


class ModelTrainer:
    def __init__(self, dataset: str = "creditcard"):
        self.dataset = dataset
        self.results: dict = {}
        self.best_model_name: str = None
        self.best_model = None

    def train_all(self, X_train, y_train, X_test, y_test, feature_names=None):
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)

        for name, model in MODELS.items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics = compute_metrics(y_test, y_pred, y_prob)

                # Log to MLflow
                mlflow.log_params({"model": name, "dataset": self.dataset})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.sklearn.log_model(model, name)

                self.results[name] = {"model": model, "metrics": metrics}
                logger.info(f"{name} | ROC-AUC: {metrics.get('roc_auc', 'N/A')} | F1: {metrics['f1_score']}")

        self._select_best()
        self._save_results()
        return self.results

    def _select_best(self):
        best_score = -1
        for name, res in self.results.items():
            score = res["metrics"].get("roc_auc", res["metrics"]["f1_score"])
            if score > best_score:
                best_score = score
                self.best_model_name = name
                self.best_model = res["model"]
        logger.info(f"Best model: {self.best_model_name} (ROC-AUC/F1={best_score:.4f})")
        joblib.dump(self.best_model, cfg.model_dir / "best_model.pkl")
        joblib.dump(self.best_model, cfg.model_dir / f"{self.best_model_name}.pkl")

    def _save_results(self):
        report = {}
        for name, res in self.results.items():
            m = res["metrics"].copy()
            m.pop("confusion_matrix", None)
            m.pop("classification_report", None)
            report[name] = m
        out = cfg.report_dir / "model_comparison.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Model comparison saved to {out}")

    def get_comparison_df(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            m = res["metrics"]
            rows.append(
                {
                    "Model": name,
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "F1 Score": m["f1_score"],
                    "ROC-AUC": m.get("roc_auc", None),
                    "Avg Precision": m.get("avg_precision", None),
                }
            )
        return pd.DataFrame(rows).set_index("Model")
