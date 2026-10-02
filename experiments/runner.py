"""Phase 7: Research Experiments with MLflow tracking."""

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

from utils.config import Config
from utils.logger import get_logger
from utils.metrics import compute_metrics

logger = get_logger("experiments")
cfg = Config()


class ExperimentRunner:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = []

    def run_all(self):
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(f"{cfg.mlflow_experiment_name}_research")

        # Experiment 1: Different algorithms
        self._experiment_algorithms()
        # Experiment 2: Different imbalance strategies
        self._experiment_imbalance()
        # Experiment 3: Different anomaly thresholds
        self._experiment_anomaly_thresholds()

        self._save_report()
        return self.results

    def _experiment_algorithms(self):
        configs = [
            (
                "XGBoost_shallow",
                XGBClassifier(
                    max_depth=3, n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42
                ),
            ),
            (
                "XGBoost_deep",
                XGBClassifier(
                    max_depth=8, n_estimators=300, use_label_encoder=False, eval_metric="logloss", random_state=42
                ),
            ),
            ("RF_100", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("RF_300", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
        for name, model in configs:
            self._run_single(name, model, self.X_train, self.y_train, "algorithm_comparison")

    def _experiment_imbalance(self):
        strategies = {
            "smote": SMOTE(random_state=42, k_neighbors=3),
            "undersample": RandomUnderSampler(random_state=42),
        }
        for strat_name, sampler in strategies.items():
            X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)
            model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42)
            self._run_single(f"XGB_{strat_name}", model, X_res, y_res, "imbalance_strategy")

    def _experiment_anomaly_thresholds(self):
        for contamination in [0.005, 0.01, 0.02, 0.05]:
            name = f"IF_contamination_{contamination}"
            model = IsolationForest(contamination=contamination, n_estimators=200, random_state=42)
            model.fit(self.X_train)
            y_pred = (model.predict(self.X_test) == -1).astype(int)
            raw = -model.decision_function(self.X_test)
            mn, mx = raw.min(), raw.max()
            y_prob = (raw - mn) / (mx - mn + 1e-9)
            metrics = compute_metrics(self.y_test, y_pred, y_prob)
            self.results.append(
                {
                    "experiment": "anomaly_threshold",
                    "name": name,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )
            logger.info(f"{name} | F1={metrics['f1_score']} | ROC-AUC={metrics.get('roc_auc', 'N/A')}")

    def _run_single(self, name, model, X_tr, y_tr, experiment_type):
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            metrics = compute_metrics(self.y_test, y_pred, y_prob)
            mlflow.log_params({"name": name, "experiment": experiment_type})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            self.results.append(
                {
                    "experiment": experiment_type,
                    "name": name,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            )
            logger.info(f"{name} | F1={metrics['f1_score']} | ROC-AUC={metrics.get('roc_auc', 'N/A')}")

    def _save_report(self):
        df = pd.DataFrame(self.results)
        out_json = cfg.experiment_dir / "research_results.json"
        out_md = cfg.experiment_dir / "research_report.md"
        df.to_json(out_json, orient="records", indent=2)

        lines = ["# Research Experiment Report\n", "## Results\n", df.to_markdown(index=False)]
        out_md.write_text("\n".join(lines))
        logger.info(f"Research report saved to {out_md}")
