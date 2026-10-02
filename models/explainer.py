"""Phase 6: Explainable AI using SHAP."""

import joblib
import numpy as np
import pandas as pd
import shap

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("explainer")
cfg = Config()


class FraudExplainer:
    def __init__(self, model=None, feature_names: list = None):
        self.model = model or joblib.load(cfg.model_dir / "best_model.pkl")
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def build_explainer(self, X_background: np.ndarray):
        logger.info("Building SHAP TreeExplainer...")
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            logger.warning("TreeExplainer failed, falling back to KernelExplainer")
            background = shap.sample(X_background, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
        return self

    def compute_shap_values(self, X: np.ndarray):
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        # For binary classifiers, take class-1 values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        return self.shap_values

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        sv = self.compute_shap_values(X)
        importance = np.abs(sv).mean(axis=0)
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]
        df = pd.DataFrame({"feature": names, "importance": importance})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        out = cfg.report_dir / "global_feature_importance.json"
        df.to_json(out, orient="records", indent=2)
        logger.info(f"Global importance saved to {out}")
        return df

    def local_explanation(self, X_single: np.ndarray, top_n: int = 5) -> dict:
        """Explain a single prediction."""
        if self.explainer is None:
            raise RuntimeError("Call build_explainer() first.")
        sv = self.explainer.shap_values(X_single)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()
        names = self.feature_names or [f"f{i}" for i in range(len(sv))]
        pairs = sorted(zip(names, sv), key=lambda x: abs(x[1]), reverse=True)
        top = pairs[:top_n]
        reasons = []
        for feat, val in top:
            direction = "increases" if val > 0 else "decreases"
            reasons.append(f"{feat} {direction} fraud risk (SHAP={val:.4f})")
        return {
            "top_features": [{"feature": f, "shap_value": round(v, 4)} for f, v in top],
            "reasons": reasons,
            "base_value": float(
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray))
                else self.explainer.expected_value
            ),
        }

    def generate_text_report(self, X: np.ndarray) -> str:
        df = self.global_importance(X)
        lines = ["# SHAP Global Feature Importance Report\n"]
        lines.append(f"Top {cfg.shap_max_display} features by mean |SHAP| value:\n")
        for _, row in df.head(cfg.shap_max_display).iterrows():
            lines.append(f"- **{row['feature']}**: {row['importance']:.4f}")
        report = "\n".join(lines)
        out = cfg.report_dir / "shap_report.md"
        out.write_text(report)
        logger.info(f"SHAP report saved to {out}")
        return report
