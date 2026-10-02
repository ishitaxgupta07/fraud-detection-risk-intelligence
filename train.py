#!/usr/bin/env python3
"""
Main Training Script - Explainable Hybrid Fraud Detection System

Runs all phases sequentially:
  Phase 1: Data Ingestion & EDA
  Phase 2: Feature Engineering
  Phase 3: ML Model Training & Comparison
  Phase 4: Anomaly Detection
  Phase 5: Hybrid Engine Evaluation
  Phase 6: SHAP Explainability
  Phase 7: Research Experiments

Usage:
    python train.py --dataset creditcard --imbalance smote
    python train.py --dataset paysim --imbalance combined
"""

import argparse

import numpy as np

from data.ingest import DataIngestion
from experiments.runner import ExperimentRunner
from models.anomaly_detector import AnomalyDetector
from models.explainer import FraudExplainer
from models.hybrid_engine import HybridFraudEngine
from models.trainer import ModelTrainer
from pipelines.feature_pipeline import get_feature_pipeline
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("train")
cfg = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fraud Detection System")
    parser.add_argument(
        "--dataset", choices=["creditcard", "paysim"], default="creditcard", help="Dataset to use (default: creditcard)"
    )
    parser.add_argument(
        "--imbalance",
        choices=["smote", "undersample", "combined", "none"],
        default="smote",
        help="Imbalance handling strategy (default: smote)",
    )
    parser.add_argument("--skip-experiments", action="store_true", help="Skip research experiments (faster run)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation (faster run)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("  Explainable Hybrid Fraud Detection System")
    logger.info(f"  Dataset: {args.dataset} | Imbalance: {args.imbalance}")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # PHASE 1: Data Ingestion
    # ----------------------------------------------------------------
    logger.info("\n[PHASE 1] Data Ingestion & EDA")
    ingestion = DataIngestion(dataset=args.dataset)
    ingestion.load()
    eda_report = ingestion.eda()
    ingestion.clean()
    X_train_raw, X_test_raw, y_train, y_test = ingestion.split(
        strategy=args.imbalance if args.imbalance != "none" else "smote"
    )
    logger.info(f"EDA: {eda_report['shape']} | Fraud rate: {eda_report['fraud_rate_pct']}%")

    # ----------------------------------------------------------------
    # PHASE 2: Feature Engineering
    # ----------------------------------------------------------------
    logger.info("\n[PHASE 2] Feature Engineering")
    pipeline = get_feature_pipeline(args.dataset)
    X_train = pipeline.fit_transform(X_train_raw, y_train)
    X_test = pipeline.transform(X_test_raw)

    # Recover feature names after engineering
    try:
        engineer = pipeline.named_steps["engineer"]
        # Build a sample to get column names
        sample_df = X_train_raw.iloc[:1].copy()
        transformed_sample = engineer.transform(sample_df)
        if hasattr(transformed_sample, "columns"):
            feature_names = list(transformed_sample.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ----------------------------------------------------------------
    # PHASE 3: ML Model Training
    # ----------------------------------------------------------------
    logger.info("\n[PHASE 3] ML Model Training & Comparison")
    trainer = ModelTrainer(dataset=args.dataset)
    trainer.train_all(X_train, y_train, X_test, y_test, feature_names)
    comparison_df = trainer.get_comparison_df()
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())
    logger.info(f"Best model: {trainer.best_model_name}")

    # ----------------------------------------------------------------
    # PHASE 4: Anomaly Detection
    # ----------------------------------------------------------------
    logger.info("\n[PHASE 4] Anomaly Detection - Isolation Forest")
    detector = AnomalyDetector()
    detector.fit(X_train)
    anomaly_metrics = detector.evaluate(X_test, y_test)
    logger.info(f"Isolation Forest metrics: {anomaly_metrics}")

    # ----------------------------------------------------------------
    # PHASE 5: Hybrid Engine
    # ----------------------------------------------------------------
    logger.info("\n[PHASE 5] Hybrid Fraud Engine")
    engine = HybridFraudEngine(
        supervised_model=trainer.best_model,
        anomaly_detector=detector.model,
    )
    sample_results = engine.predict_full(X_test[:10])
    logger.info("Sample hybrid predictions (first 10):")
    for i, r in enumerate(sample_results):
        logger.info(
            f"  [{i}] FP={r['fraud_probability']:.3f} | "
            f"Anomaly={r['anomaly_score']:.3f} | "
            f"Risk={r['risk_score']:.1f} | "
            f"Category={r['risk_category']} | "
            f"Actual={int(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i])}"
        )

    # ----------------------------------------------------------------
    # PHASE 6: SHAP Explainability
    # ----------------------------------------------------------------
    if not args.skip_shap:
        logger.info("\n[PHASE 6] SHAP Explainability")
        explainer = FraudExplainer(
            model=trainer.best_model,
            feature_names=feature_names,
        )
        # Use a background sample for efficiency
        background = X_test[:200] if len(X_test) > 200 else X_test
        explainer.build_explainer(background)
        explainer.generate_text_report(background)
        logger.info("SHAP report generated.")

        # Local explanation for first fraud case
        fraud_indices = np.where(np.array(y_test) == 1)[0]
        if len(fraud_indices) > 0:
            idx = fraud_indices[0]
            local = explainer.local_explanation(X_test[idx : idx + 1])  # noqa: E203
            logger.info("\nLocal explanation for first fraud case:")
            for reason in local["reasons"]:
                logger.info(f"  - {reason}")
    else:
        logger.info("[PHASE 6] SHAP skipped (--skip-shap flag)")

    # ----------------------------------------------------------------
    # PHASE 7: Research Experiments
    # ----------------------------------------------------------------
    if not args.skip_experiments:
        logger.info("\n[PHASE 7] Research Experiments")
        runner = ExperimentRunner(X_train, y_train, X_test, y_test)
        exp_results = runner.run_all()
        logger.info(f"Completed {len(exp_results)} experiments.")
    else:
        logger.info("[PHASE 7] Experiments skipped (--skip-experiments flag)")

    logger.info("\n" + "=" * 60)
    logger.info("  Training Complete!")
    logger.info(f"  Models saved to: {cfg.model_dir}")
    logger.info(f"  Reports saved to: {cfg.report_dir}")
    logger.info("  Start API:       uvicorn api.main:app --reload")
    logger.info("  Start Dashboard: streamlit run dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
