from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    experiment_dir: Path = Path("experiments")

    # Dataset
    creditcard_file: str = "creditcard.csv"
    paysim_file: str = "PS_20174392719_1491204439457_log.csv"

    # Training
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Isolation Forest
    contamination: float = 0.01
    n_estimators_if: int = 200

    # XGBoost
    n_estimators_xgb: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    scale_pos_weight: float = 100.0

    # Risk thresholds
    low_threshold: float = 0.25
    medium_threshold: float = 0.50
    high_threshold: float = 0.75

    # SHAP
    shap_max_display: int = 20

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "fraud_detection"

    def __post_init__(self):
        for d in [self.data_dir, self.model_dir, self.report_dir, self.experiment_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
