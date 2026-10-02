"""Phase 1: Data Ingestion, EDA, and Preprocessing."""

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from utils.config import Config
from utils.logger import get_logger

logger = get_logger("data_ingest")
cfg = Config()


class DataIngestion:
    def __init__(self, dataset: str = "creditcard"):
        self.dataset = dataset
        self.df: pd.DataFrame = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        if self.dataset == "creditcard":
            path = cfg.data_dir / cfg.creditcard_file
        else:
            path = cfg.data_dir / cfg.paysim_file

        if not path.exists():
            logger.warning(f"{path} not found – generating synthetic data for demo.")
            self.df = self._synthetic_creditcard() if self.dataset == "creditcard" else self._synthetic_paysim()
        else:
            logger.info(f"Loading {path}")
            self.df = pd.read_csv(path)

        logger.info(f"Loaded {len(self.df):,} rows, {self.df.shape[1]} columns")
        return self.df

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    def eda(self) -> dict:
        df = self.df
        report = {
            "shape": list(df.shape),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "class_distribution": df[self._label_col()].value_counts().to_dict(),
            "describe": df.describe().round(4).to_dict(),
        }
        fraud_col = self._label_col()
        total = len(df)
        fraud_count = int(df[fraud_col].sum())
        report["fraud_rate_pct"] = round(fraud_count / total * 100, 4)
        logger.info(f"EDA complete. Fraud rate: {report['fraud_rate_pct']}%")

        # Save report
        out = cfg.report_dir / "eda_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"EDA report saved to {out}")
        return report

    # ------------------------------------------------------------------
    # 3. Clean
    # ------------------------------------------------------------------
    def clean(self) -> pd.DataFrame:
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        logger.info(f"Removed {before - len(self.df):,} duplicate/null rows")
        return self.df

    # ------------------------------------------------------------------
    # 4. Split + Imbalance Handling
    # ------------------------------------------------------------------
    def split(self, strategy: str = "smote"):
        label = self._label_col()
        drop_cols = self._drop_cols()
        feature_cols = [c for c in self.df.columns if c not in drop_cols + [label]]

        X = self.df[feature_cols].select_dtypes(include=[np.number])
        y = self.df[label]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )

        if strategy == "smote":
            logger.info("Applying SMOTE oversampling")
            sm = SMOTE(random_state=cfg.random_state, k_neighbors=3)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        elif strategy == "undersample":
            logger.info("Applying Random Under-sampling")
            rus = RandomUnderSampler(random_state=cfg.random_state)
            X_tr, y_tr = rus.fit_resample(X_tr, y_tr)
        elif strategy == "combined":
            logger.info("Applying combined SMOTE + Under-sampling")
            pipeline = ImbPipeline(
                [
                    ("smote", SMOTE(random_state=cfg.random_state, k_neighbors=3)),
                    ("under", RandomUnderSampler(random_state=cfg.random_state)),
                ]
            )
            X_tr, y_tr = pipeline.fit_resample(X_tr, y_tr)

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        self.feature_names = list(X.columns)

        logger.info(f"Train: {X_tr.shape}, Test: {X_te.shape}")
        logger.info(f"Train class dist: {dict(pd.Series(y_tr).value_counts())}")
        return X_tr, X_te, y_tr, y_te

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _label_col(self) -> str:
        return "Class" if self.dataset == "creditcard" else "isFraud"

    def _drop_cols(self) -> list:
        if self.dataset == "paysim":
            return ["nameOrig", "nameDest", "isFlaggedFraud"]
        return []

    def _synthetic_creditcard(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 10_000
        fraud_n = 200
        normal = pd.DataFrame(np.random.randn(n, 28), columns=[f"V{i}" for i in range(1, 29)])
        normal["Amount"] = np.abs(np.random.exponential(50, n))
        normal["Time"] = np.sort(np.random.randint(0, 172800, n))
        normal["Class"] = 0
        fraud = pd.DataFrame(np.random.randn(fraud_n, 28) * 2 + 3, columns=[f"V{i}" for i in range(1, 29)])
        fraud["Amount"] = np.abs(np.random.exponential(300, fraud_n))
        fraud["Time"] = np.random.randint(0, 172800, fraud_n)
        fraud["Class"] = 1
        return pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)

    def _synthetic_paysim(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 10_000
        fraud_n = 200
        types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        normal = pd.DataFrame(
            {
                "step": np.random.randint(1, 744, n),
                "type": np.random.choice(types, n),
                "amount": np.abs(np.random.exponential(200, n)),
                "nameOrig": [f"C{i}" for i in range(n)],
                "oldbalanceOrg": np.abs(np.random.exponential(5000, n)),
                "newbalanceOrig": np.abs(np.random.exponential(4800, n)),
                "nameDest": [f"M{i}" for i in range(n)],
                "oldbalanceDest": np.abs(np.random.exponential(3000, n)),
                "newbalanceDest": np.abs(np.random.exponential(3200, n)),
                "isFraud": 0,
                "isFlaggedFraud": 0,
            }
        )
        fraud = normal.sample(fraud_n, random_state=42).copy()
        fraud["isFraud"] = 1
        fraud["amount"] = np.abs(np.random.exponential(2000, fraud_n))
        return pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42)
