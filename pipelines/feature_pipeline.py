"""Phase 2: Feature Engineering Pipeline."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from utils.logger import get_logger

logger = get_logger("feature_engineering")


class CreditCardFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for the Credit Card dataset."""

    def fit(self, X, y=None):
        self.amount_mean_ = X["Amount"].mean()
        self.amount_std_ = X["Amount"].std() + 1e-9
        return self

    def transform(self, X):
        X = X.copy()
        # Amount deviation score
        X["amount_deviation"] = (X["Amount"] - self.amount_mean_) / self.amount_std_
        # Log-transform amount
        X["log_amount"] = np.log1p(X["Amount"])
        # Time-based features
        if "Time" in X.columns:
            X["hour_of_day"] = (X["Time"] // 3600) % 24
            X["is_night"] = ((X["hour_of_day"] >= 22) | (X["hour_of_day"] <= 5)).astype(int)
        # High-value flag
        X["is_high_value"] = (X["Amount"] > X["Amount"].quantile(0.95)).astype(int)
        return X


class PaySimFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for the PaySim dataset."""

    def fit(self, X, y=None):
        self.amount_mean_ = X["amount"].mean()
        self.amount_std_ = X["amount"].std() + 1e-9
        return self

    def transform(self, X):
        X = X.copy()

        # Balance error features
        X["balance_error_orig"] = X["newbalanceOrig"] - (X["oldbalanceOrg"] - X["amount"])
        X["balance_error_dest"] = X["newbalanceDest"] - (X["oldbalanceDest"] + X["amount"])

        # Amount deviation
        X["amount_deviation"] = (X["amount"] - self.amount_mean_) / self.amount_std_
        X["log_amount"] = np.log1p(X["amount"])

        # Zero balance flags
        X["orig_zero_before"] = (X["oldbalanceOrg"] == 0).astype(int)
        X["dest_zero_before"] = (X["oldbalanceDest"] == 0).astype(int)
        X["orig_zero_after"] = (X["newbalanceOrig"] == 0).astype(int)

        # Transaction type encoding
        type_map = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
        X["type_encoded"] = X["type"].map(type_map).fillna(-1).astype(int)

        # High-risk type flag
        X["is_high_risk_type"] = X["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

        # Step-based time features
        X["hour_of_day"] = X["step"] % 24
        X["day_of_week"] = (X["step"] // 24) % 7

        # Customer risk score (ratio of amount to original balance)
        X["customer_risk_score"] = X["amount"] / (X["oldbalanceOrg"] + 1)

        # Merchant risk score (ratio of amount to dest balance)
        X["merchant_risk_score"] = X["amount"] / (X["oldbalanceDest"] + 1)

        # Drop non-numeric
        X.drop(columns=["type"], errors="ignore", inplace=True)
        return X


def build_creditcard_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("engineer", CreditCardFeatureEngineer()),
            ("scaler", RobustScaler()),
        ]
    )


def build_paysim_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("engineer", PaySimFeatureEngineer()),
            ("scaler", RobustScaler()),
        ]
    )


def get_feature_pipeline(dataset: str = "creditcard") -> Pipeline:
    if dataset == "creditcard":
        return build_creditcard_pipeline()
    return build_paysim_pipeline()
