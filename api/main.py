"""Phase 9: FastAPI Backend."""

from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.config import Config
from utils.logger import get_logger
from utils.metrics import risk_category

logger = get_logger("api")
cfg = Config()

app = FastAPI(
    title="Explainable Hybrid Fraud Detection API",
    description="Production-ready fraud detection with XGBoost, Isolation Forest & SHAP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_supervised = None
_anomaly = None


def _load_models():
    global _supervised, _anomaly
    sup_path = cfg.model_dir / "best_model.pkl"
    iso_path = cfg.model_dir / "isolation_forest.pkl"
    if sup_path.exists():
        _supervised = joblib.load(sup_path)
    if iso_path.exists():
        _anomaly = joblib.load(iso_path)


@app.on_event("startup")
async def startup():
    _load_models()
    logger.info("API started.")


class TransactionFeatures(BaseModel):
    features: List[float] = Field(..., description="Numeric feature vector")
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_score: float
    risk_category: str
    anomaly_score: float


class ExplainResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    top_reasons: List[str]
    shap_values: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supervised_model_loaded": _supervised is not None,
        "anomaly_model_loaded": _anomaly is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionFeatures):
    if _supervised is None:
        raise HTTPException(503, "Model not loaded. Run training first.")
    X = np.array(payload.features).reshape(1, -1)
    fp = float(_supervised.predict_proba(X)[0, 1])
    an = 0.0
    if _anomaly is not None:
        raw = -float(_anomaly.decision_function(X)[0])
        an = max(0.0, min(1.0, raw))
    rs = round(0.65 * fp * 100 + 0.35 * an * 100, 2)
    return PredictionResponse(
        fraud_probability=round(fp, 4),
        is_fraud=fp >= 0.5 or rs >= 75,
        risk_score=rs,
        risk_category=risk_category(rs),
        anomaly_score=round(an, 4),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: TransactionFeatures):
    if _supervised is None:
        raise HTTPException(503, "Model not loaded.")
    import shap

    X = np.array(payload.features).reshape(1, -1)
    fp = float(_supervised.predict_proba(X)[0, 1])
    try:
        explainer = shap.TreeExplainer(_supervised)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv.flatten()
    except Exception:
        sv = np.zeros(X.shape[1])
    names = payload.feature_names or [f"f{i}" for i in range(len(sv))]
    pairs = sorted(zip(names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
    top5 = pairs[:5]
    reasons = [f"{f} {'increases' if v > 0 else 'decreases'} fraud risk (SHAP={v:.4f})" for f, v in top5]
    rs = round(fp * 100, 2)
    return ExplainResponse(
        fraud_probability=round(fp, 4),
        risk_level=risk_category(rs),
        top_reasons=reasons,
        shap_values=[{"feature": f, "shap_value": round(v, 4)} for f, v in top5],
    )


@app.post("/risk-score")
def risk_score_endpoint(payload: TransactionFeatures):
    result = predict(payload)
    return {
        "risk_score": result.risk_score,
        "risk_category": result.risk_category,
        "fraud_probability": result.fraud_probability,
        "anomaly_score": result.anomaly_score,
    }
