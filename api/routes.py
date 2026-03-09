"""
FastAPI Backend — Step 6
Exposes the AI Customer Retention Engine via REST API.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from services.clustering_service import ClusteringService, CLUSTER_LABELS
from services.churn_service import ChurnService
from services.recommendation_service import RecommendationService
from utils.logger import get_logger

logger = get_logger("api")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Customer Retention Engine",
    description="""
## 🎯 Customer Retention API

Analyzes customer behavior, predicts churn, segments customers,
and recommends data-driven retention strategies.

### Endpoints
- **POST /analyze_customer** — Full analysis (segment + churn + recommendation)
- **POST /predict_churn** — Churn probability only
- **POST /segment_customer** — Cluster assignment only
- **GET /health** — Service health check
- **GET /model_metrics** — Trained model performance metrics
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_clustering_svc   = ClusteringService()
_churn_svc        = ChurnService()
_rec_svc          = RecommendationService()
_scaler           = None
_feature_names    = None

def _load_models():
    global _scaler, _feature_names
    try:
        _clustering_svc.load()
        _churn_svc.load()
        _scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        logger.info("✅ All models loaded successfully")
    except Exception as e:
        logger.error("⚠️ Model loading failed: %s", str(e))
        logger.warning("Run `python main.py` first to train models.")

_load_models()


# ── Request / Response schemas ─────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """Input features for a single customer."""
    tenure:           float = Field(..., ge=0,   le=72,  example=12,    description="Months with company")
    MonthlyCharges:   float = Field(..., ge=0,   le=200, example=65.5,  description="Monthly bill ($)")
    TotalCharges:     float = Field(..., ge=0,            example=786.0, description="Total billed ($)")
    SeniorCitizen:    int   = Field(0,  ge=0,   le=1,   example=0,     description="1 if senior citizen")
    # Optional extras (default to most common)
    gender:           int   = Field(0,  ge=0,   le=1,   example=0,     description="0=Female, 1=Male")
    Partner:          int   = Field(0,  ge=0,   le=1,   example=0)
    Dependents:       int   = Field(0,  ge=0,   le=1,   example=0)
    PhoneService:     int   = Field(1,  ge=0,   le=1,   example=1)
    PaperlessBilling: int   = Field(1,  ge=0,   le=1,   example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 75.50,
                "TotalCharges": 906.0,
                "SeniorCitizen": 0,
                "gender": 1,
                "Partner": 0,
                "Dependents": 0,
                "PhoneService": 1,
                "PaperlessBilling": 1,
            }
        }


class AnalysisResponse(BaseModel):
    cluster_id:           int
    cluster_label:        str
    churn_probability:    float
    churn_risk:           str
    recommended_actions:  list
    priority:             str
    estimated_impact:     str


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_risk:        str
    churn_predicted:   bool


class SegmentResponse(BaseModel):
    cluster_id:    int
    cluster_label: str


# ── Helper: build feature vector ───────────────────────────────────────────
def _build_feature_vector(customer: CustomerFeatures) -> np.ndarray:
    """
    Converts customer input into the model's expected feature vector.
    Missing one-hot columns are filled with 0.
    """
    if _feature_names is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    # Base numeric features (will be scaled)
    base = {
        "tenure":           customer.tenure,
        "MonthlyCharges":   customer.MonthlyCharges,
        "TotalCharges":     customer.TotalCharges,
        "SeniorCitizen":    customer.SeniorCitizen,
        "gender":           customer.gender,
        "Partner":          customer.Partner,
        "Dependents":       customer.Dependents,
        "PhoneService":     customer.PhoneService,
        "PaperlessBilling": customer.PaperlessBilling,
    }

    # Build row aligned to training feature names
    row = {feat: 0.0 for feat in _feature_names}
    for k, v in base.items():
        if k in row:
            row[k] = float(v)

    vec = np.array([row[f] for f in _feature_names], dtype=float)

    # Scale the numeric columns
    NUMERIC_IDX = [_feature_names.index(c) for c in
                   ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
                   if c in _feature_names]
    numeric_vals = vec[NUMERIC_IDX].reshape(1, -1)

    # We only have a partial vector — scale in place
    import pandas as _pd
    tmp = _pd.DataFrame([row], columns=_feature_names)
    from data.data_pipeline import NUMERIC_COLS
    existing = [c for c in NUMERIC_COLS if c in _feature_names]
    tmp[existing] = _scaler.transform(tmp[existing])
    return tmp.values[0]


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns service status and model readiness."""
    return {
        "status":            "healthy",
        "models_loaded":     _clustering_svc.model is not None,
        "churn_model":       _churn_svc.model is not None,
        "n_clusters":        4,
        "cluster_labels":    CLUSTER_LABELS,
    }


@app.get("/model_metrics", tags=["System"])
def model_metrics() -> Dict[str, Any]:
    """Returns trained model evaluation metrics."""
    return {
        "churn_model_metrics": _churn_svc.metrics,
        "cluster_labels":      CLUSTER_LABELS,
    }


@app.post("/analyze_customer", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_customer(customer: CustomerFeatures) -> AnalysisResponse:
    """
    Full customer analysis:
    1. Segments the customer (cluster)
    2. Predicts churn probability
    3. Returns targeted retention recommendation
    """
    try:
        vec        = _build_feature_vector(customer)
        cluster_id = _clustering_svc.predict(vec)
        churn_prob = _churn_svc.predict_proba(vec)
        rec        = _rec_svc.get_recommendation(cluster_id, churn_prob)

        logger.info("Analyzed customer | cluster=%d | churn_prob=%.4f | risk=%s",
                    cluster_id, churn_prob, rec.churn_risk)

        return AnalysisResponse(
            cluster_id=rec.cluster_id,
            cluster_label=rec.cluster_label,
            churn_probability=rec.churn_prob,
            churn_risk=rec.churn_risk,
            recommended_actions=rec.actions,
            priority=rec.priority,
            estimated_impact=rec.estimated_impact,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in analyze_customer: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_churn", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures) -> ChurnResponse:
    """Returns churn probability and binary prediction for a customer."""
    try:
        from services.recommendation_service import _classify_risk
        vec        = _build_feature_vector(customer)
        churn_prob = _churn_svc.predict_proba(vec)
        risk       = _classify_risk(churn_prob)
        return ChurnResponse(
            churn_probability=churn_prob,
            churn_risk=risk,
            churn_predicted=churn_prob >= 0.50,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment_customer", response_model=SegmentResponse, tags=["Segmentation"])
def segment_customer(customer: CustomerFeatures) -> SegmentResponse:
    """Returns the behavioral segment (cluster) for a customer."""
    try:
        vec        = _build_feature_vector(customer)
        cluster_id = _clustering_svc.predict(vec)
        return SegmentResponse(
            cluster_id=cluster_id,
            cluster_label=ClusteringService.get_label(cluster_id),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=True)
