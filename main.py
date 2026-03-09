"""
main.py — AI Customer Retention Engine
Orchestrates the full training pipeline:
  Data → Segmentation → Churn Model → Save Artifacts
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.data_pipeline import run_pipeline
from services.clustering_service import ClusteringService
from services.churn_service import ChurnService
from services.recommendation_service import RecommendationService
from utils.logger import get_logger

logger = get_logger("main")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_pipeline() -> None:
    """Full end-to-end training pipeline."""

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   AI Customer Retention Engine — Training    ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── Step 1: Data Pipeline ────────────────────────────────────────────────
    logger.info("\n▶ STEP 1 — Data Pipeline")
    X, y, scaler, encoders, df_clean = run_pipeline()

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)

    # Save encoders
    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")
    joblib.dump(encoders, enc_path)
    logger.info("Encoders saved → %s", enc_path)

    # Save feature names
    feat_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    joblib.dump(list(X.columns), feat_path)
    logger.info("Feature names saved → %s", feat_path)

    # ── Step 2: Clustering ────────────────────────────────────────────────────
    logger.info("\n▶ STEP 2 — Behavioral Segmentation")
    clustering_svc = ClusteringService(n_clusters=4)
    clustering_svc.plot_elbow(X)
    labels = clustering_svc.train(X)
    df_labeled, profile = clustering_svc.get_cluster_profiles(X, labels, df_clean)
    clustering_svc.plot_clusters_pca(X, labels)
    clustering_svc.plot_cluster_distribution(labels)
    clustering_svc.save()

    # Save labeled dataset
    labeled_path = os.path.join(os.path.dirname(__file__), "data", "customers_segmented.csv")
    df_labeled.to_csv(labeled_path, index=False)
    logger.info("Segmented dataset saved → %s", labeled_path)

    # ── Step 3: Churn Prediction ──────────────────────────────────────────────
    logger.info("\n▶ STEP 3 — Churn Prediction Model")
    churn_svc = ChurnService()
    metrics = churn_svc.train(X, y)
    churn_svc.save()

    logger.info("\n── Final Metrics ──────────────────────────────")
    for k, v in metrics.items():
        logger.info("  %-20s : %.4f", k, v)

    # ── Step 4: Validate Recommendations ─────────────────────────────────────
    logger.info("\n▶ STEP 4 — Recommendation Engine (smoke test)")
    rec_svc = RecommendationService()
    sample_x = X.iloc[0].values
    cluster_id  = clustering_svc.predict(sample_x)
    churn_prob  = churn_svc.predict_proba(sample_x)
    rec = rec_svc.get_recommendation(cluster_id, churn_prob)
    logger.info("Sample prediction:")
    logger.info("  Cluster   : %s", rec.cluster_label)
    logger.info("  Churn Prob: %.4f", rec.churn_prob)
    logger.info("  Risk      : %s", rec.churn_risk)
    logger.info("  Actions   : %s", rec.actions)

    logger.info("\n╔══════════════════════════════════════════════╗")
    logger.info("║           TRAINING COMPLETE ✅               ║")
    logger.info("╚══════════════════════════════════════════════╝")
    logger.info("Run the API:        uvicorn api.routes:app --reload")
    logger.info("Run the Dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    train_pipeline()
