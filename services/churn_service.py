"""
Churn Prediction Service — Step 3
Trains a RandomForestClassifier, evaluates it, and exposes churn probability.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


class ChurnService:
    """Manages churn prediction model lifecycle."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: RandomForestClassifier = None
        self.feature_names: list = []
        self.metrics: dict = {}

    # ── training ─────────────────────────────────────────────────────────────
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.20) -> dict:
        """
        Trains RandomForestClassifier and returns evaluation metrics.
        """
        self.feature_names = list(X.columns)
        X_arr = X.values
        y_arr = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size,
            random_state=self.random_state, stratify=y_arr,
        )

        logger.info("Training RandomForestClassifier …")
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
        }

        logger.info("── Model Evaluation ──")
        for k, v in self.metrics.items():
            logger.info("  %s : %.4f", k.upper(), v)
        logger.info(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_arr, y_arr, cv=StratifiedKFold(5),
            scoring="f1", n_jobs=-1,
        )
        logger.info("5-Fold CV F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
        self.metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        self.metrics["cv_f1_std"]  = round(cv_scores.std(), 4)

        # Plots
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_proba)
        self._plot_feature_importance(top_n=15)

        return self.metrics

    # ── prediction ───────────────────────────────────────────────────────────
    def predict_proba(self, X_scaled: np.ndarray) -> float:
        """Returns churn probability for a single customer (0‒1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        proba = self.model.predict_proba(X_scaled.reshape(1, -1))[0][1]
        return round(float(proba), 4)

    def predict(self, X_scaled: np.ndarray) -> int:
        """Returns binary churn prediction (0 or 1)."""
        return int(self.model.predict(X_scaled.reshape(1, -1))[0])

    # ── plots ─────────────────────────────────────────────────────────────────
    def _plot_confusion_matrix(self, y_true, y_pred) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"],
                    linewidths=0.5, ax=ax)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
        plt.close()
        logger.info("Confusion matrix plot saved.")

    def _plot_roc_curve(self, y_true, y_proba) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
                label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_title("ROC Curve — Churn Prediction", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
        plt.close()
        logger.info("ROC curve plot saved.")

    def _plot_feature_importance(self, top_n: int = 15) -> None:
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        features = [self.feature_names[i] for i in indices]
        values   = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=12)
        for bar, val in zip(bars, values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        logger.info("Feature importance plot saved.")

    # ── save / load ──────────────────────────────────────────────────────────
    def save(self) -> None:
        path = os.path.join(MODELS_DIR, "churn_model.pkl")
        joblib.dump({"model": self.model, "features": self.feature_names,
                     "metrics": self.metrics}, path)
        logger.info("Churn model saved → %s", path)

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, "churn_model.pkl")
        bundle = joblib.load(path)
        self.model         = bundle["model"]
        self.feature_names = bundle["features"]
        self.metrics       = bundle.get("metrics", {})
        logger.info("Churn model loaded ← %s", path)


if __name__ == "__main__":
    from data.data_pipeline import run_pipeline
    X, y, scaler, encoders, df_clean = run_pipeline()
    svc = ChurnService()
    metrics = svc.train(X, y)
    svc.save()
    print("Metrics:", metrics)
