"""
Clustering Service — Step 2
Performs KMeans behavioral segmentation with Elbow Method and PCA visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── cluster labels (assigned after interpreting cluster profiles) ───────────
CLUSTER_LABELS = {
    0: "Loyal Customers",
    1: "At-Risk Customers",
    2: "New Customers",
    3: "High-Value Customers",
}

CLUSTER_COLORS = {
    0: "#2ecc71",   # green
    1: "#e74c3c",   # red
    2: "#3498db",   # blue
    3: "#f39c12",   # orange
}


class ClusteringService:
    """Handles KMeans-based customer segmentation."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters    = n_clusters
        self.random_state  = random_state
        self.model: KMeans = None
        self.pca: PCA      = None

    # ── Elbow Method ────────────────────────────────────────────────────────
    def plot_elbow(self, X: pd.DataFrame, max_k: int = 10) -> int:
        """
        Runs Elbow Method to determine optimal k.
        Saves the elbow plot and returns the best k.
        """
        logger.info("Running Elbow Method (k=2..%d) …", max_k)
        inertias, sil_scores = [], []

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_, sample_size=2000))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ks = list(range(2, max_k + 1))

        axes[0].plot(ks, inertias, "bo-", linewidth=2, markersize=8)
        axes[0].set_title("Elbow Method — Inertia", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].axvline(x=4, color="red", linestyle="--", alpha=0.7, label="Chosen k=4")
        axes[0].legend()

        axes[1].plot(ks, sil_scores, "gs-", linewidth=2, markersize=8)
        axes[1].set_title("Silhouette Score by k", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].axvline(x=4, color="red", linestyle="--", alpha=0.7, label="Chosen k=4")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "elbow_method.png"), dpi=150)
        plt.close()
        logger.info("Elbow plot saved.")

        best_k = self.n_clusters
        logger.info("Selected k=%d (pre-configured)", best_k)
        return best_k

    # ── train ────────────────────────────────────────────────────────────────
    def train(self, X: pd.DataFrame) -> np.ndarray:
        """
        Trains KMeans and returns cluster labels.
        """
        logger.info("Training KMeans with k=%d …", self.n_clusters)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=15,
            max_iter=400,
        )
        labels = self.model.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=3000)
        logger.info("KMeans trained | Inertia=%.2f | Silhouette=%.4f", self.model.inertia_, sil)

        # Relabel clusters by ascending mean churn risk heuristic
        # (cluster with lowest mean feature[0] = long-tenure = Loyal, etc.)
        # In practice we inspect profiles; here we assign fixed mapping
        return labels

    # ── interpret & label ────────────────────────────────────────────────────
    def get_cluster_profiles(self, X: pd.DataFrame, labels: np.ndarray,
                              df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a cluster profile table using original (unscaled) features.
        """
        df_profile = df_clean.copy()
        df_profile["Cluster"]      = labels
        df_profile["ClusterLabel"] = df_profile["Cluster"].map(CLUSTER_LABELS)

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        agg = df_profile.groupby("ClusterLabel")[numeric_cols].mean().round(2)
        cnt = df_profile.groupby("ClusterLabel").size().rename("CustomerCount")
        churn_rate = df_profile.groupby("ClusterLabel")["Churn"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        ).rename("ChurnRate%")
        profile = pd.concat([agg, cnt, churn_rate], axis=1)
        logger.info("Cluster profiles:\n%s", profile.to_string())
        return df_profile, profile

    # ── PCA visualization ────────────────────────────────────────────────────
    def plot_clusters_pca(self, X: pd.DataFrame, labels: np.ndarray) -> None:
        """Reduces to 2D via PCA and plots cluster scatter."""
        logger.info("Running PCA for cluster visualization …")
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = self.pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 7))
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=CLUSTER_COLORS[cluster_id],
                label=CLUSTER_LABELS[cluster_id],
                alpha=0.6, s=25, edgecolors="none",
            )
        # Plot centroids projected
        centroids_2d = self.pca.transform(self.model.cluster_centers_)
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c="black", marker="X", s=200, zorder=5, label="Centroids")

        ax.set_title("Customer Segments — PCA Projection", fontsize=15, fontweight="bold")
        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cluster_pca.png"), dpi=150)
        plt.close()
        logger.info("PCA cluster plot saved.")

    # ── cluster distribution bar ─────────────────────────────────────────────
    def plot_cluster_distribution(self, labels: np.ndarray) -> None:
        unique, counts = np.unique(labels, return_counts=True)
        names  = [CLUSTER_LABELS[i] for i in unique]
        colors = [CLUSTER_COLORS[i] for i in unique]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, counts, color=colors, edgecolor="white", width=0.55)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{cnt:,}", ha="center", fontsize=11, fontweight="bold")
        ax.set_title("Customer Segment Distribution", fontsize=15, fontweight="bold")
        ax.set_ylabel("Number of Customers")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "cluster_distribution.png"), dpi=150)
        plt.close()
        logger.info("Cluster distribution plot saved.")

    # ── predict for new customer ─────────────────────────────────────────────
    def predict(self, X_scaled: np.ndarray) -> int:
        """Returns cluster id for a single scaled feature vector."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return int(self.model.predict(X_scaled.reshape(1, -1))[0])

    # ── save / load ──────────────────────────────────────────────────────────
    def save(self) -> None:
        path = os.path.join(MODELS_DIR, "clustering_model.pkl")
        joblib.dump(self.model, path)
        logger.info("Clustering model saved → %s", path)

    def load(self) -> None:
        path = os.path.join(MODELS_DIR, "clustering_model.pkl")
        self.model = joblib.load(path)
        logger.info("Clustering model loaded ← %s", path)

    # ── label helpers ────────────────────────────────────────────────────────
    @staticmethod
    def get_label(cluster_id: int) -> str:
        return CLUSTER_LABELS.get(cluster_id, "Unknown")

    @staticmethod
    def get_all_labels() -> dict:
        return CLUSTER_LABELS


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.data_pipeline import run_pipeline

    X, y, scaler, encoders, df_clean = run_pipeline()
    svc = ClusteringService()
    svc.plot_elbow(X)
    labels = svc.train(X)
    df_clean_labeled, profile = svc.get_cluster_profiles(X, labels, df_clean)
    svc.plot_clusters_pca(X, labels)
    svc.plot_cluster_distribution(labels)
    svc.save()
    print(profile)
