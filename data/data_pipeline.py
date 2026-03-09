"""
Data Pipeline — Step 1
Loads, cleans, encodes, normalizes the Telco Churn dataset
and produces EDA visualizations.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.dirname(__file__)
PLOTS_DIR  = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── synthetic dataset matching Telco schema ────────────────────────────────
def generate_synthetic_telco(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mirrors the IBM Telco Customer Churn
    schema (same columns, realistic distributions).
    """
    rng = np.random.default_rng(seed)

    genders        = rng.choice(["Male", "Female"], n)
    senior         = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner        = rng.choice(["Yes", "No"], n)
    dependents     = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure         = rng.integers(0, 72, n)
    phone_service  = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi_lines    = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n)
    )
    internet       = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_sec     = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    online_bkp     = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    device_prot    = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    tech_sup       = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    streaming_tv   = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    streaming_mv   = np.where(internet == "No", "No internet service",
                              rng.choice(["Yes", "No"], n))
    contract       = rng.choice(["Month-to-month", "One year", "Two year"],
                                n, p=[0.55, 0.21, 0.24])
    paperless      = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment        = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    monthly        = np.round(rng.uniform(18, 120, n), 2)

    # TotalCharges ~ tenure * monthly_charges (+ small noise), 11 blanks
    total = np.round(tenure * monthly + rng.normal(0, 50, n), 2)
    total = np.where(total < 0, 0, total)
    blank_idx = rng.choice(n, 11, replace=False)
    total_str = total.astype(str)
    total_str[blank_idx] = " "   # mimic real dataset blanks

    # Churn probability (higher for month-to-month, fiber, high monthly)
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.002 * (monthly - 18) / (120 - 18) * 20
        - 0.003 * tenure
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(n)]

    df = pd.DataFrame({
        "customerID":        customer_ids,
        "gender":            genders,
        "SeniorCitizen":     senior,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "MultipleLines":     multi_lines,
        "InternetService":   internet,
        "OnlineSecurity":    online_sec,
        "OnlineBackup":      online_bkp,
        "DeviceProtection":  device_prot,
        "TechSupport":       tech_sup,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_mv,
        "Contract":          contract,
        "PaperlessBilling":  paperless,
        "PaymentMethod":     payment,
        "MonthlyCharges":    monthly,
        "TotalCharges":      total_str,
        "Churn":             churn,
    })
    logger.info("Synthetic Telco dataset generated — shape %s", df.shape)
    return df


# ── loading ────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Loads dataset (synthetic fallback when network is unavailable)."""
    raw_path = os.path.join(DATA_DIR, "raw_telco.csv")
    if os.path.exists(raw_path):
        logger.info("Loading dataset from cache: %s", raw_path)
        return pd.read_csv(raw_path)
    logger.info("Generating synthetic Telco dataset …")
    df = generate_synthetic_telco()
    df.to_csv(raw_path, index=False)
    logger.info("Dataset saved → %s", raw_path)
    return df


# ── cleaning ───────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
    - Converts TotalCharges to numeric
    - Drops customerID (not predictive)
    - Fills missing values
    """
    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # TotalCharges can have blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna().sum()
    if missing:
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        logger.info("Filled %d missing TotalCharges with median", missing)

    logger.info("Data cleaned — shape %s", df.shape)
    return df


# ── encoding ───────────────────────────────────────────────────────────────
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]
MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def encode_features(df: pd.DataFrame):
    """
    Encodes categorical columns.
    Returns (encoded_df, label_encoders_dict).
    """
    df = df.copy()
    encoders = {}

    for col in BINARY_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True)
    logger.info("Encoding done — %d features", df.shape[1])
    return df, encoders


# ── scaling ────────────────────────────────────────────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Normalizes numeric columns using StandardScaler.
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    logger.info("Scaling applied to columns: %s", NUMERIC_COLS)
    return df, scaler


# ── EDA visualizations ─────────────────────────────────────────────────────
def run_eda(df_raw: pd.DataFrame) -> None:
    """Produces and saves EDA plots from the raw (pre-encoded) dataframe."""
    df = df_raw.copy()
    # Ensure TotalCharges numeric for EDA
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    churn_col = "Churn" if "Churn" in df.columns else None

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Churn distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_counts = df["Churn"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, churn_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Customer Churn Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Churn Status", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_ylim(0, churn_counts.max() * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"), dpi=150)
    plt.close()
    logger.info("Plot saved: churn_distribution.png")

    # 2. Tenure vs Churn
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
        subset = df[df["Churn"] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.7, label=f"Churn={label}", color=color, edgecolor="white")
    ax.set_title("Tenure Distribution by Churn", fontsize=15, fontweight="bold")
    ax.set_xlabel("Tenure (months)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tenure_vs_churn.png"), dpi=150)
    plt.close()
    logger.info("Plot saved: tenure_vs_churn.png")

    # 3. Monthly Charges vs Churn
    fig, ax = plt.subplots(figsize=(9, 5))
    data_to_plot = [
        df[df["Churn"] == "No"]["MonthlyCharges"].dropna(),
        df[df["Churn"] == "Yes"]["MonthlyCharges"].dropna(),
    ]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_xticklabels(["No Churn", "Churned"], fontsize=12)
    ax.set_title("Monthly Charges vs Churn", fontsize=15, fontweight="bold")
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "monthly_charges_vs_churn.png"), dpi=150)
    plt.close()
    logger.info("Plot saved: monthly_charges_vs_churn.png")

    # 4. Correlation heatmap (numeric only)
    df_num = df.select_dtypes(include=[np.number])
    if "Churn" in df.columns:
        df_num["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(df_num.corr(), dtype=bool))
    sns.heatmap(df_num.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    logger.info("Plot saved: correlation_heatmap.png")

    # 5. Contract type vs Churn
    fig, ax = plt.subplots(figsize=(9, 5))
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_churn.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"],
                        edgecolor="white", width=0.6)
    ax.set_title("Contract Type vs Churn", fontsize=15, fontweight="bold")
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(title="Churn", fontsize=10)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "contract_vs_churn.png"), dpi=150)
    plt.close()
    logger.info("Plot saved: contract_vs_churn.png")

    logger.info("✅ EDA complete — all plots saved to %s", PLOTS_DIR)


# ── full pipeline ──────────────────────────────────────────────────────────
def run_pipeline():
    """Executes the full data pipeline and returns processed artefacts."""
    logger.info("═══ DATA PIPELINE START ═══")
    df_raw     = load_data()
    df_clean   = clean_data(df_raw)
    run_eda(df_raw)

    df_encoded, encoders = encode_features(df_clean)

    # Separate target
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_scaled, scaler = scale_features(X)

    # Safety: fill any residual NaN
    X_scaled = X_scaled.fillna(0)

    logger.info("═══ DATA PIPELINE COMPLETE ═══")
    logger.info("X shape: %s | Churn rate: %.2f%%", X_scaled.shape, y.mean() * 100)
    return X_scaled, y, scaler, encoders, df_clean


if __name__ == "__main__":
    run_pipeline()
