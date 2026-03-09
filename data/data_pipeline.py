"""
Data Pipeline — Step 1 (Real HuggingFace Dataset)
"""
import os, sys, warnings
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

DATA_DIR  = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

NUMERIC_COLS = [
    "Tenure in Months", "Monthly Charge", "Total Charges", "Total Revenue",
    "Total Refunds", "Total Extra Data Charges", "Total Long Distance Charges",
    "Avg Monthly GB Download", "Avg Monthly Long Distance Charges",
    "Number of Referrals", "Number of Dependents", "Satisfaction Score",
    "CLTV", "Age",
]

CATEGORICAL_COLS = [
    "Gender", "Senior Citizen", "Married", "Dependents", "Partner",
    "Phone Service", "Multiple Lines", "Internet Service", "Internet Type",
    "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music",
    "Unlimited Data", "Paperless Billing", "Payment Method", "Contract",
    "Referred a Friend", "Offer",
]

SELECTED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ["Churn"]


def load_data() -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, "raw_telco.csv")
    if os.path.exists(raw_path):
        logger.info("Loading real Telco dataset — %s", raw_path)
        df = pd.read_csv(raw_path)
        logger.info("Loaded shape: %s", df.shape)
        return df
    raise FileNotFoundError(f"Dataset not found: {raw_path}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    available = [c for c in SELECTED_COLS if c in df.columns]
    df = df[available]
    if df["Churn"].dtype != object:
        df["Churn"] = df["Churn"].map({1:"Yes", 0:"No", True:"Yes", False:"No"})
    df["Churn"] = df["Churn"].fillna("No")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0]).astype(str)
    logger.info("Data cleaned — shape %s", df.shape)
    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    le = LabelEncoder()
    df["Churn"] = le.fit_transform(df["Churn"].astype(str))
    encoders["Churn"] = le
    cat_available = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_available, drop_first=True)
    logger.info("Encoding done — %d features", df.shape[1])
    return df, encoders


def scale_features(df: pd.DataFrame, scaler=None):
    df = df.copy()
    num_available = [c for c in NUMERIC_COLS if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[num_available] = scaler.fit_transform(df[num_available])
    else:
        df[num_available] = scaler.transform(df[num_available])
    return df, scaler


def run_eda(df: pd.DataFrame) -> None:
    df = df.copy()
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["Churn"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=["#2ecc71","#e74c3c"], width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Customer Churn Distribution", fontsize=15, fontweight="bold")
    ax.set_ylim(0, counts.max()*1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"), dpi=150)
    plt.close()

    if "Tenure in Months" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, color in [("No","#2ecc71"),("Yes","#e74c3c")]:
            ax.hist(df[df["Churn"]==label]["Tenure in Months"].dropna(),
                    bins=30, alpha=0.7, label=f"Churn={label}", color=color)
        ax.set_title("Tenure vs Churn", fontsize=15, fontweight="bold")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "tenure_vs_churn.png"), dpi=150)
        plt.close()

    if "Monthly Charge" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot([df[df["Churn"]=="No"]["Monthly Charge"].dropna(),
                         df[df["Churn"]=="Yes"]["Monthly Charge"].dropna()],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        ax.set_xticklabels(["No Churn","Churned"])
        ax.set_title("Monthly Charge vs Churn", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "monthly_charges_vs_churn.png"), dpi=150)
        plt.close()

    num_df = df[[c for c in NUMERIC_COLS if c in df.columns]].copy()
    num_df["Churn_bin"] = (df["Churn"]=="Yes").astype(int)
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if "Contract" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        df.groupby(["Contract","Churn"]).size().unstack(fill_value=0).plot(
            kind="bar", ax=ax, color=["#2ecc71","#e74c3c"], edgecolor="white")
        ax.set_title("Contract Type vs Churn", fontsize=15, fontweight="bold")
        plt.xticks(rotation=20); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "contract_vs_churn.png"), dpi=150)
        plt.close()

    logger.info("✅ EDA complete — plots saved to %s", PLOTS_DIR)


def run_pipeline():
    logger.info("═══ DATA PIPELINE START ═══")
    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    run_eda(df_clean)
    df_enc, encoders = encode_features(df_clean)
    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]
    X_scaled, scaler = scale_features(X)
    X_scaled = X_scaled.fillna(0)
    logger.info("═══ PIPELINE COMPLETE ═══")
    logger.info("X shape: %s | Churn rate: %.2f%%", X_scaled.shape, y.mean()*100)
    return X_scaled, y, scaler, encoders, df_clean

if __name__ == "__main__":
    run_pipeline()
