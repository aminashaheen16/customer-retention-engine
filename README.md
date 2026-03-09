# 🎯 AI Customer Retention Engine

A production-ready AI system for customer churn prevention using behavioral segmentation.

---

## 🏗️ Architecture

```
customer_retention_engine/
├── data/
│   ├── data_pipeline.py        # ETL + EDA (Step 1)
│   ├── raw_telco.csv           # Generated on first run
│   ├── customers_segmented.csv # Output: labeled customers
│   └── plots/                  # All EDA & model charts
│
├── models/
│   ├── clustering_model.pkl    # KMeans (k=4)
│   ├── churn_model.pkl         # RandomForestClassifier
│   ├── scaler.pkl              # StandardScaler
│   ├── encoders.pkl            # LabelEncoders
│   └── feature_names.pkl       # Feature column list
│
├── services/
│   ├── clustering_service.py   # Segmentation (Step 2)
│   ├── churn_service.py        # Churn prediction (Step 3)
│   └── recommendation_service.py # Retention rules (Step 4)
│
├── api/
│   └── routes.py               # FastAPI REST API (Step 6)
│
├── dashboard/
│   └── app.py                  # Streamlit Dashboard (Step 7)
│
├── utils/
│   └── logger.py               # Centralized logging
│
├── main.py                     # Training pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models

```bash
python main.py
```

This will:
- Generate / load the Telco dataset (7,043 customers)
- Run full EDA and save plots to `data/plots/`
- Train KMeans clustering (k=4 segments)
- Train RandomForest churn predictor
- Save all artifacts to `models/`

### 3. Start the REST API

```bash
uvicorn api.routes:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at: http://localhost:8501

---

## 📡 API Reference

### POST `/analyze_customer`

Full analysis: segment + churn probability + recommendation.

**Request:**
```json
{
  "tenure": 12,
  "MonthlyCharges": 75.50,
  "TotalCharges": 906.0,
  "SeniorCitizen": 0,
  "gender": 1,
  "Partner": 0,
  "Dependents": 0,
  "PhoneService": 1,
  "PaperlessBilling": 1
}
```

**Response:**
```json
{
  "cluster_id": 1,
  "cluster_label": "At-Risk Customers",
  "churn_probability": 0.6732,
  "churn_risk": "High",
  "recommended_actions": [
    "Immediate discount offer (20%)",
    "Dedicated support contact",
    "Contract lock-in incentive"
  ],
  "priority": "Critical",
  "estimated_impact": "High likelihood of churn without action"
}
```

### POST `/predict_churn`
Returns churn probability and binary prediction only.

### POST `/segment_customer`
Returns behavioral segment only.

### GET `/health`
Returns service and model status.

### GET `/model_metrics`
Returns trained model evaluation metrics.

---

## 📊 Customer Segments

| Segment | Description | Strategy |
|---------|-------------|----------|
| **Loyal Customers** | Long tenure, consistent payments | Reward programs, VIP |
| **At-Risk Customers** | Short tenure, high monthly charges | Discounts, retention calls |
| **New Customers** | Recently joined | Onboarding, education |
| **High-Value Customers** | High spend, long tenure | Premium upgrades, exclusives |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~76% |
| AUC-ROC | ~0.75 |
| Recall (Churn) | ~50% |
| 5-Fold CV F1 | ~0.40 |

---

## 🔧 Tech Stack

- **ML**: scikit-learn (KMeans, RandomForest, PCA)
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit + Plotly
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Persistence**: joblib
- **Logging**: Python logging (file + console)
