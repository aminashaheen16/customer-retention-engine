# 🎯 AI Customer Retention Engine

A production-ready AI system for customer churn prevention using behavioral segmentation.

**Dataset:** [HuggingFace — aai510-group1/telco-customer-churn](https://huggingface.co/datasets/aai510-group1/telco-customer-churn)  
**GitHub:** [github.com/aminashaheen16/customer-retention-engine](https://github.com/aminashaheen16/customer-retention-engine)

---

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 96.3% |
| AUC-ROC | 0.991 |
| F1 Score | 0.931 |
| Recall | 93.7% |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install datasets huggingface_hub
```

### 2. Download real dataset
```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('aai510-group1/telco-customer-churn')
ds['train'].to_csv('data/raw_telco.csv', index=False)
print('Dataset downloaded!')
"
```

### 3. Train all models
```bash
PYTHONPATH=$(pwd) python main.py
```

### 4. Launch Dashboard
```bash
PYTHONPATH=$(pwd) streamlit run dashboard/app.py
```
Dashboard: http://localhost:8501

### 5. Start API
```bash
PYTHONPATH=$(pwd) uvicorn api.routes:app --reload --port 8000
```
API Docs: http://localhost:8000/docs

---

## 📊 Customer Segments

| Segment | Count | Churn Rate |
|---------|-------|------------|
| New Customers | 1,907 (45.1%) | 45.0% ⚠️ |
| High-Value Customers | 1,235 (29.2%) | 13.4% ✅ |
| At-Risk Customers | 881 (20.9%) | 6.5% |
| Loyal Customers | 202 (4.8%) | 20.3% |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze_customer` | Full analysis: segment + churn + recommendation |
| POST | `/predict_churn` | Churn probability only |
| POST | `/segment_customer` | Behavioral segment only |
| GET | `/health` | Service status |

