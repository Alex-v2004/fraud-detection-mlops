# 🛡️ Fraud Detection MLOps

Real-time credit card fraud detection API — built with production MLOps practices.

## 🎯 Problem
Detect fraudulent credit card transactions in real-time using a trained XGBoost classifier, served via a REST API.

## 🏗️ Architecture
```
Data (Kaggle) → Preprocess → Train (XGBoost) → MLflow → FastAPI → Docker → AWS EC2 → Grafana
```

## 🛠️ Tech Stack
| Layer | Tool |
|---|---|
| Model | XGBoost + scikit-learn |
| Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2 + S3 |
| Monitoring | Prometheus + Grafana |
| Container | Docker |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/fraud-detection-mlops.git
cd fraud-detection-mlops

# Install
pip install -r requirements.txt

# Run API
uvicorn src.api.main:app --reload
```

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.8, ...]}'
```

Response:
```json
{
  "is_fraud": false,
  "confidence": 0.03,
  "model_version": "v1"
}
```

## 📁 Project Structure
```
fraud-detection-mlops/
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Load + validate dataset
│   │   └── preprocess.py       # Feature engineering + split
│   ├── model/
│   │   └── train.py            # XGBoost training + MLflow logging
│   └── api/
│       └── main.py             # FastAPI app
├── tests/                      # pytest test suite
├── docker/                     # Dockerfile + compose
├── notebooks/                  # EDA notebooks
└── requirements.txt
```

## 📊 Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |

---
