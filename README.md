# 🛡️ Fraud Detection MLOps

> Real-time credit card fraud detection API — built with production MLOps practices.

[![CI](https://github.com/Alex-v2004/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Alex-v2004/fraud-detection-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)

## 🎯 Problem
Credit card fraud costs billions annually. This project builds a **real-time fraud detection API** that predicts whether a transaction is fraudulent in milliseconds.

- Dataset: 284,807 transactions, only 0.17% fraud (heavily imbalanced!)
- Model: XGBoost with class weight balancing
- Served via REST API — plug into any payment system

## 🏗️ Architecture
```
Kaggle Data → Preprocess → XGBoost Train → MLflow Track → FastAPI → Docker → AWS EC2 → Grafana
```

## 🛠️ Tech Stack
| Layer | Tool | Purpose |
|---|---|---|
| Model | XGBoost + scikit-learn | Fraud classification |
| Tracking | MLflow | Experiment tracking & model registry |
| Data Versioning | DVC | Dataset versioning |
| API | FastAPI | Real-time predictions |
| Testing | pytest | 5/5 tests passing |
| CI/CD | GitHub Actions | Auto test + build on every push |
| Container | Docker | Portable deployment |
| Cloud | AWS EC2 + S3 | Live hosting |
| Monitoring | Prometheus + Grafana | Model health dashboard |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Alex-v2004/fraud-detection-mlops.git
cd fraud-detection-mlops

# Install
pip install -r requirements.txt

# Train model
python src/model/train.py

# Run API locally
uvicorn src.api.main:app --reload
```

## 🐳 Run with Docker

```bash
# Build image
docker build -t fraud-detection:v1 .

# Run container
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" fraud-detection:v1

# Windows CMD users:
docker run -p 8000:8000 -v "%CD%/models:/app/models" fraud-detection:v1
```

API will be live at `http://localhost:8000` 🚀

## 📡 API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict fraud:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.8, ...]}'
```

**Response:**
```json
{
  "is_fraud": false,
  "confidence": 0.03,
  "model_version": "v1"
}
```

Or visit `http://localhost:8000/docs` for interactive Swagger UI 🎯

## 📊 Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | ~0.98 |
| Precision | ~0.95 |
| Recall | ~0.82 |
| F1 Score | ~0.88 |

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
├── tests/
│   └── test_api.py             # pytest test suite (5/5 passing)
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── Dockerfile                  # Container definition
├── notebooks/                  # EDA notebooks
└── requirements.txt

```

## 🗺️ Roadmap
- [x] Modular Python scripts with logging
- [x] XGBoost model training
- [x] MLflow experiment tracking
- [x] FastAPI REST API
- [x] Docker containerization
- [x] GitHub Actions CI/CD
- [ ] AWS EC2 deployment
- [ ] Prometheus + Grafana monitoring
- [ ] Kubernetes scaling

---
