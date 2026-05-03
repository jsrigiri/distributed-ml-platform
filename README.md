# 🚀 Distributed ML Platform (Spark + Kafka + MLflow + XGBoost + LightGBM)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Spark](https://img.shields.io/badge/Spark-Batch%20Processing-orange)
![Streaming](https://img.shields.io/badge/Streaming-Kafka-red)
![ML](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM-yellow)
![API](https://img.shields.io/badge/API-FastAPI-green)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-purple)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project implements a **production-grade distributed machine learning platform** combining:

- Spark batch feature engineering  
- Kafka real-time streaming ingestion  
- Online + offline feature pipelines  
- Multi-model training (Logistic Regression, XGBoost, LightGBM)  
- Champion/Challenger deployment  
- MLflow experiment tracking  
- Real-time drift detection  
- FastAPI serving layer  
- Docker + CI/CD  

---

## 🧠 Problem Statement

Build a system that:

- Handles batch + real-time data  
- Maintains feature consistency  
- Supports multiple models  
- Automatically promotes models  
- Detects drift in real-time  
- Serves predictions with low latency  

---

## 🏗 Architecture

```text
Raw Events (CSV)
   ↓
Spark Batch Pipeline
   ↓
Offline Feature Store (Parquet)
   ↓
Training (MLflow + Multi-Model)
   ↓
Model Registry (Champion/Challenger)
   ↓
FastAPI API
   ↓
Online Feature Builder
   ↓
Kafka Streaming Layer
   ↓
Drift Detection (Realtime + MLflow Logging)
   ↓
Predictions + Metrics
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|------|------|
| Batch Processing | PySpark |
| Streaming | Kafka |
| ML | Scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow |
| API | FastAPI |
| Storage | Parquet |
| DevOps | Docker, GitHub Actions |
| Testing | Pytest |

---

## 🧠 Models

### Baseline
- Logistic Regression

### Advanced
- XGBoost (Classifier + Regressor)
- LightGBM (Classifier + Regressor)

---

## 🔁 Champion / Challenger System

- New models saved as **challenger**
- Promotion happens only if better than champion
- Safe production rollout strategy

---

## 📊 MLflow Tracking

```bash
mlflow ui
```

Tracks:
- Training metrics
- Drift metrics
- Model artifacts

---

## 🔴 Real-Time Drift Detection

### API-based drift

```text
/predict → Drift Check → MLflow Logging
```

### Kafka-based drift monitoring

```bash
python -m scripts.kafka_drift_monitor
```

---

## 🐳 Docker + Kafka

Start Kafka:

```bash
docker-compose up kafka
```

⚠️ Use:

```yaml
image: apache/kafka:4.1.2
```

---

## 🔁 Kafka Streaming

```bash
python -m scripts.kafka_producer
python -m scripts.kafka_drift_monitor
```

---

## 🧪 Testing

```bash
pytest -v
```

---

## ▶️ Run System

```bash
python generate_data.py
python main.py
uvicorn api.app:app --reload
```

---

## 🔥 Key Highlights

- Distributed ML system  
- Real-time + batch pipelines  
- Kafka + Spark integration  
- Multi-model training  
- Automated model promotion  
- Drift detection + MLflow logging  

---

## 📌 Author

Machine Learning + Quant + Systems Design Portfolio Project
