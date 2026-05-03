# 🚀 Distributed ML Platform (Spark + Kafka + XGBoost + LightGBM)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Spark](https://img.shields.io/badge/Spark-Batch%20Processing-orange)
![Streaming](https://img.shields.io/badge/Streaming-Kafka-red)
![Boosting](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM-yellow)
![API](https://img.shields.io/badge/API-FastAPI-green)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-purple)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project implements a **production-grade distributed machine learning platform** combining:

- Spark batch feature engineering  
- Kafka real-time streaming ingestion  
- Online + offline feature pipelines  
- Multi-model training (LR, XGBoost, LightGBM)  
- Champion/Challenger deployment  
- MLflow experiment tracking  
- FastAPI serving layer  
- Docker + CI/CD  

---

## 🧠 Problem Statement

Design a system that:

- Handles batch + real-time data  
- Maintains feature consistency  
- Supports multiple models  
- Automates model promotion  
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

### Strategy
- Train multiple models
- Evaluate performance
- Promote best model

---

## 🐳 Docker

```bash
docker build -t ml-platform .
docker run -p 8000:8000 ml-platform
```

---

## 🔁 CI/CD

GitHub Actions pipeline:

- Install dependencies
- Generate sample data
- Run pytest
- Validate project

---

## 🔴 Kafka Streaming

### Start Kafka

```bash
docker-compose up kafka
```

### IMPORTANT

Use:

```yaml
image: apache/kafka:4.1.2
```

---

### Run consumer

```bash
python -m scripts.kafka_consumer
```

### Run producer

```bash
python -m scripts.kafka_producer
```

---

### Streaming Flow

```text
raw_events.csv → Kafka → Consumer → Online Features → Predictions
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

## 🔌 API Example

```json
{
  "event_value": 1.1,
  "amount": 10.0,
  "hour": 12,
  "is_mobile": 1,
  "lag_event_value_1": 1.0,
  "lag_amount_1": 9.5,
  "rolling_mean_event_value_5": 1.05,
  "rolling_mean_amount_5": 9.8,
  "rolling_std_event_value_5": 0.2,
  "rolling_mobile_rate_5": 0.6,
  "events_seen": 5
}
```

---

## 🔥 Key Highlights

- Distributed ML system  
- Real-time + batch processing  
- Multi-model training  
- Model lifecycle management  
- Kafka + Spark integration  
- Production-ready architecture  

---

## 🧠 Interview Talking Points

- Built end-to-end ML platform  
- Designed feature store system  
- Implemented streaming + batch pipelines  
- Added model promotion logic  
- Integrated MLOps tools (MLflow, CI/CD, Docker)  

---

## 📌 Author

Machine Learning + Quant + Systems Design Portfolio Project
