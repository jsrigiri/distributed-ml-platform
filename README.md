# 🚀 Distributed ML Platform (Spark + Online Features + XGBoost + LightGBM)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Spark](https://img.shields.io/badge/Spark-Batch%20Processing-orange)
![Boosting](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM-yellow)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project implements a **production-grade distributed machine learning platform** with:

- Batch feature engineering using Spark  
- Online feature generation (real-time stateful features)  
- Hybrid model architecture (baseline + boosting)  
- Model registry + promotion logic  
- FastAPI inference service  

Supports:

- Classification → event prediction  
- Regression → continuous target prediction  
- Models:
  - Logistic Regression  
  - XGBoost  
  - LightGBM  

---

## 🧠 Problem Statement

Build a scalable ML system that:

- Processes event streams  
- Maintains online + offline feature parity  
- Supports multiple model families  
- Automatically selects best model  
- Serves predictions with low latency  

---

## 🏗 Architecture

```text
Raw Events (CSV)
   ↓
Spark Batch Feature Pipeline
   ↓
Offline Feature Store (Parquet)
   ↓
Training Pipeline (Multi-Model)
   ↓
Model Registry
   ↓
FastAPI Service
   ↓
Online Feature Builder (Per-user state)
   ↓
Predictions + Metrics
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | PySpark, Pandas, NumPy |
| ML Models          | Scikit-learn, XGBoost, LightGBM |
| API                | FastAPI |
| Storage            | Parquet |
| Serialization      | Joblib |
| Testing            | Pytest |

---

## 📂 Project Structure

```text
distributed-ml-platform/
├── api/
├── core/
├── models/
├── pipelines/
├── store/
├── monitoring/
├── tests/
├── artifacts/
├── data/
├── main.py
├── generate_data.py
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### Baseline Model
- Logistic Regression  
- Fast and interpretable  

### Boosting Models
- XGBoost (Classifier + Regressor)  
- LightGBM (Classifier + Regressor)  
- Captures non-linear relationships  

### Model Selection
- Automatically selects best model based on accuracy  
- Stores all model metrics in registry  

---

## 📊 Metrics

### Classification
- Accuracy  
- Precision  
- Recall  
- F1 Score  

### Regression
- RMSE  

### System Metrics
- Prediction requests  
- Online feature updates  
- Model readiness  

---

## ⚡ Feature Engineering

### Batch (Spark)
- Lag features  
- Rolling statistics  
- Aggregations  

### Online
- Stateful per-user tracking  
- Rolling windows  
- Warmup logic  

---

## 🧪 Testing (Pytest)

Run:

```bash
pytest -v
```

### Coverage

- API endpoints  
- Feature pipelines  
- Model training  
- Model registry  

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Generate data

```bash
python generate_data.py
```

---

### 3. Run pipeline

```bash
python main.py
```

---

### 4. Start API

```bash
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Example

### Request

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

### Response

```json
{
  "prediction": 1,
  "probability_positive": 0.78,
  "model_ready": true
}
```

---

## 🔥 Key Highlights

- Distributed ML system with Spark  
- Hybrid feature store (online + offline)  
- Multi-model training (LR + XGBoost + LightGBM)  
- Automated model selection  
- Production-grade API  

---

## 🧠 Talking Points

- Built end-to-end distributed ML platform  
- Designed feature store architecture  
- Integrated gradient boosting models  
- Implemented model lifecycle management  
- Enabled real-time inference  

---

## 📌 Author

Machine Learning + Quant + Systems Design Portfolio Project
