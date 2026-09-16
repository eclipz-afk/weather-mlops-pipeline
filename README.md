# Weather Temperature Forecasting — MLOps Pipeline

End-to-end MLOps pipeline for **1-hour-ahead temperature forecasting** using historical weather observations.

The project implements the complete machine learning lifecycle:

* weather data ingestion from Open-Meteo;
* data validation with Pandera;
* raw and processed data storage in MinIO;
* feature engineering for time-series forecasting;
* chronological train / validation / test splitting;
* Random Forest model training;
* experiment tracking with MLflow;
* model registration and promotion with MLflow Model Registry;
* prediction serving through FastAPI;
* web interface with Streamlit;
* workflow orchestration with Apache Airflow;
* containerized deployment with Docker Compose.

---

## Architecture

```text
Open-Meteo
    │
    ▼
Data Ingestion
    │
    ▼
Pandera Validation
    │
    ▼
MinIO
    │
    ▼
Feature Engineering
    │
    ▼
ML Dataset
(train / val / test)
    │
    ▼
RandomForestRegressor
    │
    ▼
MLflow
    │
    ▼
Model Registry
    │
    ▼
champion
    │
    ▼
FastAPI
    │
    ▼
Streamlit
```

Apache Airflow orchestrates recurring data ingestion and model retraining.

---

## Forecasting Task

The model predicts air temperature **one hour into the future**.

For an observation timestamp `t`, the prediction target is:

```text
temperature_2m(t + 1 hour)
```

Features are constructed only from information available at or before timestamp `t`.

This keeps the forecasting setup causal and prevents future observations from leaking into model training.

---

## Technology Stack

| Component              | Technology                         |
| ---------------------- | ---------------------------------- |
| Language               | Python                             |
| Data processing        | Pandas                             |
| Data validation        | Pandera                            |
| ML model               | Scikit-learn RandomForestRegressor |
| Weather source         | Open-Meteo                         |
| Object storage         | MinIO                              |
| Experiment tracking    | MLflow                             |
| Model registry         | MLflow Model Registry              |
| Workflow orchestration | Apache Airflow                     |
| Prediction API         | FastAPI                            |
| Web UI                 | Streamlit                          |
| Airflow metadata DB    | PostgreSQL                         |
| Containerization       | Docker / Docker Compose            |

---

## Repository Structure

```text
weather-mlops-pipeline/
│
├── weather_forecasting/
│   ├── datasets/
│   │   ├── audit.py
│   │   ├── backfill.py
│   │   ├── cleaning.py
│   │   ├── features.py
│   │   ├── splitting.py
│   │   ├── storage.py
│   │   ├── validation.py
│   │   └── weather_api.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── mlflow_utils.py
│   │   └── registry.py
│   │
│   ├── pipelines/
│   │   ├── prepare_training_data.py
│   │   └── train_pipeline.py
│   │
│   ├── serving/
│   │   └── service.py
│   │
│   └── config.py
│
├── services/
│   ├── airflow/
│   │   ├── Dockerfile
│   │   └── dags/
│   │       ├── weather_ingestion.py
│   │       └── weather_retraining.py
│   │
│   ├── api/
│   │   ├── Dockerfile
│   │   └── main.py
│   │
│   ├── app/
│   │   ├── Dockerfile
│   │   └── app.py
│   │
│   └── mlflow/
│       └── Dockerfile
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# Quick Start

## Prerequisites

Install:

* Git;
* Docker Desktop or Docker Engine;
* Docker Compose v2.

Check the installation:

```bash
git --version
docker --version
docker compose version
```

The complete stack contains Airflow, MLflow, MinIO, PostgreSQL, FastAPI and Streamlit.

Allocating approximately **4 GB of memory to Docker/WSL** is recommended for running all services simultaneously.

---

## 1. Clone the Repository

```bash
git clone https://github.com/eclipz-afk/weather-mlops-pipeline.git
cd weather-mlops-pipeline
```

All commands below are expected to be executed from the repository root.

---

## 2. Configure Environment Variables

Copy the example configuration.

### Linux / macOS

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Review `.env` before starting the project.

It contains configuration for MinIO, S3-compatible storage, MLflow, Airflow, PostgreSQL and application services.

Do not commit `.env` to Git.

---

## 3. Start the Platform

Build and start the services:

```bash
docker compose up -d --build
```

Check their state:

```bash
docker compose ps
```

The first startup may take longer because Docker needs to build or download the required images and initialize persistent services.

---

## 4. Initialize the Project

A clean installation requires historical weather data, a prepared ML dataset and an initial registered model.

### Load historical observations

Example:

```bash
python -m weather_forecasting.datasets.backfill \
  --start-date 2000-01-01 \
  --end-date 2026-09-16
```

This downloads historical weather observations, validates and processes them, and stores the resulting data in MinIO.

### Prepare the ML dataset

```bash
python -m weather_forecasting.pipelines.prepare_training_data \
  --start-date 2000-01-01 \
  --end-date 2026-09-16 \
  --dataset-id weather-kazan-2000-2026-v1
```

Use the same historical interval that was loaded during backfill.

The command performs feature engineering and publishes the chronological train, validation and test splits under the specified dataset ID.

### Train the initial model

```bash
python -m weather_forecasting.pipelines.train_pipeline \
  --dataset-id weather-kazan-2000-2026-v1
```

The training run and model artifact are recorded in MLflow.

### Initialize the serving model

Open Airflow:

```text
http://localhost:8080
```

Trigger the:

```text
weather_retraining
```

DAG.

The retraining workflow evaluates the candidate model and, when it passes the quality gate, registers it and assigns the `champion` alias used by the prediction service.

---

## 5. Normal Startup

After the initial data and model artifacts have been created, start the project with:

```bash
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Airflow handles subsequent scheduled ingestion and retraining.

---

## 6. Service URLs

| Service         | Address                        |
| --------------- | ------------------------------ |
| Streamlit       | `http://localhost:8501`        |
| FastAPI         | `http://localhost:8000`        |
| FastAPI Swagger | `http://localhost:8000/docs`   |
| FastAPI health  | `http://localhost:8000/health` |
| Airflow         | `http://localhost:8080`        |
| MLflow          | `http://localhost:5000`        |
| MinIO Console   | `http://localhost:9001`        |

Airflow and MinIO credentials are configured through `.env`.

---

## 7. Stop the Platform

```bash
docker compose down
```

Persistent Docker volumes remain available for the next startup.

Do not delete the volumes unless a complete reset of datasets, MLflow artifacts, registered models and service state is intended.

---

# Data Pipeline

Historical weather observations are retrieved from Open-Meteo.

The ingestion flow is:

```text
Open-Meteo
    │
    ▼
Raw observations
    │
    ▼
Pandera validation
    │
    ▼
Data cleaning
    │
    ▼
Processed observations
    │
    ▼
MinIO
```

Raw and processed datasets are stored separately.

MinIO is also used for prepared ML datasets and MLflow model artifacts.

Recurring ingestion is orchestrated by the Airflow DAG:

```text
weather_ingestion
```

defined in:

```text
services/airflow/dags/weather_ingestion.py
```

---

# Feature Engineering

Processed weather observations are transformed into supervised time-series samples.

The model receives features available at or before observation time `t` and predicts temperature at `t + 1 hour`.

The feature engineering layer includes historical and temporal information such as:

```text
lag features
rolling features
temporal features
```

The resulting relationship is:

```text
historical observations
        │
        ▼
feature engineering
        │
        ▼
features at time t
        │
        ▼
RandomForestRegressor
        │
        ▼
temperature at t + 1 hour
```

The same feature-generation logic is reused during training and inference so that the serving feature schema matches the schema used to train the model.

---

# ML Dataset

Prepared datasets are versioned using a dataset ID, for example:

```text
weather-kazan-2000-2026-v1
```

Each dataset contains chronological:

```text
train
validation
test
```

splits.

The data is not randomly shuffled because this is a time-series forecasting problem.

Chronological splitting ensures that future observations do not leak into earlier training samples.

---

# Model Training

The forecasting model is a Scikit-learn:

```text
RandomForestRegressor
```

A Random Forest trains multiple decision trees on different samples and feature subsets and combines their predictions.

For regression, the final forecast is obtained by averaging the predictions produced by the individual trees.

The default project configuration is:

```text
n_estimators       = 64
max_depth          = 12
min_samples_split  = 2
min_samples_leaf   = 4
random_state       = 143
n_jobs             = 1
```

The training pipeline:

```text
load dataset
    │
    ▼
train / val / test
    │
    ├───────────────┐
    ▼               ▼
persistence      Random Forest
baseline          training
    │               │
    ▼               ▼
baseline val     model val
metrics          metrics
    │               │
    └───────┬───────┘
            │
            ▼
       test metrics
            │
            ▼
          MLflow
```

The training implementation is separated into:

```text
weather_forecasting/models/train.py
```

for model training and:

```text
weather_forecasting/pipelines/train_pipeline.py
```

for the complete training workflow.

---

# Baseline and Evaluation

The Random Forest is evaluated against a persistence baseline.

The baseline assumes:

```text
prediction(t + 1) = temperature(t)
```

In other words, the most recent observed temperature is used as the forecast for the next hour.

Three groups of metrics are recorded:

```text
baseline validation metrics
model validation metrics
model test metrics
```

MLflow stores them using:

```text
baseline_val_*
model_val_*
test_*
```

The validation set is used for candidate comparison and model promotion decisions.

The test set remains a separate final evaluation split.

---

# MLflow and Model Registry

MLflow tracks:

* training parameters;
* validation metrics;
* test metrics;
* model artifacts;
* model versions.

The trained model is not committed to Git.

Instead:

```text
training
    │
    ▼
MLflow run
    │
    ▼
model artifact
    │
    ▼
MinIO
```

The registered model is:

```text
weather-temperature-forecaster
```

The model selected for serving receives the alias:

```text
champion
```

FastAPI therefore loads:

```text
models:/weather-temperature-forecaster@champion
```

This separates model training from deployment and allows the serving model to be updated through the registry without committing generated model files to the repository.

---

# Automated Retraining

Automated retraining is orchestrated by:

```text
weather_retraining
```

defined in:

```text
services/airflow/dags/weather_retraining.py
```

The workflow performs:

```text
train candidate
      │
      ▼
evaluate validation metrics
      │
      ▼
compare with baseline
      │
      ├── candidate rejected
      │       │
      │       ▼
      │   keep current champion
      │
      └── candidate accepted
              │
              ▼
        register model
              │
              ▼
        assign champion
```

Airflow handles orchestration while the reusable data and ML implementation remains inside the `weather_forecasting` package.

---

# Prediction Service

FastAPI provides the model-serving layer.

Interactive API documentation:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Windows PowerShell:

```powershell
curl.exe http://localhost:8000/health
```

The prediction endpoint is:

```text
POST /predict
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2020-01-02T12:00:00Z"}'
```

For each request, the service:

```text
timestamp
    │
    ▼
load required weather history
    │
    ▼
construct model features
    │
    ▼
load champion from MLflow
    │
    ▼
predict temperature
```

The response contains the observation timestamp, forecast target timestamp and predicted temperature.

---

# Streamlit Application

The Streamlit application is available at:

```text
http://localhost:8501
```

Streamlit does not load the ML model directly.

Instead:

```text
Browser
    │
    ▼
Streamlit
    │
    │ POST /predict
    ▼
FastAPI
    │
    ├────────▶ MLflow Model Registry
    │
    └────────▶ MinIO
                   │
                   ▼
               prediction
```

This keeps the user interface separate from the model-serving layer.

---

# Project Lifecycle

The complete workflow is:

```text
Open-Meteo
    │
    ▼
Data Ingestion
    │
    ▼
Validation & Cleaning
    │
    ▼
MinIO
    │
    ▼
Feature Engineering
    │
    ▼
Chronological Train / Val / Test
    │
    ▼
Persistence Baseline
    │
    ▼
RandomForestRegressor
    │
    ▼
MLflow
    │
    ▼
Model Registry
    │
    ▼
champion
    │
    ▼
FastAPI
    │
    ▼
Streamlit
```

Airflow automates recurring ingestion and retraining, MLflow tracks experiments and manages model versions, MinIO stores datasets and model artifacts, FastAPI provides the prediction interface, and Streamlit provides the user-facing application.
