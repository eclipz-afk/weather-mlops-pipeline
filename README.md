# Weather temperature forecasting MLOps pipeline

## Architecture

```text
Open-Meteo → MinIO (raw / processed / ML datasets) → training pipeline → MLflow
                                                                    ↓
Streamlit UI → FastAPI prediction service → MLflow Registry + MinIO history
```

The prediction service receives a historical UTC timestamp. It obtains the
current weather row and the preceding 24-hour temperature history from the
processed MinIO layer, then creates exactly the same features as the training
pipeline.

## Repository layout

```text
weather_forecasting/       reusable data, training and serving logic
services/
  airflow/                 scheduler image, dependencies and DAGs
  api/                     FastAPI application image
  app/                     Streamlit application image
  mlflow/                  MLflow tracking server image
docker-compose.yml         complete local platform
.env.example               configuration template
```

## Local run

1. Create the root `.env` from `deployment/.env.example` and choose a local
   MinIO password.
2. Start the platform from the repository root:

   ```powershell
   docker compose up -d --build
   ```

3. Open the services:

   - MLflow: `http://localhost:5000`
   - MinIO console: `http://localhost:9001`
   - Prediction API docs: `http://localhost:8000/docs`
   - Streamlit UI: `http://localhost:8501`

## Model promotion

After a successful training run, make it available to the API with:

```powershell
python -m weather_forecasting.models.registry --run-id <finished_mlflow_run_id>
```

The command creates a new version of `weather-temperature-forecaster` and
assigns it the `champion` alias. The API always loads that alias, so promoting a
better model does not require changing API code or rebuilding its image.

## Airflow orchestration

Airflow, its scheduler and PostgreSQL metadata database start with the rest of
the local stack. Copy `.env.example` to the root `.env`, replace the local
passwords and secret values, then start it from the repository root:

```powershell
docker compose up -d --build
```

Open `http://localhost:8080` and sign in with `AIRFLOW_ADMIN_USERNAME` and
`AIRFLOW_ADMIN_PASSWORD`. Both DAGs are paused initially to avoid accidental
network calls or expensive retraining:

- `weather_ingestion` obtains exactly the completed daily UTC data interval,
  validates it, and writes the raw and processed partitions to MinIO.
- `weather_retraining` trains a candidate from the configured dataset, rejects
  it if validation MAE is no better than the persistence baseline, and only
  then moves its MLflow Registry alias to `champion`.

The DAG files contain orchestration only; reusable ingestion, preparation,
training and registry logic stays in `weather_forecasting`.
