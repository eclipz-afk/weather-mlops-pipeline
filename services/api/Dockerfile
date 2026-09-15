FROM ghcr.io/mlflow/mlflow:v3.16.0

WORKDIR /app
# The MLflow base already contains FastAPI, pandas, pyarrow, skops and
# uvicorn. Pin the learner runtime to the version that created the artifact.
RUN pip install --no-cache-dir boto3==1.43.93 scikit-learn==1.9.1

COPY weather_forecasting /app/weather_forecasting
COPY services/api/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
