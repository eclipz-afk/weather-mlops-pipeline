FROM ghcr.io/mlflow/mlflow:v3.16.0

# MLflow needs boto3 in the server container to proxy model artifacts to MinIO.
RUN pip install --no-cache-dir boto3==1.43.93
