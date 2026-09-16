FROM ghcr.io/mlflow/mlflow:v3.16.0

WORKDIR /app

# Keep the serving environment intentionally small and deterministic.
RUN pip install --no-cache-dir boto3==1.43.93 scikit-learn==1.9.1

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

COPY weather_forecasting /app/weather_forecasting
COPY services/api/main.py /app/main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
