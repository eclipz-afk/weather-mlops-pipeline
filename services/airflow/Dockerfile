FROM apache/airflow:2.10.5-python3.12

# Airflow orchestrates the project but does not own its business logic.
# Install the runtime dependencies required by weather_forecasting tasks.
COPY services/airflow/requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt

COPY --chown=airflow:root weather_forecasting /opt/airflow/weather_forecasting
