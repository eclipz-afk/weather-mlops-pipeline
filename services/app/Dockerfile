FROM python:3.14-slim

WORKDIR /app
COPY services/app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/app/app.py /app/app.py
COPY services/app/.streamlit/config.toml /app/.streamlit/config.toml

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
