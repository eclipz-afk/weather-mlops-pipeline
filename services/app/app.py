from datetime import date, datetime, time, timezone
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Kazan temperature forecast", page_icon="🌡️")
st.title("Temperature prediction in Innopolis")
st.caption("The model predicts temperature in 1 hour based on previous observations")

left, right = st.columns(2)
with left:
    selected_date = st.date_input("Observation date (UTC)", value=date(2020, 1, 2))
with right:
    selected_time = st.time_input("Observation time (UTC)", value=time(12, 0))

if st.button("Get prediction", type="primary"):
    timestamp = datetime.combine(selected_date, selected_time, tzinfo=timezone.utc)
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"timestamp": timestamp.isoformat().replace("+00:00", "Z")},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"API is not reached: {detail}")
    else:
        st.metric(
            "Temperature in 1 hour",
            f"{result['temperature_2m_prediction']:.2f} °C",
        )
        st.caption(f"Prediction time: {result['target_timestamp']}")
