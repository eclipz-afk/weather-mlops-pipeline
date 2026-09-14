import json
import logging
import random
import time
from typing import Any

import requests


# Project configuration -----------------------------------
from weather_mlops.config import HOURLY_VARIABLES, LOCATION

logger = logging.getLogger("weather_mlops.weather_api")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

BASE_PARAMS: dict[str, Any] = {
    "wind_speed_unit": "ms",
    "timezone": "UTC",
}

# Exceptions ---------------------------------------------------
class OpenMeteoError(RuntimeError):
    """General API/client failure.

    Raised for: HTTP 4xx (other than retried 429), invalid JSON,
    unexpected response structure, and requests that kept failing
    after all retries.
    """

class OpenMeteoQuotaError(OpenMeteoError):
    """The API request quota has been exhausted."""

# Client ------------------------------------------------------------
class OpenMeteoClient:
    """Small retry-capable HTTP client for the Open-Meteo archive API.

    Retry policy:
    * transient failures (network errors, HTTP 429/500/502/503/504) are
      retried with exponential backoff + jitter; for HTTP 429 the
      ``Retry-After`` header is respected when present;
    * a confirmed quota exhaustion raises :class:`OpenMeteoQuotaError`
      immediately — no rapid retries;
    * permanent API errors raise :class:`OpenMeteoError` immediately.
    """

    TRANSIENT_STATUS = {429, 500, 502, 503, 504}
    MAX_RETRY_AFTER = 60.0

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # -- public API ---------------------------------------------------------
    def fetch_archive(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Fetch hourly historical weather for LOCATION between two dates.

        Args:
            start_date: ISO date ``YYYY-MM-DD`` (inclusive).
            end_date: ISO date ``YYYY-MM-DD`` (inclusive).

        Returns:
            Parsed JSON response as a dict.

        Raises:
            OpenMeteoQuotaError: daily quota exhausted.
            OpenMeteoError: permanent API error.
        """
        params: dict[str, Any] = {
            **BASE_PARAMS,
            "latitude": LOCATION["latitude"],
            "longitude": LOCATION["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(ARCHIVE_URL, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt, f"network error: {exc}")
                continue

            if response.status_code in self.TRANSIENT_STATUS:
                last_exc = OpenMeteoError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                if response.status_code == 429:
                    quota = self._quota_reason(response)
                    if quota is not None:
                        raise OpenMeteoQuotaError(quota)
                    self._sleep_backoff(
                        attempt,
                        "HTTP 429 rate-limited",
                        extra_wait=self._retry_after_seconds(response),
                    )
                else:
                    self._sleep_backoff(attempt, f"HTTP {response.status_code}")
                continue

            if response.status_code >= 400:
                raise OpenMeteoError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )

            try:
                payload = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise OpenMeteoError(
                    f"invalid JSON in API response (HTTP "
                    f"{response.status_code}): {exc}"
                ) from exc
            if isinstance(payload, dict) and payload.get("error"):
                reason = str(payload.get("reason", "unknown"))
                if "limit" in reason.lower():
                    raise OpenMeteoQuotaError(reason)
                raise OpenMeteoError(reason)
            return payload

        raise OpenMeteoError(
            f"archive request failed after {self.max_retries} attempts: {last_exc}"
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _quota_reason(response: requests.Response) -> str | None:
        """Return the quota reason if the response body reports"""
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(body, dict) and body.get("error"):
            reason = str(body.get("reason", ""))
            if "limit" in reason.lower():
                return reason
        return None
    
    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        """Parse the Retry-After header in seconds"""
        raw = response.headers.get("Retry-After")
        if raw is None:
            return 0.0
        try:
            return min(float(raw), OpenMeteoClient.MAX_RETRY_AFTER)
        except ValueError:
            return 0.0
    
    def _sleep_backoff(self, attempt: int, reason: str, extra_wait: float = 0.0) -> None:
        wait = min(self.backoff_base ** attempt, 60.0) + random.uniform(0, 1)
        wait = max(wait, extra_wait)
        logger.warning(
            "attempt %d/%d failed (%s); retrying in %.1fs",
            attempt, self.max_retries, reason, wait
        )
        time.sleep(wait)
        
INT_COLUMNS: set[str] = {"weather_code", "cloud_cover", "wind_direction_10m"}

def _validate_archive_payload(payload: Any) -> dict[str, Any]:
    """Validate the expected archive response structure before indexing.

    Structural (schema-level) checks only — NOT semantic data-quality
    checks, which belong to validation.py (Pandera) in the next phase.
    """
    if not isinstance(payload, dict):
        raise OpenMeteoError(
            f"unexpected response structure: expected dict, got {type(payload).__name__}"
        )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
         raise OpenMeteoError("unexpected response structure: missing 'hourly' object")

    time_arr = hourly.get("time")
    if not isinstance(time_arr, list) or not time_arr:
        raise OpenMeteoError("unexpected response structure: 'hourly.time' is missing or empty")

    n = len(time_arr)
    missing = [var for var in HOURLY_VARIABLES if var not in hourly]
    if missing:
        raise OpenMeteoError(f"response is missing requested hourly variables: {missing}")
    mismatched = [var for var in HOURLY_VARIABLES if len(hourly[var]) != n]
    if mismatched:
        raise OpenMeteoError(
            f"hourly arrays are not aligned with 'time' for: {mismatched}"
        )
    return payload

def response_to_dataframe(payload: dict[str, Any]) -> "pd.DataFrame":
    """Convert an Open-Meteo archive JSON payload to a tidy DataFrame.

    Performs: JSON -> DataFrame; ``time`` -> ``timestamp`` (tz-aware UTC);
    location metadata columns; nullable integer conversion; sorting.

    Does NOT perform: missing-value imputation, outlier removal, semantic
    data-quality checks (Pandera), feature engineering.

    Columns: ``timestamp`` + HOURLY_VARIABLES + ``latitude`` / ``longitude``.
    """
    import pandas as pd
    
    _validate_archive_payload(payload)
    hourly = payload["hourly"]
    
    df = pd.DataFrame(hourly)
    
    df["timestamp"] = pd.to_datetime(df.pop("time"), utc=True).dt.as_unit("ns")
    df["relative_humidity_2m"] = df["relative_humidity_2m"].astype("float64")
    df["latitude"] = float(payload["latitude"])
    df["longitude"] = float(payload["longitude"])
    
    for col in INT_COLUMNS & set(df.columns):
        df[col] = df[col].astype("Int32")
    
    ordered = ["timestamp", *HOURLY_VARIABLES, "latitude", "longitude"]
    df = df[ordered].sort_values("timestamp").reset_index(drop=True)
    
    return df
