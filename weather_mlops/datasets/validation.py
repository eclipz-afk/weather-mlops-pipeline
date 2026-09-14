import logging
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

logger = logging.getLogger("weather_mlops.validation")

# Schema ---------------------------------------------------------

def _build_schema() -> pa.DataFrameSchema:
    """Build the Pandera schema for raw weather data
    
    Returns a fresh schema instance
    """
    
    return pa.DataFrameSchema(
        columns = {
            # Structural -------------------------------------------
            "timestamp": pa.Column(pd.DatetimeTZDtype(tz="UTC"), 
                nullable=False, 
                unique=True, 
                checks=pa.Check(lambda s: s.is_monotonic_increasing,
                error="timestamp must be sorted in ascending order"),
            ),
            "latitude": pa.Column(float, 
                nullable=False,
                checks=pa.Check.in_range(-90.0, 90.0),
            ),
            "longitude": pa.Column(
                float,
                nullable=False,
                checks=pa.Check.in_range(-180.0, 180.0),
            ),
            
            # temperature ----------------------------
            "temperature_2m": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(-90.0, 60.0),
            ),
            "dew_point_2m": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(-90.0, 60.0),
            ),
            "apparent_temperature": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(-100.0, 70.0),
            ),
            # humidity / cloud cover ------------------
            "relative_humidity_2m": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(0.0, 100.0),
            ),
            "cloud_cover": pa.Column(
                pd.Int32Dtype(),
                nullable=True,
                checks=pa.Check.in_range(0, 100),
            ),
            # percipitation ---------------------------
            "precipitation": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(0.0, 500.0),
            ),
            "rain": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(0.0, 500.0),
            ),
            "snowfall": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(0.0, 500.0),
            ),
            # pressure ---------------------------------
            "pressure_msl": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(800.0, 1100.0),
            ),
            "surface_pressure": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(500.0, 1100.0),
            ),
            # wind ---------------------------------
            "wind_speed_10m": pa.Column(
                float,
                nullable=True,
                checks=pa.Check.in_range(0.0, 120.0),
            ),
            "wind_direction_10m": pa.Column(
                pd.Int32Dtype(),
                nullable=True,
                checks=pa.Check.in_range(0, 360),
            ),
            # categorical ---------------------
            "weather_code": pa.Column(
                pd.Int32Dtype(),
                nullable=True,
                checks=pa.Check.in_range(0, 99),
            ),
        },
        strict = True,
        coerce = False,
    )

# Public API ----------------------------------------------------
def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a weather DataFrame against the project contract.

    Parameters
    ----------
    df:
        DataFrame produced by ``weather_api.response_to_dataframe``
        (or read back from raw Parquet).

    Returns
    -------
    pandas.DataFrame
        The same DataFrame, unchanged, if validation passes.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame violates the schema.
    """
    schema = _build_schema()
    validated = schema.validate(df, lazy=False)
    logger.info("weather DataFrame validated: shape=%s", validated.shape)
    return validated
    
