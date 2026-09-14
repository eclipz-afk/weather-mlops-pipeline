"""Historical weather backfill command.

Run from the repository root:
    python -m weather_mlops.datasets.backfill --start-date 2025-01-01 --end-date 2025-01-31
"""

import argparse
import logging
from datetime import date

from weather_mlops.datasets.cleaning import clean_weather
from weather_mlops.datasets.storage import write_partition
from weather_mlops.datasets.validation import validate_weather
from weather_mlops.datasets.weather_api import OpenMeteoClient, response_to_dataframe

logger = logging.getLogger("weather_mlops.backfill")

def _validate_date_range(start_date: str, end_date: str) -> None:
    """Raise if dates are malformed or start_date is after end_date.

    Dates must be ISO ``YYYY-MM-DD``. The range is inclusive on both
    sides, so a single-day backfill is valid.
    """
    try:
        start = date.fromisoformat(start_date)
    except ValueError as e:
        raise ValueError(
            f"start_date must be ISO YYYY-MM-DD; got {start_date!r}"
        ) from e
    try:
        end = date.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(
            f"end_date must be ISO YYYY-MM-DD; got {end_date!r}"
        ) from e

    if start > end:
        raise ValueError(
            f"start_date must be strictly before end_date; "
            f"got {start_date} and {end_date}"
        )


def backfill_weather(
    start_date: str,
    end_date: str,
) -> None:
    """Fetch, validate, clean and persist weather data for a date range.

    Parameters
    ----------
    start_date:
        ISO date ``YYYY-MM-DD`` (inclusive).
    end_date:
        ISO date ``YYYY-MM-DD`` (inclusive).
    Raises
    ------
    OpenMeteoError
        If the API request fails permanently.
    pandera.errors.PanderaError
        If validation fails.
    ValueError
        If cleaning detects a contract violation.
    botocore.exceptions.ClientError
        If the storage layer fails.
    """
    
    logger.info(f"backfill started: {start_date} .. {end_date}")
    _validate_date_range(start_date, end_date)
        
    client = OpenMeteoClient()
    payload = client.fetch_archive(start_date, end_date)
    raw_df = response_to_dataframe(payload)
    logger.info(f"fetched raw data: shape={raw_df.shape}")
    
    raw_keys = write_partition(raw_df, layer="raw")
    logger.info(f"wrote raw partitions: {len(raw_keys)} keys")
    
    validated_df = validate_weather(raw_df)
    logger.info(f"validated raw data: shape={validated_df.shape}")
    
    cleaned_df = clean_weather(validated_df)
    logger.info(f"cleaned data: shape={cleaned_df.shape}")
        
    processed_keys = write_partition(cleaned_df, layer="processed")
    logger.info(f"wrote processed partitions: {len(processed_keys)} keys")
    
    logger.info(
        "backfill completed: raw_partitions=%d processed_partitions=%d",
        len(raw_keys),
        len(processed_keys),
    )


def main(argv: list[str] | None = None) -> int:
    """Run a historical backfill from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch, validate, clean and store historical weather data."
    )
    parser.add_argument(
        "--start-date", required=True, help="Inclusive ISO date: YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date", required=True, help="Inclusive ISO date: YYYY-MM-DD"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    backfill_weather(
        args.start_date,
        args.end_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
