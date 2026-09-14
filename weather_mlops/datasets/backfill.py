"""Historical weather backfill command.

Run from the repository root:
    python -m weather_mlops.datasets.backfill --start-date 2025-01-01 --end-date 2025-01-31
"""

import argparse
import logging
from datetime import date, timedelta

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


def _year_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into calendar-year API requests."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year, 12, 31), end)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def backfill_weather(
    start_date: str,
    end_date: str,
) -> None:
    """Fetch, validate, clean and persist weather data for a date range.

    The range is split into calendar-year API chunks. The storage layer then
    writes one raw and one processed Parquet object per calendar day, with
    hourly observations retained within each file.

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
        
    chunks = _year_chunks(start_date, end_date)
    client = OpenMeteoClient()
    raw_partitions = 0
    processed_partitions = 0

    for chunk_number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        logger.info(
            "processing chunk %d/%d: %s .. %s",
            chunk_number,
            len(chunks),
            chunk_start,
            chunk_end,
        )
        payload = client.fetch_archive(chunk_start, chunk_end)
        raw_df = response_to_dataframe(payload)
        logger.info("fetched raw data: shape=%s", raw_df.shape)

        raw_keys = write_partition(raw_df, layer="raw")
        raw_partitions += len(raw_keys)
        logger.info("wrote raw partitions: %d keys", len(raw_keys))

        validated_df = validate_weather(raw_df)
        cleaned_df = clean_weather(validated_df)
        processed_keys = write_partition(cleaned_df, layer="processed")
        processed_partitions += len(processed_keys)
        logger.info("wrote processed partitions: %d keys", len(processed_keys))

    logger.info(
        "backfill completed: raw_partitions=%d processed_partitions=%d",
        raw_partitions,
        processed_partitions,
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
