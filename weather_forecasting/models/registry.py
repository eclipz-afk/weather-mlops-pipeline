"""Promotion of a validated MLflow run to the serving alias."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from weather_forecasting.config import (
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

MODEL_ARTIFACT_PATH = "model"


@dataclass(frozen=True)
class PromotionResult:
    """Immutable description of the promoted registry version."""

    model_name: str
    version: str
    alias: str
    run_id: str


def promote_run(
    run_id: str,
    *,
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    alias: str = MLFLOW_MODEL_ALIAS,
) -> PromotionResult:
    """Register the run's model and make it the serving ``alias``."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not alias.strip():
        raise ValueError("alias must not be empty")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"only FINISHED runs can be promoted: {run_id}")

    version = mlflow.register_model(
        f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}",
        model_name,
    )
    client.set_registered_model_alias(model_name, alias, version.version)
    return PromotionResult(
        model_name=model_name,
        version=version.version,
        alias=alias,
        run_id=run_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a finished MLflow run as the serving model."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=MLFLOW_REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default=MLFLOW_MODEL_ALIAS)
    args = parser.parse_args(argv)

    result = promote_run(
        args.run_id,
        model_name=args.model_name,
        alias=args.alias,
    )
    print(
        f"promoted run={result.run_id} model={result.model_name} "
        f"version={result.version} alias={result.alias}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
