import logging
import argparse
import json
from dataclasses import asdict, dataclass

from weather_forecasting.datasets.storage import read_ml_manifest, read_ml_split
from weather_forecasting.models.baseline import persistence_baseline
from weather_forecasting.models.evaluate import evaluate_model
from weather_forecasting.models.metrics import calculate_metrics
from weather_forecasting.models.mlflow_utils import log_to_mlflow
from weather_forecasting.models.train import train_model

logger = logging.getLogger("weather_forecasting.train_pipeline")

@dataclass(frozen=True)
class TrainingResult:
    dataset_id: str
    model_name: str
    baseline_val_metrics: dict[str, float]
    model_val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    mlflow_run_id: str
    
def _prefix_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in metrics.items()}

def train_and_log(
    *,
    dataset_id: str,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 143,
    n_jobs: int = 1,
    run_name: str | None = None,
) -> TrainingResult:
    manifest = read_ml_manifest(dataset_id)
    
    X_train, y_train = read_ml_split(dataset_id, "train")
    X_val, y_val = read_ml_split(dataset_id, "val")
    X_test, y_test = read_ml_split(dataset_id, "test")
    
    baseline_prediction = persistence_baseline(X_val)
    baseline_val_metrics = calculate_metrics(y_val, baseline_prediction)
    
    model = train_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    model_val_metrics = evaluate_model(model, X_val, y_val)
    
    test_metrics = evaluate_model(model, X_test, y_test)
    
    mlflow_metrics = {
        **_prefix_metrics(baseline_val_metrics, prefix="baseline_val"),
        **_prefix_metrics(model_val_metrics, prefix="model_val"),
        **_prefix_metrics(test_metrics, prefix="test"),
    }
    
    mlflow_run_id = log_to_mlflow(
        model,
        metrics=mlflow_metrics,
        run_name=run_name or f"random-forest-{dataset_id}",
        extra_params={
            "dataset_id": dataset_id,
            "target_column": manifest["target_column"],
            "feature_count": len(X_train.columns),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        },
    )
    
    result = TrainingResult(
        dataset_id=dataset_id,
        model_name="RandomForestRegressor",
        baseline_val_metrics=baseline_val_metrics,
        model_val_metrics=model_val_metrics,
        test_metrics=test_metrics,
        mlflow_run_id=mlflow_run_id,
    )
    
    logger.info(
        "training completed: dataset_id=%s baseline_val_mae=%.4f "
        "model_val_mae=%.4f test_mae=%.4f run_id=%s",
        dataset_id,
        baseline_val_metrics["mae"],
        model_val_metrics["mae"],
        test_metrics["mae"],
        mlflow_run_id,
    )
    
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and log the experiment to MLflow."
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Published ML dataset version in MinIO.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=143)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable MLflow run name.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = train_and_log(
        dataset_id=args.dataset_id,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        run_name=args.run_name,
    )

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
