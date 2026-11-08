"""Train and evaluate the flight delay severity classifier.

The script:
1. loads the temporal train/val/test splits;
2. fits the leakage-free feature pipeline on the training split only;
3. trains LightGBM with early stopping on the validation split;
4. reports overall, per-class and subgroup metrics on validation and test;
5. saves a single reusable joblib pipeline plus metrics/figures;
6. optionally logs everything to MLflow (params, metrics, artifacts, model).

Usage:
    python -m src.modeling.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor
from src.modeling.class_weights import inverse_frequency_weights
from src.modeling.evaluate import (
    classification_metrics,
    report_text,
    save_confusion_matrix,
    subgroup_metrics,
)
from src.utils.config import REPO_ROOT, load_config, resolve_path


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
            )
            .strip()
        )
    except Exception:
        return None


def read_split(split_dir: Path, name: str, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = split_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.data.prepare_splits --config configs/train.yaml"
        )
    frame = pd.read_parquet(path)
    if target not in frame.columns:
        raise KeyError(f"Target {target!r} missing from {path}")
    y = frame[target].astype(str).to_numpy()
    return frame, y


def maybe_subsample(
    frame: pd.DataFrame, y: np.ndarray, max_rows: int | None, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stratified subsample for quick local smoke tests."""
    if max_rows is None or len(frame) <= max_rows:
        return frame, y
    _, sample_idx = train_test_split(
        np.arange(len(frame)),
        test_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    print(f"Subsampling training split from {len(frame):,} to {len(sample_idx):,} rows")
    return frame.iloc[sample_idx].reset_index(drop=True), y[sample_idx]


def transform_in_chunks(
    frame: pd.DataFrame, preprocessor, chunk_size: int
) -> np.ndarray:
    """Transform a DataFrame in row chunks to limit peak memory."""
    matrices = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        matrices.append(
            np.asarray(preprocessor.transform(chunk), dtype=np.float32)
        )
    return np.vstack(matrices)


def fit_preprocessor_on_sample(
    path: Path,
    preprocessor,
    sample_rows: int,
    batch_size: int,
    random_state: int,
):
    """Fit encoders/frequencies on a random sample instead of all 11.5M rows."""
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    fraction = min(1.0, sample_rows / total_rows)
    rng = np.random.default_rng(random_state)

    samples = []
    collected = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        if fraction < 1.0:
            mask = rng.random(len(frame)) < fraction
            frame = frame.loc[mask]
        samples.append(frame)
        collected += len(frame)
        if collected >= sample_rows:
            break

    sample = pd.concat(samples, ignore_index=True)
    print(f"Fitting feature pipeline on {len(sample):,} sampled rows ...")
    preprocessor.fit(sample)
    del samples, sample
    gc.collect()
    return preprocessor


def build_training_matrix_streaming(
    path: Path,
    target: str,
    preprocessor,
    n_features: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the training matrix batch by batch, avoiding a full DataFrame copy."""
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    matrix = np.empty((total_rows, n_features), dtype=np.float32)
    targets = np.empty(total_rows, dtype=object)

    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        chunk = np.asarray(preprocessor.transform(frame), dtype=np.float32)
        matrix[offset : offset + len(frame)] = chunk
        targets[offset : offset + len(frame)] = frame[target].astype(str).to_numpy()
        offset += len(frame)
        if offset % (batch_size * 10) < batch_size:
            print(f"  transformed {offset:,}/{total_rows:,} training rows")

    if offset != total_rows:
        raise RuntimeError(f"Transformed {offset:,} rows, expected {total_rows:,}")
    return matrix, targets


def aligned_probabilities(clf: LGBMClassifier, X: np.ndarray, labels: list[str]) -> np.ndarray:
    probabilities = clf.predict_proba(X)
    class_to_index = {label: i for i, label in enumerate(clf.classes_)}
    return probabilities[:, [class_to_index[label] for label in labels]]


def macro_f1_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    """LightGBM custom metric so early stopping targets macro-F1 directly."""
    probabilities = y_pred.reshape(len(y_true), -1, order="F")
    predictions = probabilities.argmax(axis=1)
    score = f1_score(
        y_true,
        predictions,
        average="macro",
        labels=np.arange(probabilities.shape[1]),
        zero_division=0,
    )
    return "macro_f1", float(score), True


def flatten_params(prefix: str, values: dict) -> dict:
    flat = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(flatten_params(f"{prefix}{key}_", value))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def log_to_mlflow(
    cfg: dict,
    pipeline: Pipeline,
    metrics: dict,
    metadata: dict,
    artifacts: list[Path],
    input_example: pd.DataFrame,
    run_name: str | None,
    register: bool,
) -> None:
    import mlflow
    import os
    import mlflow.sklearn

    #mlflow_cfg = cfg["mlflow"]
    #mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_tracking_uri(os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "http://10.17.0.190:5050")
    ))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "flight-delay-severity"))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_params("model_", cfg["model"]["params"]))
        mlflow.log_params(
            {
                "split_strategy": metrics["split_strategy"],
                "train_rows": metadata["splits"]["train"]["rows"],
                "val_rows": metadata["splits"]["val"]["rows"],
                "test_rows": metadata["splits"]["test"]["rows"],
                "best_iteration": metadata["best_iteration"],
                "labels": ",".join(metadata["labels"]),
            }
        )

        for split in ("val", "test"):
            split_metrics = metrics[split]
            mlflow.log_metrics(
                {
                    f"{split}_{name}": float(split_metrics[name])
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "quadratic_weighted_kappa",
                        "log_loss",
                    )
                    if name in split_metrics
                }
            )
            for label, values in split_metrics["per_class"].items():
                mlflow.log_metrics(
                    {
                        f"{split}_precision_{label}": values["precision"],
                        f"{split}_recall_{label}": values["recall"],
                        f"{split}_f1_{label}": values["f1"],
                    }
                )
            for column, groups in metrics.get("subgroups", {}).get(split, {}).items():
                reliable = [g["macro_f1"] for g in groups.values() if g["reliable"]]
                if reliable:
                    safe_column = column.lower()
                    mlflow.log_metrics(
                        {
                            f"{split}_subgroup_{safe_column}_min_f1": float(np.min(reliable)),
                            f"{split}_subgroup_{safe_column}_median_f1": float(np.median(reliable)),
                        }
                    )

        for path in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path))

        mlflow.set_tags(
            {
                "git_commit": metadata.get("git_commit") or "unknown",
                "python_version": platform.python_version(),
                "lightgbm_version": lgb.__version__,
                "split_strategy": metrics["split_strategy"],
            }
        )

        registered_name = None
        if register:
            registered_name = cfg["mlflow"].get("registered_model_name")
        try:
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=input_example,
                registered_model_name=registered_name,
            )
        except Exception as exc:  # registry/file-store limitations, etc.
            print(f"WARNING: could not log model to MLflow ({exc}). Artifacts are still saved locally.")
            mlflow.log_dict(metadata, "model_metadata_fallback.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mlflow.enabled from the config.",
    )
    parser.add_argument(
        "--register",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Register the model in the MLflow Model Registry.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional stratified training subsample for quick smoke runs.",
    )
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="Skip training and only log the existing artifacts to MLflow.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    target = data_cfg.get("target", "DELAY_SEVERITY")
    labels = list(cfg["evaluation"]["labels"])
    split_dir = resolve_path(data_cfg["split_dir"])
    random_state = int(data_cfg["split"].get("random_state", 42))

    mlflow_enabled = cfg.get("mlflow", {}).get("enabled", False)
    if args.mlflow is not None:
        mlflow_enabled = args.mlflow
    register = cfg.get("mlflow", {}).get("register_model", False)
    if args.register is not None:
        register = args.register

    if args.log_only:
        model_path = resolve_path(cfg["artifacts"]["model_path"])
        metrics_path = resolve_path(cfg["evaluation"]["metrics_path"])
        metadata_path = resolve_path(cfg["artifacts"]["metadata_path"])
        figures_dir = resolve_path(cfg["evaluation"]["figures_dir"])
        missing = [
            path
            for path in (model_path, metrics_path, metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Cannot log-only, missing artifacts: {missing}")

        pipeline = joblib.load(model_path)
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        val_frame, _ = read_split(split_dir, "val", target)

        artifacts = [metrics_path, metadata_path]
        report_path = resolve_path(cfg["evaluation"]["report_path"])
        importance_path = REPO_ROOT / "reports" / "feature_importance.csv"
        split_summary_path = split_dir / "split_summary.json"
        artifacts.extend(
            [
                path
                for path in (
                    report_path,
                    importance_path,
                    split_summary_path,
                    figures_dir / "confusion_matrix_val.png",
                    figures_dir / "confusion_matrix_test.png",
                )
                if path.exists()
            ]
        )
        log_to_mlflow(
            cfg,
            pipeline,
            metrics_payload,
            metadata,
            artifacts,
            input_example=val_frame.head(3).drop(columns=[target], errors="ignore"),
            run_name=args.run_name,
            register=register,
        )
        print("MLflow log-only run complete")
        return

    print("Loading validation/test splits ...")
    val_frame, y_val = read_split(split_dir, "val", target)
    test_frame, y_test = read_split(split_dir, "test", target)

    subgroup_columns = list(cfg["evaluation"].get("subgroup_columns", []))
    val_groups = val_frame[
        [column for column in subgroup_columns if column in val_frame.columns]
    ].copy()
    test_groups = test_frame[
        [column for column in subgroup_columns if column in test_frame.columns]
    ].copy()

    chunk_size = int(data_cfg.get("chunk_size", 200_000))
    preprocessor = build_preprocessor(
        categorical=cfg["features"].get("categorical"),
        frequency=cfg["features"].get("frequency"),
        numeric=cfg["features"].get("numeric"),
    )
    started = time.perf_counter()

    if args.max_train_rows is not None:
        train_frame, y_train = read_split(split_dir, "train", target)
        train_frame, y_train = maybe_subsample(
            train_frame, y_train, args.max_train_rows, random_state
        )
        print("Fitting feature pipeline ...")
        preprocessor.fit(train_frame, y_train)
        X_train = transform_in_chunks(train_frame, preprocessor, chunk_size)
        del train_frame
    else:
        sample_rows = int(data_cfg.get("fit_sample_rows", 1_000_000))
        fit_preprocessor_on_sample(
            split_dir / "train.parquet",
            preprocessor,
            sample_rows=sample_rows,
            batch_size=chunk_size,
            random_state=random_state,
        )
        n_features = len(preprocessor.get_feature_names_out())
        print("Building training matrix in batches ...")
        X_train, y_train = build_training_matrix_streaming(
            split_dir / "train.parquet",
            target,
            preprocessor,
            n_features=n_features,
            batch_size=chunk_size,
        )

    X_val = transform_in_chunks(val_frame, preprocessor, chunk_size)

    print(f"Training matrix: {X_train.shape[0]:,} x {X_train.shape[1]}")

    unknown = set(np.unique(y_train)) - set(labels)
    if unknown:
        raise ValueError(f"Unexpected labels in training split: {sorted(unknown)}")

    model_params = dict(cfg["model"]["params"])
    model_params.pop("class_weight", None)
    weight_power = cfg["model"].get("class_weight_power")
    class_weights = inverse_frequency_weights(y_train, labels, float(weight_power or 0.0))
    print(f"Class weights (power={weight_power}): {class_weights}")
    early_stopping_rounds = int(cfg["model"].get("early_stopping_rounds", 100))
    eval_metric = cfg["model"].get("eval_metric", "macro_f1")
    metric_argument = macro_f1_eval if eval_metric == "macro_f1" else eval_metric

    classifier = LGBMClassifier(**model_params, class_weight=class_weights)
    print(f"Training LightGBM (early stopping on validation {eval_metric}) ...")
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=metric_argument,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    training_seconds = time.perf_counter() - started
    best_iteration = int(getattr(classifier, "best_iteration_", classifier.n_estimators) or 0)
    print(f"Best iteration: {best_iteration} (training took {training_seconds / 60:.1f} min)")

    pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("classifier", classifier)]
    )

    # Keep validation/test untouched until this point: test is only scored once.
    del X_train
    X_test = np.asarray(preprocessor.transform(test_frame), dtype=np.float32)

    pred_val = classifier.predict(X_val)
    proba_val = aligned_probabilities(classifier, X_val, labels)
    pred_test = classifier.predict(X_test)
    proba_test = aligned_probabilities(classifier, X_test, labels)

    val_metrics = classification_metrics(
        y_val, pred_val, proba_val, labels=labels, classes=classifier.classes_
    )
    test_metrics = classification_metrics(
        y_test, pred_test, proba_test, labels=labels, classes=classifier.classes_
    )

    min_subgroup_size = int(cfg["evaluation"].get("min_subgroup_size", 200))
    val_subgroups = subgroup_metrics(
        val_groups, y_val, pred_val, subgroup_columns, labels, min_subgroup_size
    )
    test_subgroups = subgroup_metrics(
        test_groups, y_test, pred_test, subgroup_columns, labels, min_subgroup_size
    )

    print(
        f"Validation macro-F1={val_metrics['macro_f1']:.4f}  "
        f"balanced-acc={val_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Test       macro-F1={test_metrics['macro_f1']:.4f}  "
        f"balanced-acc={test_metrics['balanced_accuracy']:.4f}"
    )

    # ------------------------------------------------------------------ artifacts
    model_path = resolve_path(cfg["artifacts"]["model_path"])
    metadata_path = resolve_path(cfg["artifacts"]["metadata_path"])
    metrics_path = resolve_path(cfg["evaluation"]["metrics_path"])
    report_path = resolve_path(cfg["evaluation"]["report_path"])
    figures_dir = resolve_path(cfg["evaluation"]["figures_dir"])

    for path in (model_path, metadata_path, metrics_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_summary_path = split_dir / "split_summary.json"
    split_summary = (
        json.loads(split_summary_path.read_text(encoding="utf-8"))
        if split_summary_path.exists()
        else {}
    )

    metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_strategy": data_cfg["split"].get("strategy", "temporal"),
        "labels": labels,
        "best_iteration": best_iteration,
        "training_seconds": round(training_seconds, 2),
        "val": val_metrics,
        "test": test_metrics,
        "subgroups": {"val": val_subgroups, "test": test_subgroups},
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    report = (
        "VALIDATION (Oct-Nov 2023)\n"
        + "=" * 60
        + "\n"
        + report_text(y_val, pred_val, labels)
        + "\n\nTEST (Dec 2023)\n"
        + "=" * 60
        + "\n"
        + report_text(y_test, pred_test, labels)
    )
    report_path.write_text(report, encoding="utf-8")

    save_confusion_matrix(
        y_val,
        pred_val,
        figures_dir / "confusion_matrix_val.png",
        "Validation confusion matrix (row-normalised)",
        labels,
    )
    save_confusion_matrix(
        y_test,
        pred_test,
        figures_dir / "confusion_matrix_test.png",
        "Test confusion matrix (row-normalised)",
        labels,
    )

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.booster_.feature_importance(importance_type="gain")
    importance_path = REPO_ROOT / "reports" / "feature_importance.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_names, "gain": importance}).sort_values(
        "gain", ascending=False
    ).to_csv(importance_path, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "lightgbm_version": lgb.__version__,
        "target": target,
        "labels": labels,
        "classes": [str(c) for c in classifier.classes_],
        "n_features": int(X_val.shape[1]),
        "feature_names": [str(name) for name in feature_names],
        "best_iteration": best_iteration,
        "training_seconds": round(training_seconds, 2),
        "splits": {
            "train": {"rows": len(y_train)},
            "val": {"rows": len(y_val)},
            "test": {"rows": len(y_test)},
            "summary": split_summary,
        },
        "metrics": {"val": val_metrics, "test": test_metrics},
        "config": cfg,
    }
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report to {report_path}")

    if mlflow_enabled:
        print("Logging run to MLflow ...")
        artifacts = [
            metrics_path,
            report_path,
            metadata_path,
            importance_path,
            figures_dir / "confusion_matrix_val.png",
            figures_dir / "confusion_matrix_test.png",
        ]
        if split_summary_path.exists():
            artifacts.append(split_summary_path)
        try:
            log_to_mlflow(
                cfg,
                pipeline,
                metrics_payload,
                metadata,
                artifacts,
                input_example=val_frame.head(3)[
                    [c for c in val_frame.columns if c != target]
                ],
                run_name=args.run_name,
                register=register,
            )
            print("MLflow run complete")
        except Exception as exc:  # tracking server unavailable, file store, ...
            print(
                f"WARNING: MLflow logging failed ({exc}). "
                "Training artifacts were saved locally."
            )


if __name__ == "__main__":
    main()
