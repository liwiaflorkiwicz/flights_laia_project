"""Experiment: class weighting vs. decision-bias correction.

``class_weight='balanced'`` compensates for the 78/14/7 class imbalance but
also pushes the model to predict delays too aggressively. This script
compares several weighting strengths and, for each model, tunes a simple
post-hoc log-probability bias on the validation set (never on test).

The bias is equivalent to changing class priors at decision time:
    prediction = argmax(log p(y|x) + b_y)
Only K-1 biases are identifiable, so the ``on_time`` bias is fixed at 0.

Usage:
    python -m src.modeling.experiment_class_weights --max-train-rows 2000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.features.build_features import build_preprocessor
from src.modeling.evaluate import classification_metrics
from src.utils.config import load_config, resolve_path

LABELS = ["on_time", "moderate_delay", "severe_delay"]


def class_weights(y: np.ndarray, power: float) -> dict[str, float] | None:
    """Inverse-frequency weights raised to ``power``.

    power=0 -> no weighting, power=1 -> sklearn's "balanced".
    """
    if power <= 0:
        return None
    counts = pd.Series(y).value_counts()
    total = len(y)
    weights = {label: (total / (len(LABELS) * counts[label])) ** power for label in LABELS}
    # Keep the effective scale close across powers (weighted mean = 1).
    prevalence = {label: counts[label] / total for label in LABELS}
    mean_weight = sum(prevalence[label] * weights[label] for label in LABELS)
    return {label: value / mean_weight for label, value in weights.items()}


def macro_f1_from_codes(y_codes: np.ndarray, pred_codes: np.ndarray, n_classes: int = 3) -> float:
    """Fast macro-F1 from integer-coded labels (no sklearn overhead)."""
    matrix = np.bincount(
        y_codes * n_classes + pred_codes, minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(f1.mean())


def tune_bias(
    probabilities: np.ndarray,
    y_codes: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float, float]:
    """Grid-search moderate/severe log-probability biases on validation."""
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    best_score = -1.0
    best = (0.0, 0.0, 0.0)
    for moderate_bias in grid:
        for severe_bias in grid:
            bias = np.array([0.0, moderate_bias, severe_bias])
            predictions = np.argmax(log_probabilities + bias, axis=1)
            score = macro_f1_from_codes(y_codes, predictions)
            if score > best_score:
                best_score = score
                best = (0.0, float(moderate_bias), float(severe_bias))
    return best


def apply_bias(probabilities: np.ndarray, bias: tuple[float, float, float]) -> np.ndarray:
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    return np.asarray(LABELS)[np.argmax(log_probabilities + np.asarray(bias), axis=1)]


def evaluate(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bias: tuple[float, float, float] | None = None,
):
    if bias is None:
        predictions = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    else:
        predictions = apply_bias(probabilities, bias)
    metrics = classification_metrics(y_true, predictions)
    predicted_on_time = float(np.mean(predictions == "on_time"))
    actual_on_time = float(np.mean(np.asarray(y_true) == "on_time"))
    return metrics, predicted_on_time, actual_on_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-train-rows", type=int, default=2_000_000)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--grid-points", type=int, default=13)
    parser.add_argument(
        "--out", default="reports/class_weight_experiments.json"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    target = data_cfg.get("target", "DELAY_SEVERITY")
    split_dir = resolve_path(data_cfg["split_dir"])
    labels = list(cfg["evaluation"]["labels"])

    print("Loading splits ...")
    train_frame = pd.read_parquet(split_dir / "train.parquet")
    val_frame = pd.read_parquet(split_dir / "val.parquet")
    test_frame = pd.read_parquet(split_dir / "test.parquet")

    y_train = train_frame[target].astype(str).to_numpy()
    y_val = val_frame[target].astype(str).to_numpy()
    y_test = test_frame[target].astype(str).to_numpy()
    label_to_code = {label: index for index, label in enumerate(LABELS)}
    y_val_codes = np.array([label_to_code[label] for label in y_val], dtype=np.int8)
    y_test_codes = np.array([label_to_code[label] for label in y_test], dtype=np.int8)

    if args.max_train_rows and len(train_frame) > args.max_train_rows:
        _, idx = train_test_split(
            np.arange(len(train_frame)),
            test_size=args.max_train_rows,
            stratify=y_train,
            random_state=42,
        )
        train_frame = train_frame.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]
        print(f"Using a stratified {len(train_frame):,}-row training sample")

    print("Building feature matrices ...")
    preprocessor = build_preprocessor(
        categorical=cfg["features"].get("categorical"),
        frequency=cfg["features"].get("frequency"),
        numeric=cfg["features"].get("numeric"),
    )
    X_train = np.asarray(preprocessor.fit_transform(train_frame), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(val_frame), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_frame), dtype=np.float32)
    del train_frame, val_frame, test_frame

    base_params = dict(cfg["model"]["params"])
    base_params.pop("class_weight", None)
    base_params["n_estimators"] = args.n_estimators
    grid = np.linspace(-3.0, 3.0, args.grid_points)

    results = []
    for power in (0.0, 0.35, 0.55, 0.75, 1.0):
        weights = class_weights(y_train, power)
        print(f"\nTraining power={power} weights={weights} ...")
        classifier = LGBMClassifier(**base_params, class_weight=weights)
        classifier.fit(X_train, y_train)

        classes = list(classifier.classes_)
        order = [classes.index(label) for label in labels]
        proba_val = classifier.predict_proba(X_val)[:, order]
        proba_test = classifier.predict_proba(X_test)[:, order]

        raw_val, raw_val_share, actual_val_share = evaluate(proba_val, y_val)
        raw_test, raw_test_share, actual_test_share = evaluate(proba_test, y_test)

        bias = tune_bias(proba_val, y_val_codes, grid)
        tuned_val, tuned_val_share, _ = evaluate(proba_val, y_val, bias)
        tuned_test, tuned_test_share, _ = evaluate(proba_test, y_test, bias)

        row = {
            "power": power,
            "weights": weights,
            "bias": bias,
            "val": {
                "raw_macro_f1": raw_val["macro_f1"],
                "tuned_macro_f1": tuned_val["macro_f1"],
                "raw_on_time_recall": raw_val["per_class"]["on_time"]["recall"],
                "tuned_on_time_recall": tuned_val["per_class"]["on_time"]["recall"],
                "actual_on_time_share": actual_val_share,
                "raw_predicted_on_time_share": raw_val_share,
                "tuned_predicted_on_time_share": tuned_val_share,
                "accuracy": tuned_val["accuracy"],
            },
            "test": {
                "raw_macro_f1": raw_test["macro_f1"],
                "tuned_macro_f1": tuned_test["macro_f1"],
                "raw_on_time_recall": raw_test["per_class"]["on_time"]["recall"],
                "tuned_on_time_recall": tuned_test["per_class"]["on_time"]["recall"],
                "actual_on_time_share": actual_test_share,
                "raw_predicted_on_time_share": raw_test_share,
                "tuned_predicted_on_time_share": tuned_test_share,
                "accuracy": tuned_test["accuracy"],
            },
        }
        results.append(row)
        print(
            f"  val raw/tuned macro-F1 = {raw_val['macro_f1']:.4f}/{tuned_val['macro_f1']:.4f} | "
            f"test raw/tuned = {raw_test['macro_f1']:.4f}/{tuned_test['macro_f1']:.4f}"
        )
        print(
            f"  on_time recall val raw/tuned = "
            f"{raw_val['per_class']['on_time']['recall']:.3f}/"
            f"{tuned_val['per_class']['on_time']['recall']:.3f} | "
            f"test raw/tuned = {raw_test['per_class']['on_time']['recall']:.3f}/"
            f"{tuned_test['per_class']['on_time']['recall']:.3f}"
        )

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    table = pd.DataFrame(
        [
            {
                "power": row["power"],
                "val_raw_f1": row["val"]["raw_macro_f1"],
                "val_tuned_f1": row["val"]["tuned_macro_f1"],
                "test_raw_f1": row["test"]["raw_macro_f1"],
                "test_tuned_f1": row["test"]["tuned_macro_f1"],
                "val_on_time_recall": row["val"]["tuned_on_time_recall"],
                "test_on_time_recall": row["test"]["tuned_on_time_recall"],
                "val_pred_on_time": row["val"]["tuned_predicted_on_time_share"],
                "test_pred_on_time": row["test"]["tuned_predicted_on_time_share"],
                "bias_moderate": row["bias"][1],
                "bias_severe": row["bias"][2],
            }
            for row in results
        ]
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
