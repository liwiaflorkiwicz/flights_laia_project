"""Ablation: do year-specific calendar features cause the December shift?

We compare the full numeric feature set against variants that drop long-term
seasonality features. All variants use the same mild class weighting and are
selected on validation; the test month is only reported.

Usage:
    python -m src.modeling.experiment_features --max-train-rows 2000000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.features.build_features import NUMERIC_FEATURES, build_preprocessor
from src.modeling.class_weights import inverse_frequency_weights
from src.modeling.evaluate import classification_metrics
from src.modeling.train import macro_f1_eval
from src.utils.config import load_config, resolve_path

LABELS = ["on_time", "moderate_delay", "severe_delay"]

DROP_MONTH = ["month", "month_sin", "month_cos"]
DROP_LONG_TERM = [
    "month",
    "day_of_year",
    "week_of_year",
    "quarter",
    "month_sin",
    "month_cos",
]
DROP_ALL_CALENDAR = DROP_LONG_TERM + ["day_of_month"]

VARIANTS = {
    "all": NUMERIC_FEATURES,
    "no_month": [f for f in NUMERIC_FEATURES if f not in DROP_MONTH],
    "no_long_term_seasonality": [
        f for f in NUMERIC_FEATURES if f not in DROP_LONG_TERM
    ],
    "operational_only": [
        f for f in NUMERIC_FEATURES if f not in DROP_ALL_CALENDAR
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-train-rows", type=int, default=2_000_000)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--out", default="reports/feature_ablation.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target = cfg["data"].get("target", "DELAY_SEVERITY")
    split_dir = resolve_path(cfg["data"]["split_dir"])

    print("Loading splits ...")
    train_frame = pd.read_parquet(split_dir / "train.parquet")
    val_frame = pd.read_parquet(split_dir / "val.parquet")
    test_frame = pd.read_parquet(split_dir / "test.parquet")
    y_train = train_frame[target].astype(str).to_numpy()
    y_val = val_frame[target].astype(str).to_numpy()
    y_test = test_frame[target].astype(str).to_numpy()

    if args.max_train_rows and len(train_frame) > args.max_train_rows:
        _, idx = train_test_split(
            np.arange(len(train_frame)),
            test_size=args.max_train_rows,
            stratify=y_train,
            random_state=42,
        )
        idx = np.sort(idx)
        train_frame = train_frame.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]
        print(f"Using a stratified {len(train_frame):,}-row training sample")

    base_params = dict(cfg["model"]["params"])
    base_params.pop("class_weight", None)
    base_params["n_estimators"] = args.n_estimators
    weights = inverse_frequency_weights(
        y_train, LABELS, float(cfg["model"].get("class_weight_power", 0.75))
    )

    results = []
    for name, numeric_features in VARIANTS.items():
        print(f"\nTraining variant '{name}' ({len(numeric_features)} numeric features) ...")
        preprocessor = build_preprocessor(
            categorical=cfg["features"].get("categorical"),
            frequency=cfg["features"].get("frequency"),
            numeric=numeric_features,
        )
        X_train = np.asarray(preprocessor.fit_transform(train_frame), dtype=np.float32)
        X_val = np.asarray(preprocessor.transform(val_frame), dtype=np.float32)
        X_test = np.asarray(preprocessor.transform(test_frame), dtype=np.float32)

        classifier = LGBMClassifier(**base_params, class_weight=weights)
        classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=macro_f1_eval,
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        classes = list(classifier.classes_)
        order = [classes.index(label) for label in LABELS]

        row = {"variant": name, "n_numeric_features": len(numeric_features)}
        for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            probabilities = classifier.predict_proba(X)[:, order]
            predictions = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
            metrics = classification_metrics(y, predictions)
            row[split] = {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "on_time_recall": metrics["per_class"]["on_time"]["recall"],
                "on_time_precision": metrics["per_class"]["on_time"]["precision"],
                "moderate_f1": metrics["per_class"]["moderate_delay"]["f1"],
                "severe_f1": metrics["per_class"]["severe_delay"]["f1"],
                "predicted_on_time_share": float(np.mean(predictions == "on_time")),
                "actual_on_time_share": float(np.mean(y == "on_time")),
            }
        results.append(row)
        print(
            f"  val F1={row['val']['macro_f1']:.4f} "
            f"(on-time rec {row['val']['on_time_recall']:.3f}) | "
            f"test F1={row['test']['macro_f1']:.4f} "
            f"(on-time rec {row['test']['on_time_recall']:.3f})"
        )

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "n_num": row["n_numeric_features"],
                "val_f1": row["val"]["macro_f1"],
                "test_f1": row["test"]["macro_f1"],
                "val_on_time_recall": row["val"]["on_time_recall"],
                "test_on_time_recall": row["test"]["on_time_recall"],
                "val_pred_on_time": row["val"]["predicted_on_time_share"],
                "test_pred_on_time": row["test"]["predicted_on_time_share"],
            }
            for row in results
        ]
    )
    print("\n" + table.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
