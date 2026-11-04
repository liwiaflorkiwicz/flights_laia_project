"""Diagnostics for a trained model: 3-class metrics and delay detection.

Besides the usual macro-F1, this reports the binary "delayed vs not" view
(precision/recall/F1), the delay-vs-not ROC AUC and the best achievable
delay F1 over all thresholds.  Those are the numbers that show whether a
feature set actually improves delay identification, as opposed to just
moving the operating point.

Usage:
    python -m src.modeling.diagnose --config configs/train_weather.yaml
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.modeling.evaluate import classification_metrics
from src.utils.config import load_config, resolve_path


def _binary_counts(truth: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "predicted_delayed": int(prediction.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _binary_delay_metrics(
    truth: np.ndarray, score: np.ndarray, argmax_delay: np.ndarray
) -> dict:
    pr_precision, pr_recall, thresholds = precision_recall_curve(truth, score)
    pr_f1 = 2 * pr_precision * pr_recall / np.clip(
        pr_precision + pr_recall, 1e-12, None
    )
    best = int(np.nanargmax(pr_f1))
    best_threshold = float(thresholds[min(best, len(thresholds) - 1)])

    return {
        "base_rate": float(truth.mean()),
        "auroc": float(roc_auc_score(truth, score)),
        "at_argmax": _binary_counts(truth, argmax_delay),
        "at_0_5": _binary_counts(truth, score >= 0.5),
        "best_f1": {
            "threshold": best_threshold,
            "precision": float(pr_precision[best]),
            "recall": float(pr_recall[best]),
            "f1": float(pr_f1[best]),
        },
    }


def diagnose(model, frame: pd.DataFrame, target: str, labels: list[str]) -> dict:
    y = frame[target].astype(str).to_numpy()
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    X = preprocessor.transform(frame.drop(columns=[target]))
    proba = classifier.predict_proba(X)
    classes = list(classifier.classes_)
    order = [classes.index(label) for label in labels]
    proba = proba[:, order]
    prediction = np.asarray(labels)[proba.argmax(axis=1)]

    metrics = classification_metrics(
        y,
        prediction,
        proba,
        labels=labels,
        # ``proba`` is already aligned to ``labels``; passing classifier.classes_
        # here would reorder it a second time (the production log-loss bug).
        classes=labels,
    )
    delay_score = proba[:, labels.index("moderate_delay")] + proba[
        :, labels.index("severe_delay")
    ]
    argmax_delay = proba.argmax(axis=1) != labels.index("on_time")
    metrics["delay_detection"] = _binary_delay_metrics(
        y != "on_time", delay_score, argmax_delay
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target = cfg["data"].get("target", "DELAY_SEVERITY")
    labels = list(cfg["evaluation"]["labels"])
    split_dir = resolve_path(cfg["data"]["split_dir"])
    model_path = resolve_path(args.model or cfg["artifacts"]["model_path"])
    model = joblib.load(model_path)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    payload = {}
    for split in splits:
        frame = pd.read_parquet(split_dir / f"{split}.parquet")
        payload[split] = diagnose(model, frame, target, labels)
        metrics = payload[split]
        delay = metrics["delay_detection"]
        print(
            f"{split:4s} macro-F1={metrics['macro_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"delay AUC={delay['auroc']:.4f} "
            f"argmax P/R/F1={delay['at_argmax']['precision']:.4f}/"
            f"{delay['at_argmax']['recall']:.4f}/{delay['at_argmax']['f1']:.4f} "
            f"best-F1={delay['best_f1']['f1']:.4f} "
            f"(thr={delay['best_f1']['threshold']:.3f})"
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
