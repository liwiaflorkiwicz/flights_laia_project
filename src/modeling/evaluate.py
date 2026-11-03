"""Evaluation utilities: overall metrics, per-class metrics and subgroups."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

DEFAULT_LABELS = ["on_time", "moderate_delay", "severe_delay"]


def _proba_in_label_order(
    probabilities: np.ndarray, classes: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    classes = list(classes)
    order = [classes.index(label) for label in labels]
    return probabilities[:, order]


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    classes: Sequence[str] | None = None,
) -> dict:
    """Compute the metrics used for model selection and reporting."""
    labels = list(labels or DEFAULT_LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    metrics = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        ),
        "per_class": {
            str(label): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    if y_proba is not None and classes is not None:
        # sklearn's log_loss requires the probability columns in sorted
        # label order; our configured order (on_time first) differs.
        sorted_labels = sorted(labels)
        ordered = _proba_in_label_order(np.asarray(y_proba), classes, sorted_labels)
        metrics["log_loss"] = float(log_loss(y_true, ordered, labels=sorted_labels))

    return metrics


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    columns: Iterable[str],
    labels: Sequence[str] | None = None,
    min_size: int = 200,
) -> dict:
    """Macro-F1/accuracy per airline, origin airport, ... .

    Small groups are reported with ``"reliable": false`` instead of being
    silently dropped, because monitoring must still surface them.
    """
    labels = list(labels or DEFAULT_LABELS)
    evaluation = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)})
    results: dict = {}

    for column in columns:
        if column not in frame.columns:
            continue
        evaluation["group"] = frame[column].astype("object").fillna("__MISSING__").to_numpy()
        group_results = {}
        for group, subset in evaluation.groupby("group", sort=False):
            size = int(len(subset))
            group_results[str(group)] = {
                "n_samples": size,
                "accuracy": float(accuracy_score(subset["y_true"], subset["y_pred"])),
                "macro_f1": float(
                    f1_score(
                        subset["y_true"],
                        subset["y_pred"],
                        average="macro",
                        labels=labels,
                        zero_division=0,
                    )
                ),
                "reliable": bool(size >= min_size),
            }
        results[column] = group_results

    return results


def report_text(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> str:
    labels = list(labels or DEFAULT_LABELS)
    return classification_report(
        y_true, y_pred, labels=labels, target_names=labels, zero_division=0, digits=4
    )


def save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    path,
    title: str,
    labels: Sequence[str] | None = None,
) -> None:
    labels = list(labels or DEFAULT_LABELS)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(
                col,
                row,
                f"{normalised[row, col]:.2f}\n({matrix[row, col]:,})",
                ha="center",
                va="center",
                color="white" if normalised[row, col] > 0.55 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
