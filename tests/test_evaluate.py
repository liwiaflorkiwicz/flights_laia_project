"""Tests for metric computation and subgroup reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.evaluate import classification_metrics, subgroup_metrics


def test_classification_metrics_aligns_probability_columns():
    labels = ["on_time", "moderate_delay", "severe_delay"]
    # sklearn's classes_ order is alphabetical.
    classes = ["moderate_delay", "on_time", "severe_delay"]
    y_true = ["on_time", "moderate_delay", "severe_delay", "on_time"]
    y_pred = ["on_time", "moderate_delay", "severe_delay", "moderate_delay"]
    probabilities = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.1, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.5, 0.3],
        ]
    )

    metrics = classification_metrics(
        y_true, y_pred, probabilities, labels=labels, classes=classes
    )

    assert metrics["n_samples"] == 4
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["log_loss"] > 0
    assert set(metrics["per_class"]) == set(labels)


def test_subgroup_metrics_flags_small_groups():
    labels = ["on_time", "moderate_delay", "severe_delay"]
    y_true = ["on_time", "on_time", "moderate_delay", "severe_delay", "on_time"]
    y_pred = ["on_time", "moderate_delay", "moderate_delay", "on_time", "on_time"]
    frame = pd.DataFrame({"OP_UNIQUE_CARRIER": ["AA", "AA", "AA", "XX", "XX"]})

    results = subgroup_metrics(
        frame, y_true, y_pred, ["OP_UNIQUE_CARRIER"], labels, min_size=3
    )
    groups = results["OP_UNIQUE_CARRIER"]

    assert groups["AA"]["n_samples"] == 3
    assert groups["AA"]["reliable"] is True
    assert groups["XX"]["n_samples"] == 2
    assert groups["XX"]["reliable"] is False
