"""Class-weight helpers shared by training and experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def inverse_frequency_weights(
    y: Sequence[str], labels: Sequence[str], power: float
) -> dict[str, float] | None:
    """Inverse-frequency weights raised to ``power``.

    ``power=0`` means no weighting and ``power=1`` reproduces sklearn's
    ``class_weight="balanced"``. Intermediate values are a useful compromise
    for imbalanced data: full balancing improves minority-class recall but can
    make the model over-predict delays, hurting on-time recall and calibration.
    Weights are rescaled so their prevalence-weighted mean is 1.
    """
    if power is None or power <= 0:
        return None

    counts = pd.Series(y).value_counts().reindex(labels, fill_value=0)
    total = int(counts.sum())
    weights = {
        label: (total / (len(labels) * counts[label])) ** power
        if counts[label] > 0
        else 0.0
        for label in labels
    }
    prevalence = {label: counts[label] / total for label in labels}
    mean_weight = sum(prevalence[label] * weights[label] for label in labels)
    return {label: float(value / mean_weight) for label, value in weights.items()}
