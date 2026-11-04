"""End-to-end smoke test: fit the pipeline and predict on new records."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor


def _make_rows(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    carriers = rng.choice(["AA", "DL", "WN"], size=n)
    origins = rng.choice(["JFK", "LAX", "ORD"], size=n)
    dests = rng.choice(["JFK", "LAX", "DEN"], size=n)
    return pd.DataFrame(
        {
            "YEAR": dates.year,
            "MONTH": dates.month,
            "DAY_OF_MONTH": dates.day,
            "DAY_OF_WEEK": dates.dayofweek + 1,
            "FL_DATE": dates.strftime("%Y-%m-%d"),
            "OP_UNIQUE_CARRIER": carriers,
            "ORIGIN": origins,
            "ORIGIN_STATE_NM": "State",
            "DEST": dests,
            "DEST_STATE_NM": "State",
            "CRS_DEP_TIME": (dates.hour * 100 + dates.minute).to_numpy(),
            "DEP_TIME_BLK": "1200-1259",
            "CRS_ARR_TIME": ((dates.hour + 2) % 24 * 100).to_numpy(),
            "CRS_ELAPSED_TIME": rng.uniform(60, 360, size=n),
            "DISTANCE": rng.uniform(100, 2500, size=n),
        }
    )


def test_pipeline_can_be_fitted_and_serves_unseen_values():
    frame = _make_rows(600)
    y = np.random.default_rng(1).choice(
        ["on_time", "moderate_delay", "severe_delay"], size=len(frame), p=[0.7, 0.2, 0.1]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=3,
                    n_estimators=10,
                    num_leaves=7,
                    min_child_samples=5,
                    verbose=-1,
                ),
            ),
        ]
    )
    pipeline.fit(frame, y)

    payload = _make_rows(5)
    payload["OP_UNIQUE_CARRIER"] = "NEW"
    payload["ORIGIN"] = "XYZ"
    payload["DEST"] = "QQQ"

    predictions = pipeline.predict(payload)
    probabilities = pipeline.predict_proba(payload)

    assert len(predictions) == 5
    assert set(predictions) <= {"on_time", "moderate_delay", "severe_delay"}
    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
