"""Tests for leakage-free feature engineering and unseen-category handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FlightFeatureEngineer,
    build_preprocessor,
)


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "YEAR": [2023, 2023, 2023, 2024],
            "MONTH": [10, 10, 11, 1],
            "DAY_OF_MONTH": [1, 2, 3, 4],
            "DAY_OF_WEEK": [7, 1, 2, 4],
            "FL_DATE": [
                "10/1/2023 12:00:00 AM",
                "10/2/2023 12:00:00 AM",
                "11/3/2023 12:00:00 AM",
                "2024-01-04",
            ],
            "OP_UNIQUE_CARRIER": ["AA", "DL", "WN", "AA"],
            "ORIGIN": ["JFK", "LAX", "ORD", "JFK"],
            "ORIGIN_STATE_NM": ["New York", "California", "Illinois", "New York"],
            "DEST": ["LAX", "JFK", "DEN", "LAX"],
            "DEST_STATE_NM": ["California", "New York", "Colorado", "California"],
            "CRS_DEP_TIME": [1730, 600, 2359, 1200],
            "DEP_TIME_BLK": ["1700-1759", "0600-0659", "2300-2359", "1200-1259"],
            "CRS_ARR_TIME": [2030, 930, 120, 1530],
            "CRS_ELAPSED_TIME": [360.0, 330.0, 200.0, 360.0],
            "DISTANCE": [2475.0, 2475.0, 1000.0, 2475.0],
            # Leakage columns: must never influence the features.
            "DEP_DELAY": [15.0, 0.0, 120.0, 5.0],
            "TAXI_OUT": [20.0, 12.0, 40.0, 10.0],
        }
    )


def test_engineer_outputs_expected_features(raw_frame):
    engineer = FlightFeatureEngineer()
    features = engineer.fit_transform(raw_frame)

    assert {"month", "dep_minutes", "distance", "ROUTE"} <= set(features.columns)
    assert "DEP_DELAY" not in features.columns
    assert "TAXI_OUT" not in features.columns
    assert features["ROUTE"].tolist() == ["JFK_LAX", "LAX_JFK", "ORD_DEN", "JFK_LAX"]


def test_preprocessor_handles_unseen_categories(raw_frame):
    preprocessor = build_preprocessor()
    preprocessor.fit(raw_frame, ["on_time"] * len(raw_frame))

    unseen = raw_frame.copy()
    unseen["OP_UNIQUE_CARRIER"] = "XX"
    unseen["ORIGIN"] = "ZZZ"
    unseen["DEST"] = "YYY"

    matrix = preprocessor.transform(unseen)
    assert matrix.shape[0] == len(unseen)
    assert np.isfinite(matrix).all()


def test_preprocessor_accepts_minimal_payload():
    """The API may receive only FL_DATE for a record; it must not crash."""
    preprocessor = build_preprocessor()
    example = pd.DataFrame(
        {
            "FL_DATE": ["2025-05-15"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "ORIGIN": ["JFK"],
            "DEST": ["LAX"],
            "CRS_DEP_TIME": [1730],
        }
    )
    matrix = preprocessor.fit_transform(example, ["on_time"])
    assert matrix.shape[0] == 1


def test_malformed_date_does_not_crash():
    preprocessor = build_preprocessor()
    example = pd.DataFrame({"FL_DATE": ["not-a-date"], "ORIGIN": ["JFK"]})
    matrix = preprocessor.fit_transform(example, ["on_time"])
    assert matrix.shape[0] == 1
