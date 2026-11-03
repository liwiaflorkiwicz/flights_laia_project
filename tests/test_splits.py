"""Tests for temporal and stratified split logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.prepare_splits import (
    parse_flight_dates,
    stratified_masks,
    temporal_masks,
    validate_masks,
)


def test_temporal_split_is_chronological_and_disjoint():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2023-09-30",
                "2023-10-01",
                "2023-11-30",
                "2023-12-01",
                "2024-01-15",
            ]
        )
    )
    masks = temporal_masks(dates, train_end="2023-09-30", val_end="2023-11-30")
    validate_masks(masks)

    assert masks["train"].tolist() == [True, False, False, False, False]
    assert masks["val"].tolist() == [False, True, True, False, False]
    assert masks["test"].tolist() == [False, False, False, True, True]


def test_temporal_split_has_no_overlap():
    dates = pd.Series(pd.date_range("2022-01-01", "2023-12-31", periods=500))
    masks = temporal_masks(dates, "2023-09-30", "2023-11-30")
    validate_masks(masks)
    total = sum(mask.sum() for mask in masks.values())
    assert total == len(dates)


def test_stratified_split_preserves_class_balance():
    target = pd.Series(["on_time"] * 800 + ["moderate_delay"] * 150 + ["severe_delay"] * 50)
    masks = stratified_masks(target, val_size=0.15, test_size=0.15, random_state=42)
    validate_masks(masks)

    for mask in masks.values():
        shares = target[mask].value_counts(normalize=True)
        assert abs(shares["on_time"] - 0.8) < 0.02
        assert abs(shares["severe_delay"] - 0.05) < 0.02


def test_date_parser_handles_bts_and_iso_formats():
    values = pd.Series(["1/1/2022 12:00:00 AM", "2025-05-15"])
    parsed = parse_flight_dates(values)
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2022-01-01")
    assert parsed.iloc[1] == pd.Timestamp("2025-05-15")
