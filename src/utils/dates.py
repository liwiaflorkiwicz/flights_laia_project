"""Date parsing shared by the data and feature pipelines."""

from __future__ import annotations

import pandas as pd


def parse_flight_dates(values) -> pd.Series:
    """Parse BTS ("1/1/2022 12:00:00 AM") and ISO ("2025-05-15") dates.

    The BTS format is tried first because it covers the historical files;
    any remaining values (e.g. ISO dates from the inference API) are parsed
    individually. Unparseable values become NaT instead of raising.
    """
    series = pd.Series(values, copy=False).astype("string")
    parsed = pd.to_datetime(
        series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    missing = parsed.isna() & series.notna()
    if bool(missing.any()):
        parsed = parsed.copy()
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing], errors="coerce", format="mixed"
        )
    return parsed
