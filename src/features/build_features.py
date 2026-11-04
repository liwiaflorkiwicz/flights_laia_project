"""Feature engineering for pre-departure flight information.

Only columns that are known *before* the scheduled departure are used.
Post-departure fields (DepDelay, TaxiOut, AirTime, delay causes, ...) are
never read, which prevents target leakage.

The transformers are scikit-learn compatible so the exact same object can
be used at training time and at inference time, and saved as one artifact.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils.dates import parse_flight_dates

# Numeric features produced by :class:`FlightFeatureEngineer`.
NUMERIC_FEATURES: list[str] = [
    "month",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "dep_hour",
    "dep_minute",
    "dep_minutes",
    "dep_time_sin",
    "dep_time_cos",
    "arr_hour",
    "arr_minute",
    "arr_minutes",
    "arr_time_sin",
    "arr_time_cos",
    "is_red_eye",
    "is_overnight",
    "dep_period",
    "crs_elapsed_time",
    "distance",
    "distance_per_minute",
    "implied_speed_mph",
    "schedule_buffer_min",
    "same_state",
]

DEFAULT_CATEGORICAL: list[str] = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "ORIGIN_STATE_NM",
    "DEST_STATE_NM",
    "DEP_TIME_BLK",
    "ROUTE",
]

DEFAULT_FREQUENCY: list[str] = [
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "ROUTE",
]

MISSING_CATEGORY = "__MISSING__"


def _series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    """Return a column, creating a default one when the input lacks it.

    The inference API may receive slightly different payloads across
    evaluation rounds, so transformers must never assume a column exists.
    """
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series(frame, column), errors="coerce")


def _parse_dates(frame: pd.DataFrame) -> pd.Series:
    """Parse BTS-style and ISO-8601 dates, falling back gracefully."""
    return parse_flight_dates(_series(frame, "FL_DATE"))


def _build_calendar(frame: pd.DataFrame) -> pd.Series:
    """Build a date series from YEAR/MONTH/DAY_OF_MONTH when available."""
    parsed = _parse_dates(frame)
    if {"YEAR", "MONTH", "DAY_OF_MONTH"}.issubset(frame.columns):
        constructed = pd.to_datetime(
            {
                "year": _numeric(frame, "YEAR"),
                "month": _numeric(frame, "MONTH"),
                "day": _numeric(frame, "DAY_OF_MONTH"),
            },
            errors="coerce",
        )
        # Prefer the explicit calendar columns; use FL_DATE only to fill gaps.
        return constructed.fillna(parsed)
    return parsed


def _hhmm_to_minutes(values: pd.Series) -> pd.Series:
    """Convert BTS hhmm integers (e.g. 1730) to minutes since midnight."""
    values = pd.to_numeric(values, errors="coerce")
    values = values.where((values >= 0) & (values <= 2400))
    hours = (values // 100).clip(upper=24)
    minutes = values % 100
    total = hours * 60 + minutes
    return total.where(total <= 24 * 60)


def _cyclic(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    radians = 2 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn raw BTS records into leakage-free numeric/categorical features.

    Parameters
    ----------
    categorical:
        Raw categorical columns kept for the encoder (plus the derived
        ``ROUTE`` column).
    frequency:
        Categorical columns for which training-set frequency encodings
        are produced.  Frequencies make unseen categories safe (value 0)
        and give the model a useful popularity signal.
    """

    def __init__(
        self,
        categorical: Sequence[str] | None = None,
        frequency: Sequence[str] | None = None,
        numeric: Sequence[str] | None = None,
    ) -> None:
        self.categorical = categorical
        self.frequency = frequency
        self.numeric = numeric

    def _categorical_columns(self) -> list[str]:
        if self.categorical is None:
            return list(DEFAULT_CATEGORICAL)
        return list(self.categorical)

    def _frequency_columns(self) -> list[str]:
        if self.frequency is None:
            return list(DEFAULT_FREQUENCY)
        return list(self.frequency)

    def _numeric_columns(self) -> list[str]:
        if self.numeric is None:
            return list(NUMERIC_FEATURES)
        return list(self.numeric)

    def fit(self, X: pd.DataFrame, y=None) -> "FlightFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FlightFeatureEngineer expects a pandas DataFrame")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FlightFeatureEngineer expects a pandas DataFrame")

        n = len(X)
        index = X.index
        calendar = _build_calendar(X)

        month = _numeric(X, "MONTH").fillna(calendar.dt.month)
        day_of_month = _numeric(X, "DAY_OF_MONTH").fillna(calendar.dt.day)
        day_of_week = _numeric(X, "DAY_OF_WEEK").fillna(calendar.dt.dayofweek + 1)

        out: dict[str, pd.Series | np.ndarray] = {}
        out["month"] = month
        out["day_of_month"] = day_of_month
        out["day_of_week"] = day_of_week
        out["day_of_year"] = calendar.dt.dayofyear
        out["week_of_year"] = calendar.dt.isocalendar().week.astype("float64")
        out["quarter"] = calendar.dt.quarter
        out["is_weekend"] = day_of_week.isin([6, 7]).astype("int8")

        month_values = month.to_numpy(dtype="float64")
        dow_values = day_of_week.to_numpy(dtype="float64")
        out["month_sin"], out["month_cos"] = _cyclic(month_values, 12.0)
        out["dow_sin"], out["dow_cos"] = _cyclic(dow_values, 7.0)

        dep_minutes = _hhmm_to_minutes(_series(X, "CRS_DEP_TIME"))
        arr_minutes = _hhmm_to_minutes(_series(X, "CRS_ARR_TIME"))
        dep_minutes = dep_minutes.fillna(calendar.dt.hour * 60 + calendar.dt.minute)

        dep_tod = dep_minutes % (24 * 60)
        arr_tod = arr_minutes % (24 * 60)
        out["dep_hour"] = dep_tod // 60
        out["dep_minute"] = dep_tod % 60
        out["dep_minutes"] = dep_tod
        out["dep_time_sin"], out["dep_time_cos"] = _cyclic(
            dep_tod.to_numpy(dtype="float64"), 24 * 60
        )
        out["arr_hour"] = arr_tod // 60
        out["arr_minute"] = arr_tod % 60
        out["arr_minutes"] = arr_tod
        out["arr_time_sin"], out["arr_time_cos"] = _cyclic(
            arr_tod.to_numpy(dtype="float64"), 24 * 60
        )
        out["is_red_eye"] = ((dep_tod < 6 * 60) | (dep_tod >= 22 * 60)).astype("int8")
        out["is_overnight"] = ((arr_minutes < dep_minutes) & arr_minutes.notna()).astype("int8")
        out["dep_period"] = pd.cut(
            dep_tod,
            bins=[-1, 5 * 60 + 59, 11 * 60 + 59, 17 * 60 + 59, 24 * 60],
            labels=[0, 1, 2, 3],
        ).astype("float64")

        elapsed = _numeric(X, "CRS_ELAPSED_TIME")
        distance = _numeric(X, "DISTANCE")
        elapsed_safe = elapsed.where(elapsed > 0)
        out["crs_elapsed_time"] = elapsed
        out["distance"] = distance
        out["distance_per_minute"] = distance / elapsed_safe
        out["implied_speed_mph"] = distance / (elapsed_safe / 60.0)
        out["schedule_buffer_min"] = elapsed - (distance / 500.0) * 60.0
        origin_state = _series(X, "ORIGIN_STATE_NM", MISSING_CATEGORY)
        dest_state = _series(X, "DEST_STATE_NM", MISSING_CATEGORY)
        out["same_state"] = (
            (origin_state.astype("string") == dest_state.astype("string"))
            & origin_state.notna()
            & dest_state.notna()
        ).astype("int8")

        # Categorical features are kept as strings; missing values receive
        # an explicit category so that the encoder never sees NaN.
        for column in self._categorical_columns():
            if column == "ROUTE":
                origin = _series(X, "ORIGIN", MISSING_CATEGORY).astype("string")
                dest = _series(X, "DEST", MISSING_CATEGORY).astype("string")
                values = origin.str.upper() + "_" + dest.str.upper()
            else:
                values = _series(X, column, MISSING_CATEGORY).astype("string")
            out[column] = values.fillna(MISSING_CATEGORY).astype(object)

        # Configured numeric columns that are not derived above (for example
        # weather features joined onto the splits) are passed through as-is.
        # Missing columns become NaN, so the same pipeline accepts payloads
        # with or without those optional features.
        for column in self._numeric_columns():
            if column not in out:
                out[column] = _numeric(X, column)

        transformed = pd.DataFrame(out, index=index)
        # Keep memory predictable on the 13M-row dataset and support feature
        # subsets for ablation studies.
        selected = [
            column
            for column in self._numeric_columns() + self._categorical_columns()
            if column in transformed.columns
        ]
        transformed = transformed[selected]
        for column in self._numeric_columns():
            if column in transformed.columns:
                transformed[column] = pd.to_numeric(
                    transformed[column], errors="coerce"
                ).astype("float32")
        return transformed


    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(
            self._numeric_columns() + self._categorical_columns(), dtype=object
        )


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace categories with their normalised training frequency.

    Unseen categories map to 0.0, which makes the transformer robust to
    airlines/airports that only appear in future evaluation rounds.
    """

    def __init__(self, columns: Iterable[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        self.columns_ = list(self.columns)
        self.maps_: dict[str, dict] = {}
        for column in self.columns_:
            values = X[column].astype("object").where(X[column].notna(), MISSING_CATEGORY)
            self.maps_[column] = values.value_counts(normalize=True, dropna=False).to_dict()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.columns_:
            if column in X.columns:
                values = X[column].astype("object").where(X[column].notna(), MISSING_CATEGORY)
            else:
                values = pd.Series(MISSING_CATEGORY, index=X.index, dtype="object")
            columns.append(
                values.map(self.maps_.get(column, {})).fillna(0.0).to_numpy(dtype="float32")
            )
        if not columns:
            return np.empty((len(X), 0), dtype="float32")
        return np.column_stack(columns).astype("float32", copy=False)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        columns = getattr(self, "columns_", list(self.columns))
        return np.asarray([f"freq_{column}" for column in columns], dtype=object)


def build_preprocessor(
    categorical: Sequence[str] | None = None,
    frequency: Sequence[str] | None = None,
    numeric: Sequence[str] | None = None,
) -> Pipeline:
    """Build the feature-engineering + encoding pipeline.

    The returned object accepts raw BTS records (a DataFrame) and emits a
    numeric matrix ready for LightGBM.
    """
    categorical = list(categorical) if categorical is not None else DEFAULT_CATEGORICAL
    frequency = list(frequency) if frequency is not None else DEFAULT_FREQUENCY
    numeric = list(numeric) if numeric is not None else NUMERIC_FEATURES

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            ("categorical", encoder, categorical),
            ("frequency", FrequencyEncoder(frequency), frequency),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="default")

    return Pipeline(
        steps=[
            (
                "features",
                FlightFeatureEngineer(
                    categorical=categorical, frequency=frequency, numeric=numeric
                ),
            ),
            ("preprocess", transformer),
        ]
    )
