import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# 1. Chronological Split Function (Prevents Temporal Data Leakage)
def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2023-09-30",
    val_end: str = "2023-11-15",
):
    """
    Splits the 2022-2023 dataset chronologically by FL_DATE:
    - Train: 2022-01-01 to 2023-09-30 (21 months)
    - Validation: 2023-10-01 to 2023-11-15 (1.5 months)
    - Test: 2023-11-16 to 2023-12-31 (1.5 months)
    """
    dates = pd.to_datetime(df["FL_DATE"])
    train_mask = dates <= pd.to_datetime(train_end)
    val_mask = (dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end))
    test_mask = dates > pd.to_datetime(val_end)

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


# 2. Custom Feature Engineering Transformer
class FlightFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts calendar features from FL_DATE, converts HHMM times into
    minutes since midnight + cyclical sin/cos features, and calculates
    scheduled flight speed.
    """

    def fit(self, X, y=None):
        # Stateless transformer: nothing to learn during fit
        return self

    def transform(self, X):
        df = X.copy()

        # A. Extract calendar attributes from FL_DATE (needed when API receives only FL_DATE)
        if "FL_DATE" in df.columns:
            dt = pd.to_datetime(df["FL_DATE"], errors="coerce")
            df["MONTH"] = dt.dt.month.fillna(0).astype(int)
            df["DAY_OF_MONTH"] = dt.dt.day.fillna(0).astype(int)
            # Pandas dayofweek is 0=Mon..6=Sun; BTS standard is 1=Mon..7=Sun
            df["DAY_OF_WEEK"] = (dt.dt.dayofweek + 1).fillna(0).astype(int)

        # B. Convert HHMM integer times (e.g. 1430 -> 14*60 + 30 = 870 minutes)
        for col in ["CRS_DEP_TIME", "CRS_ARR_TIME"]:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").fillna(0)
                hours = (val // 100).clip(0, 24)
                mins = (val % 100).clip(0, 59)
                total_mins = hours * 60 + mins
                df[f"{col}_MINS"] = total_mins

                # Cyclical encoding so 23:55 and 00:05 are recognized as close in time
                df[f"{col}_SIN"] = np.sin(2 * np.pi * total_mins / 1440.0)
                df[f"{col}_COS"] = np.cos(2 * np.pi * total_mins / 1440.0)

        # C. Derive DEP_TIME_BLK if not provided in raw inference payload
        if "DEP_TIME_BLK" not in df.columns and "CRS_DEP_TIME" in df.columns:
            val = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce").fillna(0)
            hour = (val // 100).clip(0, 23).astype(int)
            df["DEP_TIME_BLK"] = hour.apply(
                lambda h: "0001-0559" if h < 6 else f"{h:02d}00-{h:02d}59"
            )

        # D. Engineered Feature: Scheduled Speed (miles per minute)
        if "DISTANCE" in df.columns and "CRS_ELAPSED_TIME" in df.columns:
            dist = pd.to_numeric(df["DISTANCE"], errors="coerce").fillna(0)
            elapsed = pd.to_numeric(df["CRS_ELAPSED_TIME"], errors="coerce").replace(0, np.nan)
            df["SCHED_SPEED_MPM"] = (dist / elapsed).fillna(0)

        return df


# 3. Full Scikit-Learn Preprocessing Pipeline Builder
def build_preprocessing_pipeline() -> Pipeline:
    categorical_cols = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "DEP_TIME_BLK",
    ]

    numerical_cols = [
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "CRS_DEP_TIME_MINS",
        "CRS_DEP_TIME_SIN",
        "CRS_DEP_TIME_COS",
        "CRS_ARR_TIME_MINS",
        "CRS_ARR_TIME_SIN",
        "CRS_ARR_TIME_COS",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "SCHED_SPEED_MPM",
    ]

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
            dtype=np.float32,
        )),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numerical_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("feature_extractor", FlightFeatureExtractor()),
        ("column_transformer", column_transformer),
    ])