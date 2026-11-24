import numpy as np
import pandas as pd
from src.data.preprocessor import build_preprocessing_pipeline, chronological_split


def get_sample_training_df():
    return pd.DataFrame({
        "FL_DATE": ["2022-05-10", "2023-10-15", "2023-12-01"],
        "OP_UNIQUE_CARRIER": ["AA", "DL", "UA"],
        "ORIGIN": ["JFK", "ATL", "ORD"],
        "DEST": ["LAX", "MIA", "SFO"],
        "CRS_DEP_TIME": [830, 1445, 2350],
        "CRS_ARR_TIME": [1145, 1650, 155],
        "CRS_ELAPSED_TIME": [375.0, 125.0, 245.0],
        "DISTANCE": [2475.0, 594.0, 1846.0],
        "DELAY_SEVERITY": ["on_time", "moderate_delay", "severe_delay"],
    })


def test_chronological_split_no_leakage():
    df = get_sample_training_df()
    train, val, test = chronological_split(df)

    assert len(train) == 1 and train.iloc[0]["FL_DATE"] == "2022-05-10"
    assert len(val) == 1 and val.iloc[0]["FL_DATE"] == "2023-10-15"
    assert len(test) == 1 and test.iloc[0]["FL_DATE"] == "2023-12-01"


def test_unseen_categories_and_missing_values():
    train_df = get_sample_training_df()
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train_df)

    # Simulate a 2025 evaluation round with an UNSEEN airline ("NEW_AIR"),
    # an UNSEEN airport ("ZZZ"), and a missing elapsed time (None)
    eval_payload = pd.DataFrame({
        "FL_DATE": ["2025-05-15"],
        "OP_UNIQUE_CARRIER": ["NEW_AIR"],
        "ORIGIN": ["ZZZ"],
        "DEST": ["LAX"],
        "CRS_DEP_TIME": [915],
        "CRS_ARR_TIME": [1230],
        "CRS_ELAPSED_TIME": [None],
        "DISTANCE": [1000.0],
    })

    transformed = pipeline.transform(eval_payload)

    # Should output 1 row and 16 columns (4 categorical + 12 numerical) with zero NaNs
    assert transformed.shape == (1, 16)
    assert not np.isnan(transformed).any()
    # First two categorical columns (unseen carrier and unseen origin) must be encoded as -1.0
    assert transformed[0, 0] == -1.0
    assert transformed[0, 1] == -1.0