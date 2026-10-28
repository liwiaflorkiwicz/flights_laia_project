import glob
import os
import pandas as pd

COLUMN_MAPPING = {
    "Year": "YEAR",
    "Month": "MONTH",
    "DayofMonth": "DAY_OF_MONTH",
    "DayOfWeek": "DAY_OF_WEEK",
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Origin": "ORIGIN",
    "OriginState": "ORIGIN_STATE_NM",
    "Dest": "DEST",
    "DestState": "DEST_STATE_NM",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTimeBlk": "DEP_TIME_BLK",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE",
}

# Allowed pre-departure features + target/filter columns
ALLOWED_STANDARD_COLS = set(COLUMN_MAPPING.values())


def get_severity_label(delay: float) -> str:
    if delay < 15:
        return "on_time"
    elif delay < 60:
        return "moderate_delay"
    else:
        return "severe_delay"


def process_raw_csvs(raw_dir="data/raw", out_path="data/processed/flights_2022_2023.parquet"):
    # Find all CSVs recursively inside data/raw (in case they are inside subfolders)
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV files. Starting preprocessing...")
    chunks = []

    for file in csv_files:
        print(f"Reading {os.path.basename(file)}...")
        df = pd.read_csv(file, low_memory=False)

        # Drop empty trailing columns created by BTS CSV exports (e.g. 'Unnamed: 20')
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Standardize column names to UPPER_SNAKE_CASE
        df = df.rename(columns=COLUMN_MAPPING)

        # Keep only safe pre-departure columns + filter/target columns
        keep_cols = [c for c in df.columns if c in ALLOWED_STANDARD_COLS]
        df = df[keep_cols]

        # Ensure required filter/target columns exist
        required = {"CANCELLED", "DIVERTED", "ARR_DELAY"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"File {file} is missing required columns: {missing}")

        # Project Rule: Only neither cancelled nor diverted and with valid arrival delay
        df = df[
            (df["CANCELLED"] == 0)
            & (df["DIVERTED"] == 0)
            & (df["ARR_DELAY"].notna())
        ].copy()

        # Create 3-class target variable
        df["DELAY_SEVERITY"] = df["ARR_DELAY"].apply(get_severity_label)

        # Remove target leakage & filter columns
        df = df.drop(columns=["CANCELLED", "DIVERTED", "ARR_DELAY"])

        chunks.append(df)

    full_df = pd.concat(chunks, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full_df.to_parquet(out_path, index=False, compression="snappy")
    print(f"\nSuccess! Saved {len(full_df):,} cleaned rows to {out_path}")
    print(f"Final columns: {list(full_df.columns)}")


if __name__ == "__main__":
    process_raw_csvs()