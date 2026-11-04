"""Join weather features onto the train/val/test splits.

Reads the processed splits, attaches origin/destination weather features and
writes a parallel ``splits_weather`` directory.  The production splits and
model are left untouched, so the weather model can be compared side by side.

The join uses scheduled local times converted to UTC with each airport's IANA
timezone; every weather value comes from the archived forecast product.

Usage:
    python -m src.features.build_weather_splits \
        --splits-dir data/processed/splits \
        --output-dir data/processed/splits_weather

Optional ``--sample-train-rows 2000000`` also writes a stratified-by-size
random sample of the training split (with val/test copied) so a fast
screen can be trained without loading 11.5M rows.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download_weather import airports_for_codes, load_airports
from src.features.weather_features import (
    add_weather_features,
    join_weather_to_frame,
    weather_feature_names,
)
from src.utils.config import resolve_path

SPLITS = ("train", "val", "test")


def _coverage_accumulator(columns):
    return {"columns": list(columns), "non_null": {c: 0 for c in columns}, "rows": 0}


def _update_coverage(state, frame):
    state["rows"] += len(frame)
    for column in state["columns"]:
        if column in frame.columns:
            state["non_null"][column] += int(frame[column].notna().sum())


def _finalize_coverage(state):
    rows = max(state["rows"], 1)
    return {
        column: round(state["non_null"][column] / rows, 6)
        for column in state["columns"]
    }


def build_split(
    source: Path,
    destination: Path,
    hourly: pd.DataFrame,
    airports: pd.DataFrame,
    batch_size: int,
    sample_rows: int | None = None,
    sample_destination: Path | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    parquet = pq.ParquetFile(source)
    total_rows = parquet.metadata.num_rows
    fraction = None
    if sample_rows is not None and sample_destination is not None:
        fraction = min(1.0, sample_rows / max(total_rows, 1))
    rng = np.random.default_rng(random_state)

    writer = None
    sample_writer = None
    rows_written = 0
    sample_written = 0
    state = _coverage_accumulator(weather_feature_names())
    sample_state = _coverage_accumulator(weather_feature_names())

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            joined = join_weather_to_frame(frame, hourly, airports)
            # Keep the original column order, then append weather columns.
            weather_columns = [c for c in weather_feature_names() if c in joined.columns]
            original_columns = [c for c in frame.columns if c in joined.columns]
            joined = joined[original_columns + weather_columns]

            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
            _update_coverage(state, joined)
            rows_written += len(joined)

            if fraction is not None and sample_writer is None:
                # The sample shares the full-split schema.
                sample_destination.parent.mkdir(parents=True, exist_ok=True)
                sample_writer = pq.ParquetWriter(
                    sample_destination, table.schema, compression="snappy"
                )
            if fraction is not None:
                mask = rng.random(len(joined)) < fraction
                if mask.any():
                    sample = joined.loc[mask]
                    sample_writer.write_table(
                        pa.Table.from_pandas(sample, preserve_index=False)
                    )
                    _update_coverage(sample_state, sample)
                    sample_written += len(sample)
    finally:
        if writer is not None:
            writer.close()
        if sample_writer is not None:
            sample_writer.close()

    if rows_written != total_rows:
        raise RuntimeError(
            f"{source.name}: wrote {rows_written:,} rows, expected {total_rows:,}"
        )
    result = _finalize_coverage(state)
    print(
        f"{destination.name}: {rows_written:,} rows, "
        f"origin weather coverage={result.get('origin_temperature_2m', 0):.3f}, "
        f"dest weather coverage={result.get('dest_temperature_2m', 0):.3f}"
    )
    if fraction is not None:
        print(
            f"  sampled {sample_written:,} training rows -> "
            f"{sample_destination} (requested {sample_rows:,})"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", default="data/external/airports.dat")
    parser.add_argument(
        "--weather", default="data/external/weather/hourly_weather.parquet"
    )
    parser.add_argument("--splits-dir", default="data/processed/splits")
    parser.add_argument("--output-dir", default="data/processed/splits_weather")
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--sample-train-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    splits_dir = resolve_path(args.splits_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    airports = load_airports(args.airports)
    hourly = pd.read_parquet(resolve_path(args.weather))
    print(
        f"Loaded {len(hourly):,} hourly weather rows for "
        f"{hourly['airport'].nunique()} airports"
    )
    hourly = add_weather_features(hourly)
    # Restrict to airports that actually appear in the flights (keeps merges small).
    codes = set()
    for name in SPLITS:
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            schema = pq.ParquetFile(path).schema_arrow
            columns = [c for c in ("ORIGIN", "DEST") if c in schema.names]
            if not columns:
                continue
            table = pq.read_table(path, columns=columns)
            for column in columns:
                codes.update(table.column(column).to_pylist())
    airports = airports_for_codes(codes, airports)
    hourly = hourly[hourly["airport"].isin(set(airports["airport"]))].reset_index(
        drop=True
    )
    print(f"Using {len(airports)} airports with coordinates and {len(hourly):,} rows")

    sample_dir = (
        resolve_path(args.sample_dir)
        if args.sample_dir
        else output_dir.parent / f"{output_dir.name}_sample"
    )
    coverages = {}
    for name in SPLITS:
        source = splits_dir / f"{name}.parquet"
        if not source.exists():
            print(f"WARNING: missing split {source}, skipping")
            continue
        is_train = name == "train"
        sample_rows = args.sample_train_rows if is_train else None
        coverages[name] = build_split(
            source,
            output_dir / f"{name}.parquet",
            hourly,
            airports,
            args.batch_size,
            sample_rows=sample_rows,
            sample_destination=(
                sample_dir / "train.parquet" if sample_rows is not None else None
            ),
            random_state=args.random_state,
        )

    summary_source = splits_dir / "split_summary.json"
    if summary_source.exists():
        shutil.copy2(summary_source, output_dir / "split_summary.json")
        if args.sample_train_rows is not None:
            sample_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_source, sample_dir / "split_summary.json")
            for name in ("val", "test"):
                source = output_dir / f"{name}.parquet"
                if source.exists():
                    shutil.copy2(source, sample_dir / f"{name}.parquet")

    coverage_path = output_dir / "weather_coverage.json"
    coverage_path.write_text(
        json.dumps({"splits": coverages}, indent=2), encoding="utf-8"
    )
    print(f"Wrote weather splits to {output_dir}")
    print(f"Wrote coverage report to {coverage_path}")
    if args.sample_train_rows is not None:
        print(f"Wrote sampled training splits to {sample_dir}")


if __name__ == "__main__":
    main()
