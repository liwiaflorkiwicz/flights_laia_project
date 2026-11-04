"""Create a smaller, reproducible sample of the temporal splits.

This is used to screen experiments quickly without touching the production
splits.  The training split is subsampled with a fixed seed; validation and
test are copied unchanged so metrics remain comparable to the full run.

Usage:
    python -m src.data.sample_splits \
        --source-dir data/processed/splits \
        --output-dir data/processed/splits_sample \
        --train-rows 2000000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import resolve_path

SPLITS = ("train", "val", "test")


def sample_parquet(
    source: Path,
    destination: Path,
    rows: int,
    batch_size: int,
    random_state: int,
) -> int:
    parquet = pq.ParquetFile(source)
    total = parquet.metadata.num_rows
    fraction = min(1.0, rows / max(total, 1))
    rng = np.random.default_rng(random_state)

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            mask = rng.random(len(frame)) < fraction
            if not mask.any():
                continue
            sampled = frame.loc[mask]
            table = pa.Table.from_pandas(sampled, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
            writer.write_table(table)
            written += len(sampled)
    finally:
        if writer is not None:
            writer.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/processed/splits")
    parser.add_argument("--output-dir", default="data/processed/splits_sample")
    parser.add_argument("--train-rows", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = sample_parquet(
        source_dir / "train.parquet",
        output_dir / "train.parquet",
        args.train_rows,
        args.batch_size,
        args.random_state,
    )
    print(f"train: sampled {written:,} rows (requested {args.train_rows:,})")

    for name in ("val", "test"):
        source = source_dir / f"{name}.parquet"
        if source.exists():
            shutil.copy2(source, output_dir / f"{name}.parquet")
            print(f"{name}: copied {source.name}")

    summary = source_dir / "split_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["sampled_train_rows"] = written
        payload["sample_random_state"] = args.random_state
        (output_dir / "split_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    print(f"Wrote sampled splits to {output_dir}")


if __name__ == "__main__":
    main()
