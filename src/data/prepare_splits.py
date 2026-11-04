"""Create reproducible train / validation / test splits.

The default strategy is **temporal**: train on 2022 + Jan-Sep 2023,
validate on Oct-Nov 2023 and test on Dec 2023.  This mirrors the
centralised evaluation, which always scores the model on *future*
flights, and therefore avoids temporal leakage.

A random ``stratified`` strategy is also available for ablation studies.
It is **not** the recommended setting: shuffling flights lets the model
see future months during training and inflates offline scores.

The 13M-row Parquet file is processed in batches, so the script runs on
machines with limited RAM.

Usage:
    python -m src.data.prepare_splits --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.utils.config import load_config, resolve_path
from src.utils.dates import parse_flight_dates

SPLITS = ("train", "val", "test")
LABELS = ("on_time", "moderate_delay", "severe_delay")


def temporal_masks(
    dates: pd.Series, train_end: str, val_end: str
) -> dict[str, np.ndarray]:
    train_end_ts = pd.Timestamp(train_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    val_end_ts = pd.Timestamp(val_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return {
        "train": (dates <= train_end_ts).to_numpy(),
        "val": ((dates > train_end_ts) & (dates <= val_end_ts)).to_numpy(),
        "test": (dates > val_end_ts).to_numpy(),
    }


def stratified_masks(
    target: pd.Series,
    val_size: float,
    test_size: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Class-balanced random split (diagnostics only, not for deployment)."""
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    indices = np.arange(len(target))
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=val_size + test_size,
        stratify=target,
        random_state=random_state,
    )
    test_fraction_of_rest = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_fraction_of_rest,
        stratify=target.iloc[rest_idx],
        random_state=random_state,
    )

    masks = {name: np.zeros(len(target), dtype=bool) for name in SPLITS}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        masks[name][idx] = True
    return masks


def validate_masks(masks: dict[str, np.ndarray]) -> None:
    if any(mask.sum() == 0 for mask in masks.values()):
        empty = [name for name, mask in masks.items() if mask.sum() == 0]
        raise ValueError(f"Empty split(s): {empty}. Adjust the split boundaries.")

    overlaps = (
        masks["train"].astype(np.int8)
        + masks["val"].astype(np.int8)
        + masks["test"].astype(np.int8)
    )
    if overlaps.max() > 1:
        raise ValueError("Splits overlap")


def _empty_stats() -> dict:
    return {
        "rows": 0,
        "date_min": None,
        "date_max": None,
        "class_counts": {label: 0 for label in LABELS},
    }


def _update_stats(stats: dict, target_values: pd.Series, dates: pd.Series | None) -> None:
    stats["rows"] += int(len(target_values))
    for label, count in target_values.value_counts().items():
        stats["class_counts"][str(label)] = stats["class_counts"].get(str(label), 0) + int(count)
    if dates is not None and len(dates):
        batch_min, batch_max = dates.min(), dates.max()
        if pd.notna(batch_min):
            stats["date_min"] = (
                batch_min if stats["date_min"] is None else min(stats["date_min"], batch_min)
            )
        if pd.notna(batch_max):
            stats["date_max"] = (
                batch_max if stats["date_max"] is None else max(stats["date_max"], batch_max)
            )


def _build_stratified_codes(
    parquet: pq.ParquetFile,
    target_col: str,
    n_rows: int,
    val_size: float,
    test_size: float,
    random_state: int,
    batch_size: int,
) -> np.ndarray:
    """Assign split codes per row using an exact stratified random split."""
    class_codes = {label: index for index, label in enumerate(LABELS)}
    target_codes = np.empty(n_rows, dtype=np.int8)
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[target_col]):
        values = batch.column(0).to_pandas().astype(str)
        target_codes[offset : offset + len(values)] = values.map(class_codes).to_numpy(dtype=np.int8)
        offset += len(values)

    split_codes = np.zeros(n_rows, dtype=np.int8)  # 0=train, 1=val, 2=test
    rng = np.random.default_rng(random_state)
    for code in range(len(LABELS)):
        indices = np.flatnonzero(target_codes == code)
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_size))
        n_val = int(round(len(indices) * val_size))
        split_codes[indices[:n_test]] = 2
        split_codes[indices[n_test : n_test + n_val]] = 1
    return split_codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--strategy",
        choices=["temporal", "stratified"],
        default=None,
        help="Override the strategy from the config file.",
    )
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    split_cfg = data_cfg["split"]
    target_col = data_cfg.get("target", "DELAY_SEVERITY")
    strategy = args.strategy or split_cfg.get("strategy", "temporal")

    processed_path = resolve_path(data_cfg["processed_path"])
    split_dir = resolve_path(data_cfg["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(processed_path)
    n_rows = parquet.metadata.num_rows
    if target_col not in parquet.schema_arrow.names:
        raise KeyError(f"Target column {target_col!r} not found in {processed_path}")
    print(f"Streaming {n_rows:,} rows from {processed_path} ({strategy} split) ...")

    split_codes = None
    if strategy == "stratified":
        split_codes = _build_stratified_codes(
            parquet,
            target_col,
            n_rows,
            float(split_cfg.get("val_size", 0.15)),
            float(split_cfg.get("test_size", 0.15)),
            int(split_cfg.get("random_state", 42)),
            args.batch_size,
        )

    writers = {
        name: pq.ParquetWriter(
            split_dir / f"{name}.parquet",
            parquet.schema_arrow,
            compression="snappy",
        )
        for name in SPLITS
    }
    stats = {name: _empty_stats() for name in SPLITS}

    offset = 0
    try:
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            frame = batch.to_pandas()
            dates = parse_flight_dates(frame["FL_DATE"])

            if strategy == "temporal":
                masks = temporal_masks(
                    dates, split_cfg["train_end"], split_cfg["val_end"]
                )
                batch_codes = np.select(
                    [masks["train"], masks["val"]], [0, 1], default=2
                ).astype(np.int8)
            else:
                batch_codes = split_codes[offset : offset + len(frame)]

            for code, name in enumerate(SPLITS):
                mask = batch_codes == code
                if not mask.any():
                    continue
                writers[name].write_table(
                    pa.Table.from_batches(
                        [batch.filter(pa.array(mask))], schema=parquet.schema_arrow
                    )
                )
                _update_stats(stats[name], frame.loc[mask, target_col], dates[mask])

            offset += len(frame)
            if offset % (args.batch_size * 10) < args.batch_size:
                print(f"  processed {offset:,}/{n_rows:,} rows")
    finally:
        for writer in writers.values():
            writer.close()

    if offset != n_rows:
        raise RuntimeError(f"Processed {offset:,} rows but expected {n_rows:,}")
    if any(info["rows"] == 0 for info in stats.values()):
        raise ValueError("At least one split is empty; adjust the date boundaries")

    boundaries = (
        {
            "train_end": split_cfg["train_end"],
            "val_end": split_cfg["val_end"],
            "test": f"> {split_cfg['val_end']}",
        }
        if strategy == "temporal"
        else {
            "train": f"random {1 - split_cfg.get('val_size', 0.15) - split_cfg.get('test_size', 0.15):.0%}",
            "val": f"random {split_cfg.get('val_size', 0.15):.0%}",
            "test": f"random {split_cfg.get('test_size', 0.15):.0%}",
        }
    )

    summary = {
        "strategy": strategy,
        "boundaries": boundaries,
        "target": target_col,
        "labels": list(LABELS),
        "splits": {},
    }
    for name in SPLITS:
        info = stats[name]
        rows = max(info["rows"], 1)
        summary["splits"][name] = {
            "rows": info["rows"],
            "date_min": str(info["date_min"].date()) if info["date_min"] is not None else None,
            "date_max": str(info["date_max"].date()) if info["date_max"] is not None else None,
            "class_counts": info["class_counts"],
            "class_share": {
                label: round(count / rows, 6)
                for label, count in info["class_counts"].items()
            },
        }
        print(
            f"{name:5s}: {info['rows']:>10,} rows  "
            f"[{summary['splits'][name]['date_min']} -> {summary['splits'][name]['date_max']}]  "
            f"shares={summary['splits'][name]['class_share']}"
        )

    summary_path = split_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote splits to {split_dir}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
