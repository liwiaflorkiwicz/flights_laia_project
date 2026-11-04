"""Download archived hourly weather forecasts for every airport in the dataset.

Source: Open-Meteo **historical forecast** API
(``https://historical-forecast-api.open-meteo.com/v1/forecast``).

This is the archived *forecast* product, not a reanalysis.  Every value is a
forecast that was issued before the valid time, so joining it to a flight by
its scheduled departure/arrival hour uses only information that was available
before departure.  The reanalysis archive (``archive-api``) is deliberately
**not** used: it returns no visibility and represents what happened, not what
was forecast.

Downloads are batched (several coordinates per request) and written as one
Parquet file per batch, so the stage is resumable.  Re-running skips batches
that already exist unless ``--force`` is passed.

Usage:
    python -m src.data.download_weather \
        --start 2022-01-01 --end 2023-12-31 --batch-size 10
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from src.utils.config import resolve_path

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Variables kept for feature engineering.  ``rain`` / ``snow_depth`` /
# ``wind_direction_10m`` are intentionally omitted: they are redundant with
# precipitation / snowfall / wind speed and would only add noise.
HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "visibility",
    "cape",
]

AIRPORTS_PATH = Path("data/external/airports.dat")
DEFAULT_OUTPUT_DIR = Path("data/external/weather/hourly")


def load_airports(path: str | Path = AIRPORTS_PATH) -> pd.DataFrame:
    """Read OpenFlights ``airports.dat`` and keep IATA-keyed coordinates.

    OpenFlights columns: 0 id, 1 name, 2 city, 3 country, 4 IATA, 5 ICAO,
    6 latitude, 7 longitude, 8 altitude, 9 UTC offset, 10 DST flag,
    11 IANA timezone, ...
    """
    path = resolve_path(path)
    rows: list[dict] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.reader(fh):
            if len(raw) < 12:
                continue
            iata, icao = raw[4].strip(), raw[5].strip()
            if not iata or iata == "\\N" or iata in seen:
                continue
            try:
                lat, lon = float(raw[6]), float(raw[7])
            except ValueError:
                continue
            tz = raw[11].strip()
            if not tz or tz == "\\N":
                continue
            seen.add(iata)
            rows.append(
                {
                    "airport": iata.upper(),
                    "icao": "" if icao == "\\N" else icao.upper(),
                    "latitude": lat,
                    "longitude": lon,
                    "timezone": tz,
                }
            )
    airports = pd.DataFrame(rows)
    if airports.empty:
        raise ValueError(f"No usable airports parsed from {path}")
    return airports


def airports_for_codes(codes, airports: pd.DataFrame) -> pd.DataFrame:
    """Return coordinates/timezones for the airports present in the dataset."""
    wanted = {str(code).upper() for code in codes if code not in (None, "", "nan")}
    selected = airports[airports["airport"].isin(wanted)].reset_index(drop=True)
    missing = sorted(wanted - set(selected["airport"]))
    if missing:
        print(f"WARNING: no coordinates for {len(missing)} airports: {missing}")
    return selected


def flight_airport_counts(path: str | Path) -> pd.Series:
    """Count origin+destination appearances per airport in the flight data."""
    path = resolve_path(path)
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    elif path.exists():
        files = [path]
    else:
        raise FileNotFoundError(f"Flight data not found: {path}")

    counts: dict[str, int] = {}
    for file in files:
        schema = pq.ParquetFile(file).schema_arrow
        columns = [c for c in ("ORIGIN", "DEST") if c in schema.names]
        if not columns:
            continue
        table = pq.read_table(file, columns=columns)
        for column in columns:
            values = pd.Series(table.column(column).to_pylist()).dropna().astype(str)
            for code, count in values.value_counts().items():
                code = code.upper()
                counts[code] = counts.get(code, 0) + int(count)
    if not counts:
        raise ValueError(f"No ORIGIN/DEST codes found in {path}")
    return pd.Series(counts).sort_values(ascending=False)


def flight_airport_codes(path: str | Path) -> set[str]:
    """Collect ORIGIN/DEST codes from a processed Parquet file or split dir.

    Restricting the download to airports that actually appear in the data is
    important: OpenFlights lists thousands of airports worldwide, while the
    US flight data uses fewer than 400.  Downloading all of them wastes API
    quota and disk space.
    """
    return set(flight_airport_counts(path).index)


def top_flight_airports(path: str | Path, count: int) -> set[str]:
    """Return the busiest ``count`` airports by flight endpoints."""
    return set(flight_airport_counts(path).head(count).index)


def _request_batch(
    batch: pd.DataFrame,
    start: str,
    end: str,
    variables: list[str],
    session: requests.Session,
    max_retries: int = 6,
) -> list[dict]:
    params = {
        "latitude": ",".join(f"{v:.6f}" for v in batch["latitude"]),
        "longitude": ",".join(f"{v:.6f}" for v in batch["longitude"]),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    delay = 5.0
    for attempt in range(1, max_retries + 1):
        response = session.get(API_URL, params=params, timeout=300)
        if response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, list) else [payload]
        text = response.text[:200]
        # 429 is the minute-rate limit; wait a full minute before retrying.
        if response.status_code == 429:
            wait = max(delay, 60.0)
            print(
                f"  HTTP 429 (attempt {attempt}/{max_retries}), retrying in "
                f"{wait:.0f}s: {text}",
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 2, 180)
            continue
        # Other 5xx responses are transient.
        if response.status_code in (500, 502, 503, 504):
            print(
                f"  HTTP {response.status_code} (attempt {attempt}/{max_retries}), "
                f"retrying in {delay:.0f}s: {text}",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        raise RuntimeError(f"Open-Meteo request failed ({response.status_code}): {text}")
    raise RuntimeError("Open-Meteo request failed after retries")


def _locations_to_frame(
    locations: list[dict], batch: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    if len(locations) != len(batch):
        raise RuntimeError(
            f"Expected {len(batch)} locations in response, got {len(locations)}"
        )
    frames = []
    for (_, airport), location in zip(batch.iterrows(), locations):
        hourly = location.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            print(f"  WARNING: empty hourly payload for {airport['airport']}")
            continue
        data = {
            "airport": airport["airport"],
            "time_utc": pd.to_datetime(times),
        }
        for variable in variables:
            values = hourly.get(variable)
            data[variable] = values if values is not None else [None] * len(times)
        frames.append(pd.DataFrame(data))
    if not frames:
        return pd.DataFrame(columns=["airport", "time_utc", *variables])
    frame = pd.concat(frames, ignore_index=True)
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce").astype(
            "float32"
        )
    return frame


def _download_batch(
    batch_index: int,
    n_batches: int,
    batch: pd.DataFrame,
    path: Path,
    start: str,
    end: str,
    variables: list[str],
    force: bool,
) -> tuple[Path, int, float]:
    if path.exists() and not force:
        print(
            f"batch {batch_index + 1}/{n_batches}: exists, skipping "
            f"({len(batch)} airports)",
            flush=True,
        )
        return path, len(batch), 0.0

    print(
        f"batch {batch_index + 1}/{n_batches}: {len(batch)} airports "
        f"{batch['airport'].iloc[0]}..{batch['airport'].iloc[-1]}",
        flush=True,
    )
    started = time.perf_counter()
    session = requests.Session()
    try:
        locations = _request_batch(batch, start, end, variables, session)
    finally:
        session.close()
    frame = _locations_to_frame(locations, batch, variables)
    frame.to_parquet(path, index=False)
    elapsed = time.perf_counter() - started
    print(
        f"  wrote {len(frame):,} rows to {path.name} in {elapsed:.1f}s",
        flush=True,
    )
    return path, len(batch), elapsed


def download(
    airports: pd.DataFrame,
    output_dir: str | Path,
    start: str,
    end: str,
    batch_size: int,
    variables: list[str],
    force: bool = False,
    workers: int = 1,
    sleep_between: float = 0.0,
) -> list[Path]:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [
        airports.iloc[start_index : start_index + batch_size].reset_index(drop=True)
        for start_index in range(0, len(airports), batch_size)
    ]
    paths = [output_dir / f"batch_{index:03d}.parquet" for index in range(len(batches))]
    n_batches = len(batches)

    if workers <= 1:
        for index, batch in enumerate(batches):
            _download_batch(
                index, n_batches, batch, paths[index], start, end, variables, force
            )
            if sleep_between and index < len(batches) - 1:
                time.sleep(sleep_between)
        return paths

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_batch,
                index,
                n_batches,
                batch,
                paths[index],
                start,
                end,
                variables,
                force,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            future.result()  # surface any failed batch
    return paths


def combine(paths: list[Path], output_path: str | Path) -> Path:
    """Concatenate batch files into one sorted, de-duplicated hourly table."""
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No weather batch files were produced")
    frame = pd.concat(frames, ignore_index=True)
    frame = (
        frame.drop_duplicates(["airport", "time_utc"])
        .sort_values(["airport", "time_utc"])
        .reset_index(drop=True)
    )
    frame.to_parquet(output_path, index=False)
    print(
        f"Combined {len(paths)} batches into {output_path} "
        f"({len(frame):,} rows, {frame['airport'].nunique()} airports)"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", default=str(AIRPORTS_PATH))
    parser.add_argument(
        "--flight-data",
        default="data/processed/flights_2022_2023.parquet",
        help="Parquet file or split directory used to restrict airports.",
    )
    parser.add_argument(
        "--all-airports",
        action="store_true",
        help="Download every airport in the OpenFlights file (very large).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--combined", default="data/external/weather/hourly_weather.parquet")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--top-airports",
        type=int,
        default=None,
        help=(
            "Download only the busiest N airports by flight volume. "
            "The free Open-Meteo tier counts a two-year request as ~52 calls "
            "per airport, so ~100 airports is the practical daily budget."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between sequential batches (rate limiting).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of batches downloaded concurrently.",
    )
    parser.add_argument("--max-airports", type=int, default=None)
    parser.add_argument(
        "--codes",
        default=None,
        help="Optional comma-separated airport codes to restrict the download.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    airports = load_airports(args.airports)
    if args.codes:
        airports = airports_for_codes(args.codes.split(","), airports)
    elif args.top_airports:
        codes = top_flight_airports(args.flight_data, args.top_airports)
        airports = airports_for_codes(codes, airports)
        print(
            f"Restricted to the top {args.top_airports} airports by flight volume "
            f"({len(airports)} with coordinates)"
        )
    elif not args.all_airports:
        codes = flight_airport_codes(args.flight_data)
        airports = airports_for_codes(codes, airports)
        print(
            f"Restricted to {len(airports)} airports used by {args.flight_data} "
            f"(use --all-airports to override)"
        )
    if args.max_airports is not None:
        airports = airports.iloc[: args.max_airports].reset_index(drop=True)
    print(f"Downloading weather for {len(airports)} airports")
    print(f"  period: {args.start} -> {args.end}")
    print(f"  variables: {', '.join(HOURLY_VARIABLES)}")

    paths = download(
        airports,
        args.output_dir,
        args.start,
        args.end,
        args.batch_size,
        HOURLY_VARIABLES,
        force=args.force,
        workers=args.workers,
        sleep_between=args.sleep,
    )
    combine(paths, args.combined)


if __name__ == "__main__":
    main()
