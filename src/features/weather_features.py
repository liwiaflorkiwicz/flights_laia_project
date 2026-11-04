"""Weather feature engineering and flight-time alignment.

The functions here turn the hourly weather table produced by
:mod:`src.data.download_weather` into flight-level features that are known
before departure:

* current conditions at the **origin** at the scheduled departure hour;
* current conditions at the **destination** at the scheduled arrival hour;
* trailing 3-hour aggregates (precipitation, snow, gusts, visibility, CAPE);
* adverse-condition flags.

Times are converted from the airport-local schedule (``FL_DATE`` +
``CRS_DEP_TIME``) to UTC using each airport's IANA timezone, and
``CRS_ELAPSED_TIME`` is used for the arrival time.  Actual departure/arrival
times are never used.  All weather values are archived *forecasts*, so the
join does not leak post-departure information.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.dates import parse_flight_dates

#: Raw hourly variables stored by the download stage.
RAW_VARIABLES: list[str] = [
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

#: Features derived from a single airport-hour, before the origin/destination
#: prefix is added.
WEATHER_FEATURE_COLUMNS: list[str] = [
    *RAW_VARIABLES,
    "precip_3h_sum",
    "snow_3h_sum",
    "gust_3h_max",
    "vis_3h_min",
    "cape_3h_max",
    "flag_low_vis",
    "flag_very_low_vis",
    "flag_strong_gust",
    "flag_snow",
    "flag_thunder",
    "flag_freezing",
]

#: Thresholds for the adverse-condition flags.
LOW_VISIBILITY_M = 3000.0
VERY_LOW_VISIBILITY_M = 1000.0
STRONG_GUST_KMH = 50.0
THUNDER_WEATHER_CODE = 95.0
HIGH_CAPE_JKG = 1000.0


def add_weather_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add trailing 3-hour aggregates and adverse-weather flags.

    ``hourly`` must contain ``airport``, ``time_utc`` and the raw variables.
    The returned frame is sorted by airport/time and carries
    :data:`WEATHER_FEATURE_COLUMNS`.
    """
    frame = hourly.sort_values(["airport", "time_utc"]).reset_index(drop=True)
    for column in RAW_VARIABLES:
        if column not in frame.columns:
            frame[column] = np.nan
    grouped = frame.groupby("airport", sort=False)

    def trailing(column: str, how: str) -> pd.Series:
        rolled = getattr(grouped[column].rolling(3, min_periods=1), how)()
        rolled = rolled.reset_index(level=0, drop=True).sort_index()
        return rolled.astype("float32")

    frame["precip_3h_sum"] = trailing("precipitation", "sum")
    frame["snow_3h_sum"] = trailing("snowfall", "sum")
    frame["gust_3h_max"] = trailing("wind_gusts_10m", "max")
    frame["vis_3h_min"] = trailing("visibility", "min")
    frame["cape_3h_max"] = trailing("cape", "max")

    # Missing values are treated as "condition not observed"; the raw column
    # still carries NaN so the model can distinguish missing from benign.
    frame["flag_low_vis"] = (frame["visibility"] < LOW_VISIBILITY_M).astype("float32")
    frame["flag_very_low_vis"] = (frame["visibility"] < VERY_LOW_VISIBILITY_M).astype(
        "float32"
    )
    frame["flag_strong_gust"] = (frame["wind_gusts_10m"] >= STRONG_GUST_KMH).astype(
        "float32"
    )
    frame["flag_snow"] = (frame["snowfall"] > 0).astype("float32")
    frame["flag_thunder"] = (
        frame["weather_code"] >= THUNDER_WEATHER_CODE
    ).astype("float32")
    frame["flag_freezing"] = (frame["temperature_2m"] <= 0).astype("float32")

    for column in WEATHER_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame


def _minutes_since_midnight(values: pd.Series) -> pd.Series:
    """Convert BTS hhmm integers (e.g. 1730, 2400) to minutes."""
    values = pd.to_numeric(values, errors="coerce")
    values = values.where((values >= 0) & (values <= 2400))
    hours = (values // 100).clip(upper=24)
    minutes = values % 100
    total = hours * 60 + minutes
    return total.where(total <= 24 * 60)


def _local_to_utc(naive: pd.Series, zones: pd.Series) -> pd.Series:
    """Localise naive wall-clock times with mixed IANA timezones."""
    naive = pd.to_datetime(naive, errors="coerce")
    zones = pd.Series(zones, index=naive.index).astype("object")
    pieces = []
    for zone, subset in naive.groupby(zones):
        if not isinstance(zone, str) or not zone:
            continue
        try:
            localised = subset.dt.tz_localize(
                zone, ambiguous=True, nonexistent="shift_forward"
            )
        except Exception:
            continue
        pieces.append(localised.dt.tz_convert("UTC"))
    if not pieces:
        return pd.Series(pd.NaT, index=naive.index, dtype="datetime64[ns, UTC]")
    combined = pd.concat(pieces).sort_index()
    return combined.reindex(naive.index)


def scheduled_utc_hours(
    frame: pd.DataFrame, airports: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Return origin-departure and destination-arrival UTC hours (naive UTC).

    ``airports`` is the OpenFlights frame with ``airport`` and ``timezone``.
    Arrival is computed from the scheduled elapsed time; the scheduled arrival
    clock time is only a fallback when elapsed time is missing.
    """
    tz_map = airports.set_index("airport")["timezone"]
    origin = frame["ORIGIN"].astype(str).str.upper()
    dest = frame["DEST"].astype(str).str.upper()
    origin_zone = origin.map(tz_map)
    dest_zone = dest.map(tz_map)

    dates = parse_flight_dates(frame["FL_DATE"])
    dep_minutes = _minutes_since_midnight(frame["CRS_DEP_TIME"])
    local_departure = dates.dt.normalize() + pd.to_timedelta(dep_minutes, unit="m")
    departure_utc = _local_to_utc(local_departure, origin_zone)

    elapsed = pd.to_numeric(frame["CRS_ELAPSED_TIME"], errors="coerce")
    arrival_utc = departure_utc + pd.to_timedelta(elapsed, unit="m")

    missing_elapsed = arrival_utc.isna() & departure_utc.notna()
    if bool(missing_elapsed.any()):
        arr_minutes = _minutes_since_midnight(frame["CRS_ARR_TIME"])
        local_arrival = dates.dt.normalize() + pd.to_timedelta(arr_minutes, unit="m")
        fallback = _local_to_utc(local_arrival, dest_zone)
        # A scheduled arrival earlier than departure means the next day.
        rollover = fallback < departure_utc
        fallback = fallback.where(~rollover, fallback + pd.Timedelta(days=1))
        arrival_utc = arrival_utc.where(~missing_elapsed, fallback)

    origin_hour = departure_utc.dt.tz_localize(None).dt.floor("h")
    dest_hour = arrival_utc.dt.tz_localize(None).dt.floor("h")
    return origin_hour, dest_hour


def prefixed_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{column}" for column in WEATHER_FEATURE_COLUMNS]


def weather_feature_names() -> list[str]:
    """All weather columns produced by :func:`join_weather_to_frame`."""
    return prefixed_columns("origin") + prefixed_columns("dest")


def join_weather_to_frame(
    frame: pd.DataFrame, hourly: pd.DataFrame, airports: pd.DataFrame
) -> pd.DataFrame:
    """Attach origin/destination weather features to a batch of flights.

    The input frame is not modified.  Flights whose airport has no weather
    (or whose scheduled time is outside the downloaded range) receive NaN,
    which LightGBM handles natively.
    """
    origin_hour, dest_hour = scheduled_utc_hours(frame, airports)
    base = [
        "airport",
        "time_utc",
        *[c for c in WEATHER_FEATURE_COLUMNS if c in hourly.columns],
    ]
    weather = hourly[base].copy()

    origin_table = weather.rename(
        columns={
            "airport": "_origin_airport",
            "time_utc": "_origin_hour",
            **{c: f"origin_{c}" for c in WEATHER_FEATURE_COLUMNS if c in weather},
        }
    )
    dest_table = weather.rename(
        columns={
            "airport": "_dest_airport",
            "time_utc": "_dest_hour",
            **{c: f"dest_{c}" for c in WEATHER_FEATURE_COLUMNS if c in weather},
        }
    )

    out = frame.copy()
    out["_origin_hour"] = origin_hour.to_numpy()
    out["_dest_hour"] = dest_hour.to_numpy()
    out = out.merge(
        origin_table,
        left_on=["ORIGIN", "_origin_hour"],
        right_on=["_origin_airport", "_origin_hour"],
        how="left",
    )
    out = out.merge(
        dest_table,
        left_on=["DEST", "_dest_hour"],
        right_on=["_dest_airport", "_dest_hour"],
        how="left",
    )
    return out.drop(
        columns=["_origin_airport", "_dest_airport", "_origin_hour", "_dest_hour"]
    )


def coverage(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    """Share of non-null values per column (used to report join quality)."""
    result = {}
    for column in columns:
        if column not in frame.columns:
            result[column] = 0.0
        else:
            result[column] = float(frame[column].notna().mean())
    return result


def summarize_coverage(
    frames: dict[str, pd.DataFrame], columns: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {name: coverage(frame, columns) for name, frame in frames.items()}
