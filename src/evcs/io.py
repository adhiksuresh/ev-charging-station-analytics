"""Read 5G smart-meter exports and turn them into interval energy series.

The meter export is a semicolon-separated file with one row per register
reading. The column ``Reading values`` holds numbers with thousands separators
(e.g. ``"1,884.030"``).
"""

import warnings

import pandas as pd

from . import config


def read_meter_export(path) -> pd.DataFrame:
    """Load a raw meter export and parse times and values."""
    df = pd.read_csv(path, sep=";")
    df["Reading time"] = pd.to_datetime(df["Reading time"])
    df["value"] = (
        df["Reading values"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    return df


def _in_period(times: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start, tz=config.TIMEZONE)
    end_ts = pd.Timestamp(end, tz=config.TIMEZONE)
    return (times >= start_ts) & (times < end_ts)


def interval_energy(
    raw: pd.DataFrame,
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
) -> pd.DataFrame:
    """Energy drawn in each ~15-minute interval, from the cumulative import register.

    Steps:
    1. Keep only the instantaneous-profile *import* register (OBIS 1-0:1.8.0).
    2. Sort by time and drop duplicate timestamps.
    3. Take the difference between consecutive cumulative readings.
    4. Set negative differences (meter resets / glitches) to zero, with a warning.

    Returns a DataFrame indexed by reading time with columns
    ``cumulative_kwh`` and ``energy_kwh``.
    """
    reg = raw[
        (raw["Profile"] == "InstantaneousProfile")
        & (raw["OBIS code"] == config.OBIS_IMPORT_KWH)
    ]
    reg = reg[_in_period(reg["Reading time"], start, end)]
    reg = (
        reg.sort_values("Reading time")
        .drop_duplicates("Reading time")
        .set_index("Reading time")
    )

    out = pd.DataFrame({"cumulative_kwh": reg["value"]})
    diff = out["cumulative_kwh"].diff().fillna(0.0)
    n_negative = int((diff < 0).sum())
    if n_negative:
        warnings.warn(f"{n_negative} negative interval(s) set to zero", stacklevel=2)
    out["energy_kwh"] = diff.clip(lower=0.0)
    return out


def block_load_energy(
    raw: pd.DataFrame,
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
) -> pd.Series:
    """Energy per 30-minute block from the meter's own block-load profile.

    Used as an independent cross-check of :func:`interval_energy`.
    """
    blk = raw[
        (raw["Profile"] == "Block Load Profile")
        & (raw["OBIS code"] == config.OBIS_BLOCK_IMPORT_KWH)
    ]
    blk = blk[_in_period(blk["Reading time"], start, end)]
    return blk.sort_values("Reading time").drop_duplicates("Reading time").set_index(
        "Reading time"
    )["value"]


def load_station(station: int) -> pd.DataFrame:
    """Convenience wrapper: interval energy for one station."""
    path = config.raw_file(station)
    if path is None:
        raise FileNotFoundError(
            f"No raw meter file for station {station}. "
            f"Expected data/raw/{config.STATION_METERS.get(station)}.csv"
        )
    return interval_energy(read_meter_export(path))
