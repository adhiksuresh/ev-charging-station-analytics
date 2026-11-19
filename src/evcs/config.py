"""Project-wide settings: station-to-meter mapping, study period and file locations."""

from pathlib import Path

# Repository root (two levels up from this file: src/evcs/config.py)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

# Each charging station is metered by one 5G smart meter.
# Station numbering follows the original analysis notebooks (and the station
# totals in the NPSC paper, Table II). Note: the paper's meter table (Table I)
# lists meters 175 and 174 in the opposite order.
# Station 4 (meter 10050180) raw export is not included in this repository yet.
STATION_METERS = {
    1: "IHM10050174",
    2: "IHM10050175",
    3: "IHM10050178",
    4: "IHM10050180",
}

TIMEZONE = "Asia/Kolkata"
STUDY_START = "2023-03-01"
STUDY_END = "2023-04-01"  # exclusive

# Public holidays in the study period (Holi and Ugadi, 2023)
HOLIDAYS = ["2023-03-08", "2023-03-22"]

# DLMS/COSEM OBIS code for the cumulative active-energy IMPORT register (kWh).
# 1-0:2.8.0 is the EXPORT register and must not be mixed in.
OBIS_IMPORT_KWH = "1-0:1.8.0*255"
OBIS_BLOCK_IMPORT_KWH = "1-0:1.29.0*255"  # per-block (30 min) import energy


def raw_file(station: int) -> Path | None:
    """Return the raw meter export path for a station, or None if not available."""
    meter = STATION_METERS.get(station)
    if meter is None:
        return None
    path = RAW_DIR / f"{meter}.csv"
    return path if path.exists() else None
