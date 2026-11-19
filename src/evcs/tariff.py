"""Time-of-day (TOD) tariff bands and cost calculation."""

import pandas as pd

# Hour-of-day -> tariff band. Every hour 0-23 must be assigned exactly once.
#   off-peak  : 23:00-07:00
#   peak      : 07:00-11:00
#   mid-peak  : 11:00-19:00
#   high-peak : 19:00-23:00
#
# Note: the original thesis notebooks left hour 6 (06:00-07:00) out of the
# off-peak rule by mistake, so it fell through to "high-peak". It is
# off-peak here. Use ``ORIGINAL_NOTEBOOK_BANDS`` to reproduce the old numbers.
TOD_BANDS = {
    **{h: "off-peak" for h in [23, 0, 1, 2, 3, 4, 5, 6]},
    **{h: "peak" for h in range(7, 11)},
    **{h: "mid-peak" for h in range(11, 19)},
    **{h: "high-peak" for h in range(19, 23)},
}

ORIGINAL_NOTEBOOK_BANDS = {**TOD_BANDS, 6: "high-peak"}

BAND_ORDER = ["off-peak", "peak", "mid-peak", "high-peak"]

# Rates in INR per kWh.
TARIFF_SCENARIOS = {
    # Utility time-of-day tariff used for the station electricity bill.
    "utility_tod": {"off-peak": 6, "peak": 8, "mid-peak": 7, "high-peak": 8},
    # Two retail (customer-facing) tariff scenarios. The original notebooks
    # used scenario A for Station 1 and scenario B for Stations 2-4; both are
    # kept here so all stations can be compared on the same basis.
    "retail_tod_a": {"off-peak": 12, "peak": 18, "mid-peak": 14, "high-peak": 20},
    "retail_tod_b": {"off-peak": 10, "peak": 20, "mid-peak": 15, "high-peak": 25},
}


def assign_band(index: pd.DatetimeIndex, bands: dict = TOD_BANDS) -> pd.Series:
    """Map each timestamp to its tariff band."""
    return pd.Series(index.hour, index=index).map(bands)


def apply_tariff(
    energy: pd.DataFrame,
    rates: dict,
    bands: dict = TOD_BANDS,
) -> pd.DataFrame:
    """Add ``band`` and ``cost_inr`` columns to an interval-energy frame."""
    out = energy.copy()
    out["band"] = assign_band(out.index, bands)
    out["cost_inr"] = out["energy_kwh"] * out["band"].map(rates)
    return out


def band_summary(priced: pd.DataFrame) -> pd.DataFrame:
    """Total energy and cost per tariff band, plus a total row."""
    summary = (
        priced.groupby("band")[["energy_kwh", "cost_inr"]]
        .sum()
        .reindex(BAND_ORDER)
        .fillna(0.0)
    )
    summary.loc["total"] = summary.sum()
    summary["energy_share_pct"] = 100 * summary["energy_kwh"] / summary.loc["total", "energy_kwh"]
    return summary.round(2)
