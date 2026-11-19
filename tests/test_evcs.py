"""Unit tests on small synthetic data, so they run without the private meter files."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcs import analysis, forecast, io, tariff  # noqa: E402

TZ = "Asia/Kolkata"


def _fake_export(values, obis="1-0:1.8.0*255", start="2023-03-01 00:00"):
    times = pd.date_range(start, periods=len(values), freq="15min", tz=TZ)
    return pd.DataFrame(
        {
            "Reading values": [f"{v:,.3f}" for v in values],
            "Profile": "InstantaneousProfile",
            "Reading time": times,
            "OBIS code": obis,
            "value": values,
        }
    )


def test_every_hour_has_exactly_one_band():
    assert sorted(tariff.TOD_BANDS) == list(range(24))
    assert set(tariff.TOD_BANDS.values()) == set(tariff.BAND_ORDER)


def test_hour_six_is_off_peak():
    assert tariff.TOD_BANDS[6] == "off-peak"
    assert tariff.ORIGINAL_NOTEBOOK_BANDS[6] == "high-peak"


def test_interval_energy_uses_import_register_only():
    imp = _fake_export([100.0, 101.0, 103.0, 103.0])
    exp = _fake_export([0.0, 0.0, 0.0, 0.0], obis="1-0:2.8.0*255")
    raw = pd.concat([imp, exp])  # export rows after import rows, as in the real files
    e = io.interval_energy(raw)
    assert len(e) == 4  # export rows must not double the row count
    assert e["energy_kwh"].sum() == pytest.approx(3.0)


def test_negative_jump_is_clipped_with_warning():
    raw = _fake_export([100.0, 101.0, 50.0, 51.0])
    with pytest.warns(UserWarning):
        e = io.interval_energy(raw)
    assert (e["energy_kwh"] >= 0).all()
    assert e["energy_kwh"].sum() == pytest.approx(2.0)


def test_cost_matches_hand_calculation():
    idx = pd.DatetimeIndex(["2023-03-01 03:00", "2023-03-01 08:00"], tz=TZ)
    e = pd.DataFrame({"energy_kwh": [2.0, 1.0]}, index=idx)
    priced = tariff.apply_tariff(e, tariff.TARIFF_SCENARIOS["utility_tod"])
    assert priced["cost_inr"].sum() == pytest.approx(2 * 6 + 1 * 8)


def test_sessions_split_on_idle_and_drop_small():
    vals = [0, 1, 1, 0, 0, 0.1, 0, 2, 0]
    idx = pd.date_range("2023-03-01", periods=len(vals), freq="15min", tz=TZ)
    e = pd.DataFrame({"energy_kwh": vals}, index=idx)
    s = analysis.charging_sessions(e, min_session_kwh=0.25)
    assert list(s["energy_kwh"]) == [2.0, 2.0]


def test_rolling_features_do_not_leak_target():
    idx = pd.date_range("2023-03-01", periods=120, freq="h", tz=TZ)
    y = pd.Series(np.arange(120, dtype=float), index=idx)
    f = forecast.make_features(y)
    # roll_mean_3 at time t must equal the mean of y at t-1, t-2, t-3
    t = f.index[10]
    expected = y.loc[:t].iloc[-4:-1].mean()
    assert f.loc[t, "roll_mean_3"] == pytest.approx(expected)
    assert (f["roll_mean_3"] < f["y"]).all()
