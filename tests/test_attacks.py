"""Tests for attack injection and anomaly detection, on synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcs import anomaly, attacks  # noqa: E402

TZ = "Asia/Kolkata"


def _series(days=10, seed=0):
    """Daily charging pattern: 0.4 kWh per 15 min from 10:00 to 14:00, else ~0."""
    idx = pd.date_range("2023-03-01", periods=96 * days, freq="15min", tz=TZ)
    rng = np.random.default_rng(seed)
    base = np.where((idx.hour >= 10) & (idx.hour < 14), 0.4, 0.0)
    return pd.Series(np.clip(base + 0.02 * rng.standard_normal(len(idx)), 0, None), index=idx)


@pytest.mark.parametrize("kind", attacks.ATTACK_TYPES)
def test_labels_mark_exactly_the_changed_readings(kind):
    x = _series()
    rng = np.random.default_rng(1)
    att, lab = attacks.inject(kind, x.iloc[96 * 5:], rng, source=x.iloc[:96 * 5])
    changed = ~np.isclose(att.values, x.iloc[96 * 5:].values)
    assert lab.sum() > 0
    assert np.array_equal(lab.values.astype(bool), changed)


def test_inflate_and_theft_only_touch_nonzero_readings():
    x = _series()
    x.iloc[::2] = 0.0
    for kind in ("inflate", "theft"):
        att, lab = attacks.inject(kind, x, np.random.default_rng(2))
        assert (x[lab == 1] > 0).all()


def test_inject_does_not_modify_input():
    x = _series()
    before = x.copy()
    attacks.inject("drift", x, np.random.default_rng(3))
    pd.testing.assert_series_equal(x, before)


def test_features_are_causal():
    # changing a future reading must not change features of earlier readings
    x = _series()
    prof = anomaly.slot_profile(x.iloc[:96 * 5])
    f1 = anomaly.make_features(x, prof)
    y = x.copy()
    y.iloc[-1] += 5.0
    f2 = anomaly.make_features(y, prof)
    pd.testing.assert_frame_equal(f1.iloc[:-1], f2.iloc[:-1])


def test_regularise_snaps_to_15_minute_grid_and_keeps_gaps():
    idx = pd.DatetimeIndex(["2023-03-01 00:00:07", "2023-03-01 00:15:03", "2023-03-01 00:45:01"], tz=TZ)
    s = anomaly.regularise(pd.Series([1.0, 2.0, 3.0], index=idx))
    assert list(s.index.minute) == [0, 15, 30, 45]
    assert np.isnan(s.iloc[2])


def test_detector_flags_large_injection():
    x = _series(days=14)
    train, test = x.iloc[:96 * 10], x.iloc[96 * 10:]
    prof = anomaly.slot_profile(train)
    det = anomaly.ProfileZScore().fit(anomaly.make_features(train, prof).dropna())
    att = test.copy()
    att.iloc[50] = 3.0  # 3 kWh in 15 minutes, far above the pattern
    full = pd.concat([train, att])
    scores = det.score(anomaly.make_features(full, prof).loc[test.index].dropna())
    assert np.argmax(scores) == 50
