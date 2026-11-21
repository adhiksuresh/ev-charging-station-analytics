"""Synthetic cyber-attacks on 15-minute smart-meter energy series.

Each function takes a clean ``energy_kwh`` series (indexed by reading time),
returns an attacked copy and a 0/1 label series marking the readings that the
attack actually changed. Labels never mark readings that ended up unchanged
(for example, scaling a zero reading), so detection scores are not inflated
by "attacks" that did nothing.

Attack types
------------
inflate   False data injection: random readings multiplied by 1.5-3 (over-billing).
theft     False data injection: random readings multiplied by 0.1-0.5 (under-reporting).
replay    One day of earlier, genuine readings copied over a later day.
drift     A slowly growing offset added over two days (a tampered or miscalibrated meter).
"""

import numpy as np
import pandas as pd

ATTACK_TYPES = ("inflate", "theft", "replay", "drift")
PER_DAY = 96  # 15-minute readings per day


def _labels(clean: pd.Series, attacked: pd.Series) -> pd.Series:
    changed = ~np.isclose(clean.values, attacked.values, atol=1e-9)
    return pd.Series(changed.astype(int), index=clean.index, name="attack")


def inflate(x: pd.Series, rng, frac=0.10, low=1.5, high=3.0):
    """Multiply a random ``frac`` of the non-zero readings by U(low, high)."""
    out = x.copy()
    nz = np.flatnonzero(x.values > 0)
    idx = rng.choice(nz, size=max(1, int(frac * len(nz))), replace=False)
    out.iloc[idx] = out.iloc[idx] * rng.uniform(low, high, size=len(idx))
    return out, _labels(x, out)


def theft(x: pd.Series, rng, frac=0.10, low=0.1, high=0.5):
    """Multiply a random ``frac`` of the non-zero readings by U(low, high)."""
    return inflate(x, rng, frac=frac, low=low, high=high)


def replay(x: pd.Series, rng, source: pd.Series, length=PER_DAY):
    """Overwrite one day of ``x`` with one day of genuine readings from ``source``."""
    out = x.copy()
    start = rng.integers(0, len(x) - length)
    src_start = rng.integers(0, len(source) - length)
    out.iloc[start:start + length] = source.values[src_start:src_start + length]
    return out, _labels(x, out)


def drift(x: pd.Series, rng, length=2 * PER_DAY, slope=0.005):
    """Add a linearly growing offset (``slope`` kWh per reading) over ``length`` readings."""
    out = x.copy()
    start = rng.integers(0, len(x) - length)
    out.iloc[start:start + length] = out.iloc[start:start + length].values + slope * np.arange(1, length + 1)
    return out, _labels(x, out)


def inject(kind: str, x: pd.Series, rng, source: pd.Series | None = None):
    """Apply one attack type. ``source`` (clean history) is needed for ``replay``."""
    if kind == "inflate":
        return inflate(x, rng)
    if kind == "theft":
        return theft(x, rng)
    if kind == "replay":
        if source is None:
            raise ValueError("replay needs a source series")
        return replay(x, rng, source)
    if kind == "drift":
        return drift(x, rng)
    raise ValueError(f"unknown attack {kind!r}; choose from {ATTACK_TYPES}")
