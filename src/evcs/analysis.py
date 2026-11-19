"""Descriptive analytics on interval energy: daily totals, load profiles,
weekday/weekend and holiday comparisons, anomalous days and charging sessions."""

import pandas as pd
from scipy import stats

from . import config


def daily_energy(energy: pd.DataFrame) -> pd.Series:
    """Total kWh per calendar day."""
    return energy["energy_kwh"].groupby(energy.index.date).sum().rename("energy_kwh")


def hour_of_day_energy(energy: pd.DataFrame) -> pd.Series:
    """Total kWh for each hour of the day, summed over the whole period."""
    return (
        energy["energy_kwh"]
        .groupby(energy.index.hour)
        .sum()
        .reindex(range(24), fill_value=0.0)
        .rename("energy_kwh")
    )


def day_hour_matrix(energy: pd.DataFrame) -> pd.DataFrame:
    """kWh in a (hour of day x date) grid, for heatmaps."""
    grid = energy["energy_kwh"].groupby([energy.index.hour, energy.index.date]).sum()
    return grid.unstack(fill_value=0.0).reindex(range(24), fill_value=0.0)


def summary_stats(daily: pd.Series) -> dict:
    """Mean, median, standard deviation, min and max of daily energy."""
    return {
        "total_kwh": round(float(daily.sum()), 2),
        "mean_daily_kwh": round(float(daily.mean()), 2),
        "median_daily_kwh": round(float(daily.median()), 2),
        "std_daily_kwh": round(float(daily.std()), 2),
        "min_daily_kwh": round(float(daily.min()), 2),
        "max_daily_kwh": round(float(daily.max()), 2),
    }


def weekday_vs_weekend(daily: pd.Series, alpha: float = 0.05) -> dict:
    """Compare daily energy on weekdays vs weekends with Welch's t-test.

    The test is run on *daily totals*, not on individual 15-minute readings:
    readings within a day are strongly correlated, so treating them as
    independent samples overstates significance.
    """
    idx = pd.to_datetime(daily.index)
    is_weekend = idx.dayofweek >= 5
    weekday, weekend = daily[~is_weekend], daily[is_weekend]
    t_stat, p_value = stats.ttest_ind(weekday, weekend, equal_var=False)
    return {
        "weekday_days": int(len(weekday)),
        "weekend_days": int(len(weekend)),
        "weekday_mean_kwh": round(float(weekday.mean()), 2),
        "weekend_mean_kwh": round(float(weekend.mean()), 2),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
    }


def holiday_vs_other(daily: pd.Series, holidays=config.HOLIDAYS) -> dict:
    """Mean daily energy on public holidays vs all other days."""
    holiday_dates = set(pd.to_datetime(holidays).date)
    mask = pd.Index(daily.index).isin(holiday_dates)
    return {
        "holiday_days": int(mask.sum()),
        "holiday_mean_kwh": round(float(daily[mask].mean()), 2),
        "other_mean_kwh": round(float(daily[~mask].mean()), 2),
    }


def anomalous_days(daily: pd.Series, k: float = 1.0) -> pd.DataFrame:
    """Days whose total energy is more than ``k`` standard deviations from the mean."""
    mean, std = daily.mean(), daily.std()
    out = pd.DataFrame({"energy_kwh": daily})
    out["flag"] = ""
    out.loc[daily > mean + k * std, "flag"] = "high"
    out.loc[daily < mean - k * std, "flag"] = "low"
    return out[out["flag"] != ""]


def charging_sessions(
    energy: pd.DataFrame,
    active_kwh: float = 0.0,
    min_session_kwh: float = 0.25,
    max_gap: str = "20min",
) -> pd.DataFrame:
    """Approximate charging sessions from interval energy.

    A session is a run of consecutive intervals with energy above
    ``active_kwh``. A data gap longer than ``max_gap`` ends a session.
    Sessions below ``min_session_kwh`` in total are discarded as noise.

    Limitation: with 15-minute meter data, two vehicles charging back to back
    look like one session, so this is a lower bound on true session counts.
    """
    e = energy["energy_kwh"]
    active = e > active_kwh
    gap = e.index.to_series().diff() > pd.Timedelta(max_gap)
    new_run = active & (~active.shift(fill_value=False) | gap)
    run_id = new_run.cumsum().where(active)

    grouped = e[active].groupby(run_id[active])
    sessions = pd.DataFrame(
        {
            "start": grouped.apply(lambda s: s.index.min()),
            "end": grouped.apply(lambda s: s.index.max()),
            "energy_kwh": grouped.sum(),
            "intervals": grouped.size(),
        }
    )
    sessions = sessions[sessions["energy_kwh"] >= min_session_kwh].reset_index(drop=True)
    return sessions
