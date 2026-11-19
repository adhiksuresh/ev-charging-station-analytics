"""Figures used in the README and results folder."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .tariff import BAND_ORDER  # noqa: E402

BAND_COLOURS = {"off-peak": "#4C78A8", "peak": "#F58518", "mid-peak": "#72B7B2", "high-peak": "#E45756"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def daily_bar(daily: pd.Series, title: str, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pd.to_datetime(daily.index), daily.values, color="#4C78A8")
    ax.axhline(daily.mean(), color="black", ls="--", lw=1, label=f"Mean {daily.mean():.1f} kWh")
    ax.set(title=title, xlabel="Date", ylabel="Energy (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, path)


def heatmap(matrix: pd.DataFrame, title: str, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [pd.Timestamp(d).strftime("%d %b") for d in matrix.columns]
    sns.heatmap(matrix, cmap="Reds", ax=ax, cbar_kws={"label": "Energy (kWh)"}, xticklabels=labels)
    ax.set(title=title, xlabel="Date", ylabel="Hour of day")
    _save(fig, path)


def band_bars(summary: pd.DataFrame, title: str, path):
    s = summary.loc[BAND_ORDER]
    colours = [BAND_COLOURS[b] for b in BAND_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(BAND_ORDER, s["energy_kwh"], color=colours)
    axes[0].set(title="Energy by tariff band", ylabel="kWh")
    axes[1].bar(BAND_ORDER, s["cost_inr"], color=colours)
    axes[1].set(title="Cost by tariff band", ylabel="INR")
    fig.suptitle(title)
    _save(fig, path)


def hour_of_day_all(hour_table: pd.DataFrame, path):
    """Hour-of-day energy (summed over the month) for every station."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in hour_table.columns:
        ax.plot(hour_table.index, hour_table[col], marker="o", lw=1.5, label=col.replace("_kwh", "").replace("_", " ").title())
    for start, end, colour in [(7, 11, BAND_COLOURS["peak"]), (19, 23, BAND_COLOURS["high-peak"])]:
        ax.axvspan(start - 0.5, end - 0.5, color=colour, alpha=0.1)
    ax.set(
        title="Energy by hour of day, March 2023 (shaded: peak and high-peak tariff hours)",
        xlabel="Hour of day",
        ylabel="Energy (kWh, summed over month)",
        xticks=range(0, 24, 2),
    )
    ax.legend()
    _save(fig, path)


def daily_all(daily_table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    idx = pd.to_datetime(daily_table.index)
    for col in daily_table.columns:
        ax.plot(idx, daily_table[col], marker=".", lw=1.5, label=col.replace("_kwh", "").replace("_", " ").title())
    ax.set(title="Daily energy per station, March 2023", xlabel="Date", ylabel="Energy (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, path)


def forecast_plot(preds: pd.DataFrame, title: str, path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(preds.index, preds["actual"], color="black", lw=1.8, label="Actual")
    ax.plot(preds.index, preds["seasonal_naive"], color="grey", lw=1, ls="--", label="Seasonal naive (same hour yesterday)")
    ax.plot(preds.index, preds["xgboost"], color="#E45756", lw=1.5, label="XGBoost")
    ax.set(title=title, xlabel="Time", ylabel="Energy (kWh per hour)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, path)
