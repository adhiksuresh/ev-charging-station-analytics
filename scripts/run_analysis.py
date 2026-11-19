"""Reproduce every table and figure in results/.

Usage:
    python scripts/run_analysis.py

Stations with a raw meter file in data/raw/ get the full analysis
(tariff costs, sessions, forecasting). Cross-station figures use the
aggregated files in data/processed/, which cover all four stations.
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcs import analysis, config, forecast, io, plots, tariff  # noqa: E402


def station_analysis(station: int) -> dict:
    path = config.raw_file(station)
    raw = io.read_meter_export(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        energy = io.interval_energy(raw)
    tag = f"station_{station}"

    daily = analysis.daily_energy(energy)
    stats = analysis.summary_stats(daily)

    # Cross-check against the meter's own 30-minute block-load profile
    stats["block_load_total_kwh"] = round(io.block_load_energy(raw).sum(), 2)

    # Tariff costs for each scenario
    for name, rates in tariff.TARIFF_SCENARIOS.items():
        summary = tariff.band_summary(tariff.apply_tariff(energy, rates))
        summary.to_csv(config.TABLE_DIR / f"{tag}_tariff_{name}.csv")
        stats[f"cost_{name}_inr"] = summary.loc["total", "cost_inr"]
        if name == "utility_tod":
            plots.band_bars(summary, f"Station {station}: utility TOD tariff, March 2023",
                            config.FIGURE_DIR / f"{tag}_tariff_bands.png")
            old = tariff.band_summary(tariff.apply_tariff(energy, rates, tariff.ORIGINAL_NOTEBOOK_BANDS))
            stats["cost_utility_tod_original_bands_inr"] = old.loc["total", "cost_inr"]

    stats.update({f"weekday_weekend_{k}": v for k, v in analysis.weekday_vs_weekend(daily).items()})
    stats.update(analysis.holiday_vs_other(daily))

    sessions = analysis.charging_sessions(energy)
    sessions.to_csv(config.TABLE_DIR / f"{tag}_sessions.csv", index=False)
    stats["sessions"] = int(len(sessions))
    stats["mean_session_kwh"] = round(sessions["energy_kwh"].mean(), 2)

    analysis.anomalous_days(daily).to_csv(config.TABLE_DIR / f"{tag}_anomalous_days.csv")
    stats["data_gaps_over_20min"] = int((energy.index.to_series().diff() > pd.Timedelta("20min")).sum())

    plots.daily_bar(daily, f"Station {station}: daily energy, March 2023", config.FIGURE_DIR / f"{tag}_daily.png")
    plots.heatmap(analysis.day_hour_matrix(energy), f"Station {station}: energy by hour and day",
                  config.FIGURE_DIR / f"{tag}_heatmap.png")

    # Forecasting
    metrics, preds = forecast.evaluate(forecast.hourly_series(energy))
    metrics.insert(0, "station", station)
    stats["_forecast"] = metrics
    plots.forecast_plot(preds, f"Station {station}: hour-ahead forecast on held-out last 20% of March",
                        config.FIGURE_DIR / f"{tag}_forecast.png")
    return stats


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # --- All four stations, from aggregated data ---
    daily_all = pd.read_csv(config.PROCESSED_DIR / "daily_energy_march2023.csv", index_col="date")
    hourly_all = pd.read_csv(config.PROCESSED_DIR / "hour_of_day_energy_march2023.csv", index_col="hour")
    plots.daily_all(daily_all, config.FIGURE_DIR / "all_stations_daily.png")
    plots.hour_of_day_all(hourly_all, config.FIGURE_DIR / "all_stations_hour_of_day.png")

    overview = pd.DataFrame(
        {col: analysis.summary_stats(daily_all[col]) for col in daily_all.columns}
    ).T
    overview.loc["all_stations"] = analysis.summary_stats(daily_all.sum(axis=1))
    daily_dates = daily_all.copy()
    daily_dates.index = pd.to_datetime(daily_dates.index).date
    overview["weekday_weekend_p_value"] = [
        analysis.weekday_vs_weekend(daily_dates[c])["p_value"] for c in daily_all.columns
    ] + [analysis.weekday_vs_weekend(daily_dates.sum(axis=1))["p_value"]]
    overview.to_csv(config.TABLE_DIR / "all_stations_overview.csv")

    # --- Full analysis for stations with raw meter data ---
    rows, forecasts = {}, []
    for station in config.STATION_METERS:
        if config.raw_file(station) is None:
            print(f"Station {station}: no raw meter file, skipping detailed analysis")
            continue
        print(f"Station {station}: analysing ...")
        stats = station_analysis(station)
        forecasts.append(stats.pop("_forecast"))
        rows[f"station_{station}"] = stats

    print("\nOverview (all stations):\n", overview, sep="")
    if not rows:
        print("\nNo raw meter files found in data/raw/. Only all-station outputs were updated.")
        return

    detail = pd.DataFrame(rows).T
    detail.to_csv(config.TABLE_DIR / "station_detail.csv")
    pd.concat(forecasts).to_csv(config.TABLE_DIR / "forecast_metrics.csv")

    print("\nDetail (stations with raw data):\n", detail.T, sep="")
    print("\nForecast metrics:\n", pd.concat(forecasts), sep="")
    (config.TABLE_DIR / "run_info.json").write_text(
        json.dumps({"stations_with_raw_data": list(rows)}, indent=2)
    )


if __name__ == "__main__":
    main()
