"""Cyber-attack detection on the station meter data: all tables and figures.

Usage:
    python scripts/run_attack_detection.py

Needs the raw meter exports in data/raw/ (not public). Stations without a raw
file are skipped. Writes results/tables/attack_*.csv and results/figures/attack_*.png.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evcs import anomaly, attacks, config, io  # noqa: E402

TRAIN_END = "2023-03-15"  # train: 1-14 March
VAL_END = "2023-03-22"  # validation: 15-21 March; test: 22-31 March
FALSE_ALARM_BUDGET = 0.01
SEEDS = range(5)


def load(station):
    path = config.raw_file(station)
    if path is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = io.interval_energy(io.read_meter_export(path))["energy_kwh"]
    return anomaly.regularise(e)


def split(x):
    tz = x.index.tz
    t1, t2 = pd.Timestamp(TRAIN_END, tz=tz), pd.Timestamp(VAL_END, tz=tz)
    return x[x.index < t1], x[(x.index >= t1) & (x.index < t2)], x[x.index >= t2]


def features_for(series, profile, period_index):
    """Features over the whole (possibly attacked) series, then keep ``period_index`` rows."""
    f = anomaly.make_features(series, profile).loc[period_index]
    return f.dropna()


def run_station(station, x):
    train, val, test = split(x)
    profile = anomaly.slot_profile(train.dropna())
    f_train = features_for(x, profile, train.index)
    f_val = features_for(x, profile, val.index)

    rows = []
    for seed in SEEDS:
        for name, det in anomaly.detectors(seed).items():
            det.fit(f_train)
            thr = np.quantile(det.score(f_val), 1 - FALSE_ALARM_BUDGET)
            for kind in attacks.ATTACK_TYPES:
                rng = np.random.default_rng(1000 * seed + attacks.ATTACK_TYPES.index(kind))
                clean_test = test.dropna()
                attacked_test, lab = attacks.inject(kind, clean_test, rng, source=train.dropna())
                full = x.copy()
                full.loc[attacked_test.index] = attacked_test.values
                f_test = features_for(full, profile, test.index)
                y = lab.reindex(f_test.index).fillna(0).astype(int).values
                m = anomaly.evaluate_scores(det.score(f_test), y, thr)
                rows.append({"station": station, "seed": seed, "detector": name, "attack": kind,
                             "attacked_readings": int(y.sum()), **m})
    return pd.DataFrame(rows)


def original_protocol_comparison(station, x):
    """The first notebook's setup vs the corrected protocol, on the same kind of attack.

    Original: one feature, fit and score on the same data, contamination set to
    the true attack rate (10%), every 10th-percent reading doubled.
    """
    out = []
    train, val, test = split(x)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        # original protocol on the full month
        clean = x.dropna()
        att = clean.copy()
        idx = rng.choice(len(att), size=int(0.1 * len(att)), replace=False)
        att.iloc[idx] *= 2
        y = (~np.isclose(att.values, clean.values)).astype(int)
        iso = IsolationForest(contamination=0.1, random_state=42).fit(att.values.reshape(-1, 1))
        pred = (iso.predict(att.values.reshape(-1, 1)) == -1).astype(int)
        out.append({"protocol": "original (fit and score on same data, contamination = true rate, 1 feature)",
                    "seed": seed, "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                    "f1": f1_score(y, pred)})

        # corrected protocol, same attack (x2 on 10% of test readings)
        profile = anomaly.slot_profile(train.dropna())
        f_train = features_for(x, profile, train.index)
        f_val = features_for(x, profile, val.index)
        det = anomaly.detectors(seed)["Isolation Forest"].fit(f_train)
        thr = np.quantile(det.score(f_val), 1 - FALSE_ALARM_BUDGET)
        t = test.dropna()
        a = t.copy()
        idx = rng.choice(len(a), size=int(0.1 * len(a)), replace=False)
        a.iloc[idx] *= 2
        full = x.copy()
        full.loc[a.index] = a.values
        f_test = features_for(full, profile, test.index)
        lab = pd.Series((~np.isclose(a.values, t.values)).astype(int), index=t.index)
        yy = lab.reindex(f_test.index).fillna(0).astype(int).values
        m = anomaly.evaluate_scores(det.score(f_test), yy, thr)
        out.append({"protocol": "corrected (train on clean past, threshold from clean validation, 9 features)",
                    "seed": seed, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    return pd.DataFrame(out).groupby("protocol", sort=False)[["precision", "recall", "f1"]].mean().reset_index()


def example_figure(station, x, path):
    train, val, test = split(x)
    rng = np.random.default_rng(0)
    clean = test.dropna()
    fig, axes = plt.subplots(len(attacks.ATTACK_TYPES), 1, figsize=(12, 9), sharex=True)
    for ax, kind in zip(axes, attacks.ATTACK_TYPES):
        att, lab = attacks.inject(kind, clean, rng, source=train.dropna())
        grid = clean.reindex(test.index)  # NaN in data gaps, so no line is drawn across them
        ax.plot(grid.index, grid.values, color="0.6", lw=0.8, label="genuine")
        hit = att[lab == 1]
        ax.plot(hit.index, hit.values, "o", color="C3", ms=2.5, label="attacked readings")
        ax.set_ylabel("kWh / 15 min")
        ax.set_title(kind, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Station {station}: the four injected attack types (test period, 22-31 March 2023)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_figure(summary, path):
    piv = summary.pivot(index="attack", columns="detector", values="recall").loc[list(attacks.ATTACK_TYPES)]
    ax = piv.plot(kind="bar", figsize=(10, 4.2), rot=0)
    ax.set_ylabel("recall (share of attacked readings flagged)")
    ax.set_xlabel(f"attack type  (alarm threshold: {FALSE_ALARM_BUDGET:.0%} false alarms on clean validation week)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax.set_title("Share of manipulated readings caught (mean over stations with data, 5 seeds)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in config.STATION_METERS}
    data = {s: x for s, x in data.items() if x is not None}
    if not data:
        print("No raw meter files in data/raw/; nothing to do.")
        return

    results = pd.concat([run_station(s, x) for s, x in data.items()], ignore_index=True)
    results.to_csv(config.TABLE_DIR / "attack_detection_all_runs.csv", index=False, float_format="%.4f")

    cols = ["precision", "recall", "f1", "roc_auc", "false_alarm_rate"]
    per_station = results.groupby(["station", "attack", "detector"], sort=False)[cols].mean().reset_index()
    per_station.to_csv(config.TABLE_DIR / "attack_detection_by_station.csv", index=False, float_format="%.3f")
    summary = results.groupby(["attack", "detector"], sort=False)[cols].mean().reset_index()
    summary.to_csv(config.TABLE_DIR / "attack_detection_summary.csv", index=False, float_format="%.3f")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ref = max(data, key=lambda s: data[s].sum())
    comp = original_protocol_comparison(ref, data[ref])
    comp.insert(0, "station", ref)
    comp.to_csv(config.TABLE_DIR / "attack_detection_protocol_comparison.csv", index=False, float_format="%.3f")
    print("\n", comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    example_figure(ref, data[ref], config.FIGURE_DIR / f"attack_examples_station_{ref}.png")
    summary_figure(summary, config.FIGURE_DIR / "attack_detection_recall.png")


if __name__ == "__main__":
    main()
