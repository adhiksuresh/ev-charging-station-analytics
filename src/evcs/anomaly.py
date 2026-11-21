"""Unsupervised detection of manipulated smart-meter readings.

Protocol (no information about the attacks is used to train or tune):

1. Split the month by time: **train** (clean), **validation** (clean), **test**.
2. Fit each detector on train readings only.
3. Set its alarm threshold so that 1% of *clean validation* readings raise an
   alarm (a fixed false-alarm budget).
4. Inject one attack type into the test period and score the attacked readings.

Features are computed only from the current and past readings (no look-ahead),
plus the time of day and the typical (training-period) energy for that time slot.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PER_DAY = 96
FEATURES = [
    "energy", "delta", "mean_1h", "std_1h", "mean_4h",
    "dev_slot_profile", "dev_yesterday", "hour_sin", "hour_cos",
]


def regularise(energy: pd.Series) -> pd.Series:
    """Put readings on an exact 15-minute grid (meter timestamps jitter by seconds).

    Missing slots are left as NaN and dropped later.
    """
    s = energy.copy()
    s.index = s.index.floor("15min")
    s = s[~s.index.duplicated(keep="first")]
    full = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    return s.reindex(full)


def slot_profile(train: pd.Series) -> pd.Series:
    """Mean energy for each of the 96 time-of-day slots in the training period."""
    slot = train.index.hour * 4 + train.index.minute // 15
    return train.groupby(slot).mean()


def make_features(x: pd.Series, profile: pd.Series) -> pd.DataFrame:
    """Causal features for every reading of ``x``."""
    slot = x.index.hour * 4 + x.index.minute // 15
    f = pd.DataFrame(index=x.index)
    f["slot"] = slot
    f["energy"] = x
    f["delta"] = x.diff()
    f["mean_1h"] = x.rolling(4, min_periods=1).mean()
    f["std_1h"] = x.rolling(4, min_periods=2).std()
    f["mean_4h"] = x.rolling(16, min_periods=1).mean()
    f["dev_slot_profile"] = x.values - profile.reindex(slot).values
    f["dev_yesterday"] = x - x.shift(PER_DAY)
    ang = 2 * np.pi * slot / PER_DAY
    f["hour_sin"], f["hour_cos"] = np.sin(ang), np.cos(ang)
    return f


class ProfileZScore:
    """Baseline: |reading - typical reading for that time slot|, scaled by the slot's spread."""

    def fit(self, f: pd.DataFrame):
        self.sd = f["dev_slot_profile"].groupby(f["slot"]).std().clip(lower=1e-3)
        self.global_sd = float(f["dev_slot_profile"].std())
        return self

    def score(self, f: pd.DataFrame) -> np.ndarray:
        sd = self.sd.reindex(f["slot"].values).fillna(self.global_sd).values
        return np.abs(f["dev_slot_profile"].values) / sd


class SklearnDetector:
    """Wrap an sklearn novelty detector so that higher score = more anomalous."""

    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def fit(self, f: pd.DataFrame):
        self.model.fit(self.scaler.fit_transform(f[FEATURES].values))
        return self

    def score(self, f: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self.scaler.transform(f[FEATURES].values))


def detectors(seed=0) -> dict:
    return {
        "Time-slot z-score (baseline)": ProfileZScore(),
        "Isolation Forest": SklearnDetector(IsolationForest(n_estimators=300, random_state=seed)),
        "Local Outlier Factor": SklearnDetector(LocalOutlierFactor(n_neighbors=35, novelty=True)),
        "One-Class SVM": SklearnDetector(OneClassSVM(nu=0.05, gamma="scale")),
    }


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
    out = {
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan,
        "false_alarm_rate": float(pred[labels == 0].mean()) if (labels == 0).any() else np.nan,
    }
    return out
