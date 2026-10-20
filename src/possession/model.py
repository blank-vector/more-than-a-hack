"""LightGBM frame classifier + Viterbi smoothing."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=0,
)


def train(X: pd.DataFrame, y: np.ndarray, rounds: int = 400, params: dict | None = None) -> lgb.Booster:
    # subsample: neighbouring frames are near-duplicates
    idx = np.arange(0, len(y), 5)
    ds = lgb.Dataset(X.iloc[idx], y[idx])
    return lgb.train({**PARAMS, **(params or {})}, ds, num_boost_round=rounds)


def predict_proba(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X)


def viterbi(proba: np.ndarray, switch_penalty: float = 8.0) -> np.ndarray:
    """Most likely label sequence given per-frame class probabilities and a
    constant log-penalty for every change of state."""
    T, K = proba.shape
    logp = np.log(np.clip(proba, 1e-9, 1))
    trans = -switch_penalty * (1 - np.eye(K))
    score = logp[0].copy()
    back = np.empty((T, K), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans  # from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    out = np.empty(T, np.int8)
    out[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def smooth(proba: np.ndarray, period: np.ndarray, switch_penalty: float = 8.0) -> np.ndarray:
    out = np.empty(len(proba), np.int8)
    for p in np.unique(period):
        s = period == p
        out[s] = viterbi(proba[s], switch_penalty)
    return out
