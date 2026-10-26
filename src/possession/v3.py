"""Version 3: v2 + short-range features + a CHANGE detector that sets the decoder's switch costs.

Why: v2's decoder charges the same cost for a team switch everywhere. High enough to stop
flicker, it also blocks real 1-3 s possessions (v2 finds ~85% of DFL possessions). v3 trains a
third model, P(a change of label happens within ±0.25 s), and makes switching cheap where it
fires and expensive elsewhere:

    cost_t = max(0, base_cost - k * logit(p_change_t))

k = 0 gives back v2's constant costs, so the CV sweep includes v2-style decoding.
v2 (and the frozen v2d) are untouched: v3 has its own features, models and decoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import v2
from .atd import ATD, AWAY, HOME

D, H, A = v2.D, v2.H, v2.A
LETTERS = v2.LETTERS
CHANGE_WIN = 6  # frames either side of a true change that count as "change here" (±0.24 s)


# ----------------------------------------------------------------------------- features
def features(a: ATD, period: np.ndarray, pitch, fps: float = 25.0) -> pd.DataFrame:
    X = v2.features(a, period, pitch, fps)
    ball = a.ball
    ok = ~np.isnan(ball[:, 0])
    bs = pd.DataFrame(ball).interpolate(limit=5, limit_area="inside").rolling(3, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(bs, axis=0) * fps
    # direction of play for home, per period (same rule as v2: a team stands in its own half)
    pres = ~np.isnan(a.xy[..., 0])
    hm, am = a.team == HOME, a.team == AWAY
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        cxh = np.nanmean(np.where(pres[:, hm], a.xy[:, hm, 0], np.nan), axis=1)
        cxa = np.nanmean(np.where(pres[:, am], a.xy[:, am, 0], np.nan), axis=1)
    dir_h = np.ones(len(ball), np.float32)
    for q in np.unique(period):
        s = period == q
        d = -np.sign(np.nanmean(cxh[s]) - np.nanmean(cxa[s]))
        dir_h[s] = d if d != 0 else 1.0
    f = {}
    f["ball_vx_home_dir"] = np.nan_to_num(v[:, 0] * dir_h).clip(-40, 40)
    f["ball_vy"] = np.nan_to_num(np.abs(v[:, 1])).clip(0, 40)
    f["ball_turn"] = np.nan_to_num(np.abs(np.gradient(np.arctan2(v[:, 1], v[:, 0])))).clip(0, 3.2)

    h1, a1 = X["h1"].to_numpy(), X["a1"].to_numpy()
    f["h1_rate"] = np.gradient(pd.Series(h1).rolling(5, center=True, min_periods=1).mean().to_numpy()) * fps
    f["a1_rate"] = np.gradient(pd.Series(a1).rolling(5, center=True, min_periods=1).mean().to_numpy()) * fps
    tight_h = (ok & (h1 <= 1.0) & (h1 <= a1)).astype(np.float32)
    tight_a = (ok & (a1 <= 1.0) & (a1 < h1)).astype(np.float32)
    for w in (5, 12):
        f[f"tight_h_{w}"] = v2._roll(tight_h, w)
        f[f"tight_a_{w}"] = v2._roll(tight_a, w)
        f[f"d1_diff_{w}"] = v2._roll(X["d1_diff"].to_numpy(), w)
        f[f"ball_ok_{w}"] = v2._roll(X["ball_ok"].to_numpy(), w)
    # past vs future contrast: a change shows up as "home before, away after" (or the reverse)
    for w in (12, 25, 50):
        past = pd.Series(X["d1_diff"].to_numpy()).rolling(w, min_periods=1).mean().to_numpy()
        fut = pd.Series(X["d1_diff"].to_numpy()[::-1]).rolling(w, min_periods=1).mean().to_numpy()[::-1]
        f[f"d1_contrast_{w}"] = fut - past
        sp = X["mean_speed"].to_numpy()
        f[f"speed_contrast_{w}"] = (pd.Series(sp[::-1]).rolling(w, min_periods=1).mean().to_numpy()[::-1]
                                    - pd.Series(sp).rolling(w, min_periods=1).mean().to_numpy())
    for k, val in f.items():
        X[k] = np.asarray(val, np.float32)
    return X


# ----------------------------------------------------------------------------- models
@dataclass
class Models:
    dead: lgb.Booster
    team: lgb.Booster
    change: lgb.Booster


def change_target(y: np.ndarray, period: np.ndarray, win: int = CHANGE_WIN) -> np.ndarray:
    """1 within ±win frames of any change of label (inside a period), else 0."""
    c = np.zeros(len(y), np.int8)
    for q in np.unique(period):
        idx = np.flatnonzero(period == q)
        ch = np.flatnonzero(np.diff(y[idx]) != 0) + 1
        for i in ch:
            c[idx[max(0, i - win)]: idx[min(len(idx) - 1, i + win)] + 1] = 1
    return c


def fit(X: pd.DataFrame, y: np.ndarray, period: np.ndarray, rounds: int = 300, step: int = 5) -> Models:
    base = v2.fit(X, y, rounds=rounds, step=step)
    c = change_target(y, period)
    idx = np.arange(0, len(y), max(1, step // 2))  # changes are rare: sample twice as densely
    change = lgb.train(v2.PARAMS, lgb.Dataset(X.iloc[idx], c[idx]), rounds)
    return Models(base.dead, base.team, change)


def proba(m: Models, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = v2.proba(v2.Models(m.dead, m.team), X)
    return p, m.change.predict(X)


def fit_dead_v2(X: pd.DataFrame, y: np.ndarray, v2_columns: list[str], rounds: int = 300, step: int = 5) -> lgb.Booster:
    """Dead-vs-alive model on v2's 62 features only (v3's extra short-range features hurt it on
    the real match)."""
    idx = np.arange(0, len(y), step)
    return lgb.train(v2.PARAMS, lgb.Dataset(X.iloc[idx][v2_columns], (y[idx] == D).astype(int)), rounds)


def proba_hybrid(m: Models, dead: lgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """P(D) from a v2-feature dead model, P(home | alive) and P(change) from the v3 models."""
    pd_ = dead.predict(X[dead.feature_name()])
    ph = m.team.predict(X)
    p = np.stack([pd_, (1 - pd_) * ph, (1 - pd_) * (1 - ph)], axis=1)
    return p, m.change.predict(X)


def save_models(m: Models, folder) -> None:
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("dead", "team", "change"):
        getattr(m, name).save_model(str(folder / f"{name}.txt"))


def load_models(folder) -> Models:
    from pathlib import Path
    folder = Path(folder)
    return Models(*(lgb.Booster(model_file=str(folder / f"{n}.txt")) for n in ("dead", "team", "change")))


# ----------------------------------------------------------------------------- decoder
@dataclass
class Decoder:
    switch_ha: float = 20.0
    switch_d: float = 40.0
    k: float = 0.0          # how strongly the change detector lowers (or raises) switch costs


def viterbi(logp: np.ndarray, cost_ha: np.ndarray, cost_d: np.ndarray) -> np.ndarray:
    T = len(logp)
    trans = np.zeros((T, 3, 3))          # trans[t, i, j] = -cost of moving from i to j at frame t
    trans[:, D, H] = trans[:, D, A] = trans[:, H, D] = trans[:, A, D] = -cost_d
    trans[:, H, A] = trans[:, A, H] = -cost_ha
    score = logp[0].copy()
    back = np.empty((T, 3), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans[t]
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(p: np.ndarray, pc: np.ndarray, period: np.ndarray, dec: Decoder) -> np.ndarray:
    logp = np.log(np.clip(p, 1e-6, 1))
    lc = np.log(np.clip(pc, 1e-4, 1 - 1e-4) / np.clip(1 - pc, 1e-4, 1))
    cost_ha = np.maximum(0.0, dec.switch_ha - dec.k * lc)
    cost_d = np.maximum(0.0, dec.switch_d - dec.k * lc)
    out = np.empty(len(p), np.int8)
    for q in np.unique(period):
        s = np.flatnonzero(period == q)
        out[s] = viterbi(logp[s], cost_ha[s], cost_d[s])
    return out
