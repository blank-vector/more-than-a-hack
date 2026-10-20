"""Per-frame features for the possession classifier.

Uses the whole match (past and future frames) because predictions are made
offline on a complete match. If the competition turns out to require causal
predictions, drop the `next_*` and centred-window features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import ball_dead, team_distances
from .data import Match

CLOSE_M = 1.5
CAP = 250  # frames (10 s) cap for "time since / until" features


def _since(mask: np.ndarray, cap: int = CAP) -> np.ndarray:
    """Frames since mask was last True (cap if never)."""
    T = len(mask)
    last = np.where(mask, np.arange(T), -10**9)
    np.maximum.accumulate(last, out=last)
    return np.minimum(np.arange(T) - last, cap).astype(float)


def _until(mask: np.ndarray, cap: int = CAP) -> np.ndarray:
    return _since(mask[::-1], cap)[::-1]


def _rolling(x: np.ndarray, w: int, fn: str = "mean") -> np.ndarray:
    return getattr(pd.Series(x).rolling(w, center=True, min_periods=1), fn)().to_numpy()


def _per_period(m: Match, fn, *arrays):
    out = np.empty(m.n_frames)
    for p in np.unique(m.period):
        s = m.period == p
        out[s] = fn(*(a[s] for a in arrays))
    return out


def build(m: Match, ball: np.ndarray | None = None) -> pd.DataFrame:
    b = m.ball if ball is None else ball
    dh, da = team_distances(m, b)
    h1 = np.nan_to_num(dh[:, 0], nan=60.0).clip(max=60)
    a1 = np.nan_to_num(da[:, 0], nan=60.0).clip(max=60)
    h2 = np.nan_to_num(dh[:, 1], nan=60.0).clip(max=60)
    a2 = np.nan_to_num(da[:, 1], nan=60.0).clip(max=60)
    diff = h1 - a1
    dead = ball_dead(b)
    missing = np.isnan(b[:, 0])

    # ball kinematics from a lightly smoothed, gap-filled track
    bs = pd.DataFrame(b).interpolate(limit=10, limit_area="inside").rolling(5, center=True, min_periods=1).mean().to_numpy()
    v = np.gradient(bs, axis=0) * m.fps
    speed = np.nan_to_num(np.linalg.norm(v, axis=1), nan=0.0).clip(max=40)
    accel = np.nan_to_num(np.abs(np.gradient(speed)) * m.fps, nan=0.0).clip(max=200)

    hc = (h1 <= CLOSE_M) & (h1 <= a1) & ~dead
    ac = (a1 <= CLOSE_M) & (a1 < h1) & ~dead

    f = {
        "home_d1": h1, "away_d1": a1, "home_d2": h2, "away_d2": a2, "d1_diff": diff,
        "ball_x": np.nan_to_num(b[:, 0]), "ball_y": np.nan_to_num(b[:, 1]),
        "ball_missing": missing.astype(float), "ball_dead": dead.astype(float),
        "speed": speed, "accel": accel,
        "since_dead": _per_period(m, _since, dead), "until_dead": _per_period(m, _until, dead),
        "since_home_close": _per_period(m, _since, hc), "since_away_close": _per_period(m, _since, ac),
        "until_home_close": _per_period(m, _until, hc), "until_away_close": _per_period(m, _until, ac),
    }
    for r in (2.0, 5.0, 10.0):
        f[f"home_n{int(r)}"] = (dh <= r).sum(1).astype(float)
        f[f"away_n{int(r)}"] = (da <= r).sum(1).astype(float)
    f["since_close_diff"] = f["since_home_close"] - f["since_away_close"]
    f["until_close_diff"] = f["until_home_close"] - f["until_away_close"]
    for w in (5, 25, 75):
        f[f"diff_mean_{w}"] = _rolling(diff, w)
        f[f"diff_min_{w}"] = _rolling(diff, w, "min")
        f[f"diff_max_{w}"] = _rolling(diff, w, "max")
        f[f"dead_mean_{w}"] = _rolling(dead.astype(float), w)
    return pd.DataFrame(f)
