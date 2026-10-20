"""Rule-based baseline: nearest player to the ball, with hysteresis.

Per frame:
  * ball missing or off the pitch            -> NONE
  * closest player within `radius` metres    -> candidate = that player's team
  * nobody close (ball in flight / loose)    -> keep current owner
The owner only switches after the candidate has held for `min_frames`
consecutive frames, which removes flicker during challenges.
"""
from __future__ import annotations

import numpy as np

from .data import PITCH_LENGTH, PITCH_WIDTH, Match
from .labels import AWAY, HOME, NONE


def team_distances(m: Match, ball: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distances of each team's players to the ball: (T, P), (T, Q). NaN-padded."""
    b = m.ball if ball is None else ball
    dh = np.linalg.norm(m.home - b[:, None, :], axis=-1)
    da = np.linalg.norm(m.away - b[:, None, :], axis=-1)
    dh.sort(axis=1)  # NaN sort to the end
    da.sort(axis=1)
    return dh, da


def ball_dead(ball: np.ndarray, margin: float = 0.5) -> np.ndarray:
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = (np.abs(x) > PITCH_LENGTH / 2 + margin) | (np.abs(y) > PITCH_WIDTH / 2 + margin)
    return np.isnan(x) | out


def raw_calls(m: Match, radius: float = 1.5, ball: np.ndarray | None = None) -> np.ndarray:
    """-1 = no call (keep previous), else NONE/HOME/AWAY."""
    b = m.ball if ball is None else ball
    dh, da = team_distances(m, b)
    h, a = np.nan_to_num(dh[:, 0], nan=np.inf), np.nan_to_num(da[:, 0], nan=np.inf)
    calls = np.full(len(b), -1, np.int8)
    near = np.minimum(h, a) <= radius
    calls[near & (h <= a)] = HOME
    calls[near & (a < h)] = AWAY
    calls[ball_dead(b)] = NONE
    return calls


def hysteresis(calls: np.ndarray, min_frames: int = 3, start: int = NONE) -> np.ndarray:
    out = np.empty(len(calls), np.int8)
    owner, cand, run = start, start, 0
    for i, c in enumerate(calls):
        if c >= 0 and c != owner:
            run = run + 1 if c == cand else 1
            cand = c
            # going dead is immediate; switching teams needs confirmation
            if c == NONE or run >= min_frames:
                owner, run = c, 0
        elif c == owner:
            run = 0
        out[i] = owner
    return out


def predict(m: Match, radius: float = 1.5, min_frames: int = 3, ball: np.ndarray | None = None) -> np.ndarray:
    pred = np.empty(m.n_frames, np.int8)
    calls = raw_calls(m, radius, ball)
    for p in np.unique(m.period):  # reset at each period
        s = m.period == p
        pred[s] = hysteresis(calls[s], min_frames)
    return pred
