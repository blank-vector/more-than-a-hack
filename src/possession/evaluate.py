"""Metrics + a ball-noise stress test.

The ball in Metrica's public sample is unrealistically clean (it sits on the
passer's feet and is NaN whenever play is dead). Real automatic tracking is
noisier, so always check how a method degrades with `degrade_ball`.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .labels import CLASS_NAMES, transition_mask


def score(y: np.ndarray, pred: np.ndarray, fps: float = 25.0) -> dict:
    trans = transition_mask(y, radius=int(fps))  # ±1 s around label changes
    return dict(
        acc=float((y == pred).mean()),
        macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        acc_transitions=float((y[trans] == pred[trans]).mean()),
        acc_steady=float((y[~trans] == pred[~trans]).mean()),
        switches_true=int((np.diff(y) != 0).sum()),
        switches_pred=int((np.diff(pred) != 0).sum()),
    )


def confusion(y: np.ndarray, pred: np.ndarray) -> str:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2], normalize="true")
    rows = ["true\\pred " + " ".join(f"{c:>6}" for c in CLASS_NAMES)]
    for name, r in zip(CLASS_NAMES, cm):
        rows.append(f"{name:>9} " + " ".join(f"{v:6.3f}" for v in r))
    return "\n".join(rows)


def degrade_ball(
    ball: np.ndarray,
    noise_m: float = 0.0,
    dropout: float = 0.0,
    fill_dead: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Make the sample ball track look more like raw automatic tracking.

    noise_m:   std of smooth (random-walk-ish) positional error in metres
    dropout:   fraction of in-play frames lost, in bursts of ~0.5-2 s
    fill_dead: replace NaN-during-dead-ball with the last known position
               (removes the "ball is NaN => dead ball" leak in the sample data)
    """
    rng = np.random.default_rng(seed)
    b = ball.copy()
    T = len(b)
    if noise_m > 0:
        # low-pass filtered noise so it behaves like tracking error, not jitter
        e = rng.normal(0, 1, (T, 2))
        k = np.ones(13) / 13
        e = np.stack([np.convolve(e[:, j], k, mode="same") for j in range(2)], 1)
        b += e / e.std(0) * noise_m
    if fill_dead:
        idx = np.where(~np.isnan(b[:, 0]), np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        b = b[idx]
    if dropout > 0:
        live = np.flatnonzero(~np.isnan(b[:, 0]))
        n_bursts = int(dropout * len(live) / 30)
        for s in rng.choice(live, n_bursts, replace=False):
            b[s: s + rng.integers(12, 50)] = np.nan
    return b
