"""Frame-level possession labels derived from Metrica event data.

Classes: 0 = no possession (dead ball / before kick-off), 1 = home, 2 = away.

This is a *proxy* for the competition's ground truth (whose exact definition we
won't know until 20 Oct). Rules:
  * an on-ball event (pass, carry, recovery, ball lost, shot, set piece) gives
    possession to its team from its start frame;
  * ball out, a foul, or a shot that ends out / in goal makes the ball dead
    until the next set piece;
  * possession persists between events (passes in flight stay with the passer's
    team until someone else touches the ball).
Challenges and cards don't change possession.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NONE, HOME, AWAY = 0, 1, 2
CLASS_NAMES = ["none", "home", "away"]

ON_BALL = {"PASS", "CARRY", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE"}
DEAD_AFTER_START = {"BALL OUT", "FAULT RECEIVED"}


def event_changes(events: pd.DataFrame) -> list[tuple[int, int]]:
    """(frame, label) change points, in frame order."""
    changes = []
    for e in events.itertuples(index=False):
        team = HOME if e.team == "home" else AWAY
        if e.type in ON_BALL:
            changes.append((e.start_frame, team))
            if e.type == "SHOT" and any(k in e.subtype for k in ("OUT", "GOAL")) and "BLOCKED" not in e.subtype:
                changes.append((max(e.end_frame, e.start_frame + 1), NONE))
        elif e.type in DEAD_AFTER_START:
            changes.append((e.start_frame, NONE))
    # stable sort keeps file order for same-frame events
    return sorted(changes, key=lambda c: c[0])


def frame_labels(frames: np.ndarray, periods: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(frames), dtype=np.int8)
    changes = event_changes(events)
    if not changes:
        return y
    cf = np.array([c[0] for c in changes])
    cl = np.array([c[1] for c in changes], dtype=np.int8)
    # index of last change at or before each frame
    idx = np.searchsorted(cf, frames, side="right") - 1
    valid = idx >= 0
    y[valid] = cl[idx[valid]]

    # dead ball at period boundaries: from the last event of a period to its end,
    # and before the first event of each period
    ev_period_first = events.groupby("period").start_frame.min()
    ev_period_last = events.groupby("period").end_frame.max()
    for p in np.unique(periods):
        in_p = periods == p
        if p in ev_period_first.index:
            y[in_p & (frames < ev_period_first[p])] = NONE
            # ball keeps being played a few seconds after the last event; leave 2s grace
            y[in_p & (frames > ev_period_last[p] + 50)] = NONE
        else:
            y[in_p] = NONE
    return y


def transition_mask(y: np.ndarray, radius: int) -> np.ndarray:
    """True for frames within `radius` frames of a label change."""
    change = np.flatnonzero(np.diff(y) != 0) + 1
    m = np.zeros(len(y), bool)
    for c in change:
        m[max(0, c - radius): c + radius] = True
    return m
