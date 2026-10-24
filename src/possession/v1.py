"""Version 1: rule-based evidence + an H/A/D Viterbi decoder. No training.

Design (from the profiling, see outputs/profile_*.txt):
  * Most missing ball is DEAD ball: players stand still, the ball was last seen near a line.
    So "ball missing + players slow + ball outside the lines" is evidence for D.
  * When the ball is visible, the team of the nearest tracked player (within a few metres) is
    evidence for that team. Tracks are long (median ~35 s), so this is reasonably stable.
  * Smart Tagging: set pieces and kick-offs mark restarts (D before, play after); each phase's
    Team tag is weak evidence about possession. Whether a code's team means "in possession" or
    "defending" is learned from agreement with the nearest-player evidence over the whole match
    (unsupervised, no ground truth used).
  * A Viterbi decoder turns per-frame evidence into whole possessions: every change of state
    pays a penalty, which is what stops flicker (spellF1 punishes flicker hard).

Every number below is a hand-set prior, documented next to it. v2 will learn them from DFL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import challenge as ch
from .atd import ATD, AWAY, HOME

D, H, A = 0, 1, 2          # decoder state order
LETTERS = np.array(["D", "H", "A"])


@dataclass
class Params:
    near_m: float = 2.0            # nearest player within this distance "has" the ball
    slow_ms: float = 1.5           # mean player speed below this suggests dead ball (profile: 78% recall, 15% FA)
    speed_window_s: float = 1.0    # smoothing of mean player speed
    # dead-ball logit = bias + weights * evidence  (positive = D more likely)
    d_bias: float = -2.0
    d_slow: float = 1.5            # players slow
    d_missing: float = 1.5         # ball not visible
    d_in_play: float = -1.5        # ball visible inside the lines (live-play evidence)
    d_ball_moving: float = -1.0    # visible ball moving faster than 2 m/s
    d_ball_still: float = 2.5      # visible ball nearly static (< 0.5 m/s median over 2 s): set-piece setup
    still_window_s: float = 2.0
    d_outside: float = 3.0         # visible ball outside the pitch lines
    d_few_players: float = 1.0     # fewer than 18 tracked (stoppages, camera moves)
    d_set_piece: float = 1.5       # inside a tagged set-piece window
    # team logit (positive = home)
    t_near: float = 2.5            # nearest player within near_m
    t_far: float = 0.3             # ball visible, nobody close: weak nearest-team hint
    t_tag: float = 0.6             # Smart Tagging phase team (sign learned per code)
    # decoder: cost (in log units) of each change of state
    switch_ha: float = 25.0        # H <-> A directly
    switch_d: float = 12.0         # into or out of D
    d_cost_ratio: float = 1.0      # extra multiplier on switch_d only (>1: stoppages harder to start/end)
    stickiness: float = 1.0        # multiplies both switch costs (scaling them together stops the
                                   # decoder dodging an expensive H->A switch through a brief D)
    field_extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- evidence
def ball_outside(ball: np.ndarray, margin: float = 0.5) -> np.ndarray:
    L, W = ch.PITCH
    with np.errstate(invalid="ignore"):
        return (np.abs(ball[:, 0]) > L / 2 + margin) | (np.abs(ball[:, 1]) > W / 2 + margin)


def nearest_by_team(a: ATD) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from the ball to the nearest tracked home / away player. inf if none."""
    d = np.linalg.norm(a.xy - a.ball[:, None, :], axis=-1)  # (T, S), NaN where missing
    d = np.where(np.isnan(d), np.inf, d)
    return d[:, a.team == HOME].min(1), d[:, a.team == AWAY].min(1)


def mean_speed(a: ATD, window: int) -> np.ndarray:
    pres = ~np.isnan(a.xy[..., 0])
    s = np.where(pres, a.speed, np.nan)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        m = np.nanmean(s, axis=1)
    return pd.Series(m).rolling(window, center=True, min_periods=1).mean().to_numpy()


def load_tags() -> list[dict]:
    import sys
    sys.path.insert(0, str(ch.OFFICIAL / "starter"))
    import load_smart_tagging  # official loader
    return load_smart_tagging.load()


def tag_team(value: str | None) -> int:
    v = (value or "").strip().lower()
    if v.startswith("barcelona"):
        return 1
    if v.startswith("atletic") or v.startswith("athletic"):
        return -1
    return 0


@dataclass
class Evidence:
    d_logit: np.ndarray
    t_logit: np.ndarray
    force_d: np.ndarray            # frames that must be D (before each half's kick-off)
    report: dict


def build_evidence(a: ATD, tags: list[dict], p: Params) -> Evidence:
    T = a.n_frames
    ball_ok = ~np.isnan(a.ball[:, 0])
    out = ball_outside(a.ball) & ball_ok
    n_tracked = (~np.isnan(a.xy[..., 0])).sum(1)
    speed = mean_speed(a, max(1, int(p.speed_window_s * ch.FPS)))
    slow = np.nan_to_num(speed, nan=0.0) < p.slow_ms

    dh, da = nearest_by_team(a)
    near = np.minimum(dh, da) <= p.near_m
    home_closer = dh < da

    # --- Smart Tagging
    set_piece = np.zeros(T, bool)
    force_d = np.zeros(T, bool)
    halves = ch.half_index(T)
    phase_team = np.zeros(T)          # +1 / -1 per frame from phase tags, later signed per code
    code_sign, code_agree = {}, {}
    nearest_vote = np.where(ball_ok & near, np.where(home_closer, 1, -1), 0)
    for code in sorted({t["code"] for t in tags}):
        sel = [t for t in tags if t["code"] == code]
        votes = []
        for t in sel:
            s, e = max(0, t["start_frame"]), min(T, t["end_frame"])
            tt = tag_team(t["tags"].get("Team"))
            if tt and e > s:
                v = nearest_vote[s:e]
                v = v[v != 0]
                if len(v):
                    votes.append(np.mean(v == tt))
        agree = float(np.mean(votes)) if votes else 0.5
        code_agree[code] = (agree, len(votes))
        # a code whose team usually matches the nearest player = "team in possession" (+1),
        # usually opposite = "defending team" (-1), near 0.5 = uninformative (0)
        code_sign[code] = 1 if agree > 0.6 else -1 if agree < 0.4 else 0

    for t in tags:
        s, e = max(0, t["start_frame"]), min(T, t["end_frame"])
        if e <= s:
            continue
        code = (t["code"] or "").upper()
        typ = (t["tags"].get("Type") or "").lower()
        if "SET PIECE" in code:
            set_piece[s:e] = True
            if typ.startswith("kick off (start)"):
                h = halves[min(e, T - 1)] or halves[s]
                if h:
                    first = np.flatnonzero(halves == h)[0]
                    force_d[first:s] = True     # graded but before kick-off: always dead
        sign = code_sign.get(t["code"], 0)
        tt = tag_team(t["tags"].get("Team"))
        if sign and tt:
            phase_team[s:e] += sign * tt
    phase_team = np.clip(phase_team, -1, 1)

    bspeed = np.nan_to_num(np.linalg.norm(np.gradient(a.ball, axis=0), axis=1) * ch.FPS, nan=0.0)
    in_play = ball_ok & ~out
    bmed = pd.Series(np.where(ball_ok, bspeed, np.nan)).rolling(
        max(1, int(p.still_window_s * ch.FPS)), center=True, min_periods=5).median().to_numpy()
    still = ball_ok & (np.nan_to_num(bmed, nan=99.0) < 0.5)
    d_logit = (p.d_bias + p.d_slow * slow + p.d_missing * ~ball_ok + p.d_outside * out
               + p.d_in_play * (in_play & ~still) + p.d_ball_moving * (in_play & (bspeed > 2.0))
               + p.d_ball_still * still
               + p.d_few_players * (n_tracked < 18) + p.d_set_piece * set_piece)
    t_logit = np.where(ball_ok & near, np.where(home_closer, p.t_near, -p.t_near),
                       np.where(ball_ok & np.isfinite(np.minimum(dh, da)),
                                np.where(home_closer, p.t_far, -p.t_far), 0.0))
    t_logit = t_logit + p.t_tag * phase_team

    report = dict(code_agree=code_agree, code_sign=code_sign,
                  frac_near=float(near[halves > 0].mean()),
                  frac_still=float(still[halves > 0].mean()))
    return Evidence(d_logit=d_logit, t_logit=t_logit, force_d=force_d, report=report)


# ----------------------------------------------------------------------------- decoder
def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0, -x)


def emissions(ev: Evidence) -> np.ndarray:
    """(T, 3) log-likelihoods for D, H, A."""
    ll = np.empty((len(ev.d_logit), 3))
    alive = log_sigmoid(-ev.d_logit)
    ll[:, D] = log_sigmoid(ev.d_logit)
    ll[:, H] = alive + log_sigmoid(ev.t_logit)
    ll[:, A] = alive + log_sigmoid(-ev.t_logit)
    ll[ev.force_d, H] = ll[ev.force_d, A] = -1e6
    return ll


def viterbi(ll: np.ndarray, p: Params) -> np.ndarray:
    T = len(ll)
    sd, sha = p.switch_d * p.stickiness * p.d_cost_ratio, p.switch_ha * p.stickiness
    trans = np.array([[0, -sd, -sd],
                      [-sd, 0, -sha],
                      [-sd, -sha, 0]], float)
    score = ll[0].copy()
    back = np.empty((T, 3), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + ll[t]
    path = np.empty(T, np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(a: ATD, tags: list[dict], p: Params | None = None) -> tuple[np.ndarray, Evidence]:
    """Letters H/A/D on graded frames, X elsewhere."""
    p = p or Params()
    ev = build_evidence(a, tags, p)
    ll = emissions(ev)
    halves = ch.half_index(a.n_frames)
    labels = np.full(a.n_frames, "X", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(halves == h)
        labels[idx] = LETTERS[viterbi(ll[idx], p)]
    return labels, ev
