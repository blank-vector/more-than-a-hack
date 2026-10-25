"""Version 2: features shared by DFL and the challenge, two LightGBM models, a tuned decoder.

    features(a, period, fps)  -> per-frame DataFrame, computed identically for both datasets
    fit(...)                  -> dead-vs-alive model + home-vs-away model (alive frames only)
    proba(...)                -> (T, 3) probabilities for D, H, A
    decode(...)               -> labels, with switch costs tuned by official spellF1 on DFL

Design notes
  * Only signals both datasets share: positions, the ball when visible, speeds computed from
    positions. Smart Tagging exists only for the challenge, so it stays out of the model.
  * Direction-free: each team's attacking direction per period is inferred from where it
    stands on average (a team spends more time in its own half), so "pushing forward" means
    the same thing in every half of every match.
  * Context matters more than the instant: rolling windows (past AND future, since the whole
    match is labelled offline) carry "who was near the ball a second ago / a second later" and
    "how long since the ball was last seen".
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .atd import ATD, AWAY, HOME
from .dfl import speed_from_xy

D, H, A = 0, 1, 2
LETTERS = np.array(["D", "H", "A"])
NEAR_M = 2.0
CAP = 250  # 10 s cap on time-since / time-until features


# ----------------------------------------------------------------------------- helpers
def _roll(x, w, fn="mean"):
    return getattr(pd.Series(x).rolling(w, center=True, min_periods=1), fn)().to_numpy()


def _since(mask: np.ndarray, cap: int = CAP) -> np.ndarray:
    T = len(mask)
    last = np.where(mask, np.arange(T), -10 ** 9)
    np.maximum.accumulate(last, out=last)
    return np.minimum(np.arange(T) - last, cap).astype(np.float32)


def _until(mask: np.ndarray, cap: int = CAP) -> np.ndarray:
    return _since(mask[::-1], cap)[::-1]


def _per_period(period, fn, *arrays):
    out = np.empty(len(period), np.float32)
    for p in np.unique(period):
        s = period == p
        out[s] = fn(*(a[s] for a in arrays))
    return out


def _nanmean_rows(a):
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        return np.nanmean(a, axis=1)


# ----------------------------------------------------------------------------- features
def features(a: ATD, period: np.ndarray, pitch: tuple[float, float], fps: float = 25.0) -> pd.DataFrame:
    T = a.n_frames
    L, W = pitch
    speed = speed_from_xy(a.xy, fps)
    pres = ~np.isnan(a.xy[..., 0])
    hm, am = a.team == HOME, a.team == AWAY

    ball = a.ball
    ball_ok = ~np.isnan(ball[:, 0])
    bx, by = np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1])
    with np.errstate(invalid="ignore"):
        out = ball_ok & ((np.abs(ball[:, 0]) > L / 2 + 0.5) | (np.abs(ball[:, 1]) > W / 2 + 0.5))
    bspeed = np.nan_to_num(np.linalg.norm(np.gradient(ball, axis=0), axis=1) * fps, nan=0.0).clip(0, 40)
    bmed = pd.Series(np.where(ball_ok, bspeed, np.nan)).rolling(50, center=True, min_periods=5).median().to_numpy()

    d = np.linalg.norm(a.xy - ball[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.sort(d[:, hm], axis=1)
    da = np.sort(d[:, am], axis=1)
    h1, a1 = np.minimum(dh[:, 0], 60), np.minimum(da[:, 0], 60)
    h2, a2 = np.minimum(dh[:, 1], 60), np.minimum(da[:, 1], 60)
    near_h = ball_ok & (h1 <= NEAR_M) & (h1 <= a1)
    near_a = ball_ok & (a1 <= NEAR_M) & (a1 < h1)

    # attacking direction per team and period: a team defends the half it stands in on average
    cx_h = _nanmean_rows(np.where(pres[:, hm], a.xy[:, hm, 0], np.nan))
    cx_a = _nanmean_rows(np.where(pres[:, am], a.xy[:, am, 0], np.nan))
    dir_h = np.zeros(T, np.float32)
    for p in np.unique(period):
        s = period == p
        dir_h[s] = -np.sign(np.nanmean(cx_h[s]) - np.nanmean(cx_a[s])) or 1.0
    dir_a = -dir_h

    def team_block(mask, direction, tag):
        xs = np.where(pres[:, mask], a.xy[:, mask, 0], np.nan) * direction[:, None]
        ys = np.where(pres[:, mask], a.xy[:, mask, 1], np.nan)
        sp = np.where(pres[:, mask], speed[:, mask], np.nan)
        cx = _nanmean_rows(xs)
        f = {
            f"{tag}_n": pres[:, mask].sum(1).astype(np.float32),
            f"{tag}_cx": np.nan_to_num(cx),
            f"{tag}_front": np.nan_to_num(np.nanmax(np.where(np.isnan(xs), -99, xs), axis=1)),
            f"{tag}_spread_x": np.nan_to_num(np.nanstd(xs, axis=1)),
            f"{tag}_spread_y": np.nan_to_num(np.nanstd(ys, axis=1)),
            f"{tag}_speed": np.nan_to_num(_nanmean_rows(sp)),
            f"{tag}_vx": np.nan_to_num(np.gradient(_roll(np.nan_to_num(cx), 25)) * fps),
        }
        return f

    f = {
        "ball_ok": ball_ok.astype(np.float32),
        "ball_out": out.astype(np.float32),
        "ball_speed": bspeed.astype(np.float32),
        "ball_still": (ball_ok & (np.nan_to_num(bmed, nan=99) < 0.5)).astype(np.float32),
        "ball_abs_x": np.abs(bx), "ball_abs_y": np.abs(by),
        "ball_to_touch": np.where(ball_ok, W / 2 - np.abs(by), -1),
        "ball_to_goalline": np.where(ball_ok, L / 2 - np.abs(bx), -1),
        "ball_x_home_dir": np.where(ball_ok, bx * dir_h, 0),
        "h1": h1, "a1": a1, "h2": h2, "a2": a2, "d1_diff": h1 - a1,
        "h_n5": (dh <= 5).sum(1).astype(np.float32), "a_n5": (da <= 5).sum(1).astype(np.float32),
        "n_tracked": pres.sum(1).astype(np.float32),
        "mean_speed": np.nan_to_num(_nanmean_rows(np.where(pres, speed, np.nan))),
        "since_ball": _per_period(period, _since, ball_ok),
        "until_ball": _per_period(period, _until, ball_ok),
        "since_out": _per_period(period, _since, out),
        "until_out": _per_period(period, _until, out),
        "since_near_h": _per_period(period, _since, near_h),
        "since_near_a": _per_period(period, _since, near_a),
        "until_near_h": _per_period(period, _until, near_h),
        "until_near_a": _per_period(period, _until, near_a),
    }
    f.update(team_block(hm, dir_h, "home"))
    f.update(team_block(am, dir_a, "away"))
    f["cx_diff"] = f["home_cx"] - f["away_cx"]
    f["speed_diff"] = f["home_speed"] - f["away_speed"]
    for w in (25, 75, 250):
        f[f"mean_speed_{w}"] = _roll(f["mean_speed"], w)
        f[f"ball_ok_{w}"] = _roll(f["ball_ok"], w)
        f[f"near_h_{w}"] = _roll(near_h.astype(np.float32), w)
        f[f"near_a_{w}"] = _roll(near_a.astype(np.float32), w)
        f[f"d1_diff_{w}"] = _roll(f["d1_diff"], w)
        f[f"cx_diff_{w}"] = _roll(f["cx_diff"], w)
    f["since_near_diff"] = f["since_near_h"] - f["since_near_a"]
    f["until_near_diff"] = f["until_near_h"] - f["until_near_a"]
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


# ----------------------------------------------------------------------------- models
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0, num_threads=0, deterministic=True, force_row_wise=True)


@dataclass
class Models:
    dead: lgb.Booster
    team: lgb.Booster


def fit(X: pd.DataFrame, y: np.ndarray, rounds: int = 300, step: int = 5) -> Models:
    """y in {D, H, A}. Every `step`-th frame only: neighbouring frames are near-duplicates."""
    idx = np.arange(0, len(y), step)
    Xs, ys = X.iloc[idx], y[idx]
    dead = lgb.train(PARAMS, lgb.Dataset(Xs, (ys == D).astype(int)), rounds)
    alive = ys != D
    team = lgb.train(PARAMS, lgb.Dataset(Xs[alive], (ys[alive] == H).astype(int)), rounds)
    return Models(dead, team)


def save_models(m: Models, folder) -> None:
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    m.dead.save_model(str(folder / "dead.txt"))
    m.team.save_model(str(folder / "team.txt"))


def load_models(folder) -> Models:
    from pathlib import Path
    folder = Path(folder)
    return Models(lgb.Booster(model_file=str(folder / "dead.txt")), lgb.Booster(model_file=str(folder / "team.txt")))


def proba(m: Models, X: pd.DataFrame) -> np.ndarray:
    pd_ = m.dead.predict(X)
    ph = m.team.predict(X)
    return np.stack([pd_, (1 - pd_) * ph, (1 - pd_) * (1 - ph)], axis=1)


# ----------------------------------------------------------------------------- decoder
@dataclass
class Decoder:
    switch_ha: float = 30.0   # log-cost of a direct H <-> A change
    switch_d: float = 40.0    # log-cost of entering or leaving D
    temp: float = 1.0         # divides log-probabilities (frames are correlated, so <1 sharpens less)
    shift: int = 0            # frames to move every predicted change EARLIER (corrects a systematic lag)


def viterbi(logp: np.ndarray, dec: Decoder) -> np.ndarray:
    T = len(logp)
    trans = -np.array([[0, dec.switch_d, dec.switch_d],
                       [dec.switch_d, 0, dec.switch_ha],
                       [dec.switch_d, dec.switch_ha, 0]], float)
    score = logp[0].copy()
    back = np.empty((T, 3), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(T, np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def shift_earlier(labels: np.ndarray, period: np.ndarray, k: int) -> np.ndarray:
    """Move every change of label k frames earlier, within each period."""
    if k <= 0:
        return labels
    out = labels.copy()
    for q in np.unique(period):
        s = np.flatnonzero(period == q)
        seg = labels[s]
        out[s] = np.concatenate((seg[k:], np.repeat(seg[-1:], min(k, len(seg)))))[: len(seg)]
    return out


def decode(p: np.ndarray, period: np.ndarray, dec: Decoder) -> np.ndarray:
    logp = np.log(np.clip(p, 1e-6, 1)) / dec.temp
    out = np.empty(len(p), np.int8)
    for q in np.unique(period):
        s = np.flatnonzero(period == q)
        out[s] = viterbi(logp[s], dec)
    return shift_earlier(out, period, dec.shift)


def change_lags(truth: np.ndarray, pred: np.ndarray, period: np.ndarray, window: int = 50) -> np.ndarray:
    """For each true change of label, signed lag (frames) to the nearest predicted change of the
    same kind (same new label) within ±window. Positive = we switch late."""
    lags = []
    for q in np.unique(period):
        s = np.flatnonzero(period == q)
        t, p = truth[s], pred[s]
        tc = np.flatnonzero(np.diff(t) != 0) + 1
        pc = np.flatnonzero(np.diff(p) != 0) + 1
        for c in tc:
            cand = pc[(np.abs(pc - c) <= window) & (p[np.minimum(pc, len(p) - 1)] == t[c])]
            if len(cand):
                lags.append(cand[np.argmin(np.abs(cand - c))] - c)
    return np.array(lags)


# ----------------------------------------------------------------------------- scoring
def to_official(codes: np.ndarray, period: np.ndarray) -> np.ndarray:
    """D/H/A (this module) -> official H=0, A=1, D=2, with X on each period's first frame so the
    official scorer treats periods as separate halves."""
    off = np.array([2, 0, 1], np.int8)[codes]
    first = np.flatnonzero(np.diff(period, prepend=-1) != 0)
    off[first] = 3
    return off


# ----------------------------------------------------------------------------- Smart Tagging prior
@dataclass
class TagPrior:
    """Challenge-only nudges from Smart Tagging (DFL has no tags, so these can't be learned there).

    A SET PIECES tag marks a ~5 s window around a restart: the ball is dead just before the
    window and in play just after it, owned by the restarting team when the tag names one.
    After a GOALS tag the ball is dead until the following kick-off.
    """
    dead_before_s: float = 6.0     # how far before the window start the ball is assumed dead
    dead_boost: float = 2.0        # log-odds added to D in that stretch (and the first half of the window)
    alive_after_s: float = 3.0     # stretch after the window where play is assumed live
    alive_boost: float = 1.5       # log-odds added against D there
    team_boost: float = 1.0        # log-odds for the restarting team there (if the tag names it)
    goal_max_gap_s: float = 180.0  # longest goal → kick-off gap treated as dead


def _team_of(tag: dict) -> int | None:
    v = (tag["tags"].get("Team") or "").strip().lower()
    return H if v.startswith("barcelona") else A if (v.startswith("atletic") or v.startswith("athletic")) else None


def apply_tag_prior(p: np.ndarray, tags: list[dict], prior: TagPrior, fps: float = 25.0) -> tuple[np.ndarray, dict]:
    """Return re-weighted probabilities and a small report of what was applied."""
    T = len(p)
    logit = np.log(np.clip(p, 1e-6, 1))
    n_sp = n_goal = 0
    for t in tags:
        code = (t["code"] or "").upper()
        s, e = max(0, t["start_frame"]), min(T, t["end_frame"])
        if e <= s:
            continue
        if "SET PIECE" in code:
            n_sp += 1
            mid = (s + e) // 2
            a = max(0, s - int(prior.dead_before_s * fps))
            logit[a:mid, D] += prior.dead_boost
            b = min(T, e + int(prior.alive_after_s * fps))
            logit[e:b, D] -= prior.alive_boost
            team = _team_of(t)
            if team is not None:
                logit[e:b, team] += prior.team_boost
        elif code.startswith("GOAL"):
            nxt = [u for u in tags if "SET PIECE" in (u["code"] or "").upper()
                   and "after goal" in (u["tags"].get("Type") or "").lower() and u["start_frame"] >= e]
            if nxt:
                k = min(u["start_frame"] for u in nxt)
                if k - e <= prior.goal_max_gap_s * fps:
                    logit[e:k, D] += 10.0
                    n_goal += 1
    q = np.exp(logit - logit.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True), dict(set_pieces=n_sp, goals=n_goal)


def tag_coverage(labels: np.ndarray, tags: list[dict], fps: float = 25.0) -> float:
    """Share of set-piece tags whose 6 s lead-in contains some predicted D (higher = consistent)."""
    hits = n = 0
    for t in tags:
        if "SET PIECE" not in (t["code"] or "").upper():
            continue
        s = t["start_frame"]
        seg = labels[max(0, s - int(6 * fps)): s + 1]
        if len(seg):
            n += 1
            hits += int(np.any(seg == D))
    return hits / max(n, 1)
