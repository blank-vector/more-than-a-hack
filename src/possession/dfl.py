"""Open DFL/Sportec matches as training data for the challenge.

Three steps, all cached outside the repo in $MTAH_CACHE/dfl/:

1. LOAD (load_match): kloppy -> the same arrays as the challenge ATD (ball, xy, team, all in metres,
   centre origin) plus per-frame truth: D / H / A from DFL's BallStatus and BallPossession.

2. TRANSLATE LABELS (controlled_possession): DFL flips possession on any touch (7-11 possessions per
   live minute); the challenge counts CONTROLLED possession (~4.2 per live minute in the public
   window). A team run shorter than `min_hold` frames, between stretches of the other team, is
   given back to the team that had the ball before it ("contested stays with the last team").
   `min_hold` is chosen so the DFL rate matches the challenge rate.

3. DEGRADE (degrade): make clean DFL tracking look like the challenge's automatic tracking, using
   the numbers measured by scripts/profile_atd.py and profile_gaps.py:
     * ball missing ~41% of frames, in long gaps (median ~11 s), far more often during dead ball
     * player tracks broken into fragments (median ~35 s), ~11 per team tracked, fewer in stoppages
     * small positional jitter
   Every rate is a parameter in DegradeParams, so it can be re-calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import challenge as ch
from .atd import ATD, AWAY, HOME

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
D_, H_, A_ = 0, 1, 2   # truth codes in this module (same order as the v1 decoder)


@dataclass
class Match:
    """A labelled match in challenge-like arrays. `period` replaces the challenge's halves."""
    name: str
    ball: np.ndarray      # (T, 2) metres
    xy: np.ndarray        # (T, S, 2) metres, NaN when absent
    team: np.ndarray      # (S,) 0 home, 1 away
    period: np.ndarray    # (T,) 1, 2
    truth: np.ndarray     # (T,) D_/H_/A_ as DFL labels them (touch-level)
    pitch: tuple[float, float]
    fps: float = 25.0

    def atd(self) -> ATD:
        return ATD(ball=self.ball, xy=self.xy, speed=speed_from_xy(self.xy, self.fps), team=self.team)


def cache_dir() -> Path:
    d = ch.cache_dir() / "dfl"
    d.mkdir(parents=True, exist_ok=True)
    return d


def speed_from_xy(xy: np.ndarray, fps: float, smooth: int = 5) -> np.ndarray:
    """Speed (m/s) per slot from positions, lightly smoothed. NaN where the slot is absent.

    Used for BOTH the challenge and DFL so the feature means the same thing on both sides
    (the challenge ATD's own speed column is ignored in v2).
    """
    import pandas as pd
    v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=-1) * fps
    v[0] = np.nan
    return pd.DataFrame(v).rolling(smooth, center=True, min_periods=1).median().to_numpy().astype(np.float32)


# ----------------------------------------------------------------------------- 1. load
def load_match(match_id: str, refresh: bool = False) -> Match:
    path = cache_dir() / f"{match_id}.npz"
    if path.exists() and not refresh:
        z = np.load(path)
        return Match(match_id, z["ball"], z["xy"], z["team"], z["period"], z["truth"],
                     tuple(z["pitch"]), float(z["fps"]))

    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False, coordinates="metrica")
    df = ds.to_df()
    pitch = (float(ds.metadata.pitch_dimensions.pitch_length or 105.0),
             float(ds.metadata.pitch_dimensions.pitch_width or 68.0))
    home, away = ds.metadata.teams
    cols, team = [], []
    for k, t in ((HOME, home), (AWAY, away)):
        for p in t.players:
            pid = str(p.player_id)
            if f"{pid}_x" in df.columns:
                cols.append(pid)
                team.append(k)

    def to_m(x, y):
        return np.stack([(x - 0.5) * pitch[0], (0.5 - y) * pitch[1]], axis=-1).astype(np.float32)

    ball = to_m(df["ball_x"].to_numpy(float), df["ball_y"].to_numpy(float))
    xy = to_m(df[[c + "_x" for c in cols]].to_numpy(float), df[[c + "_y" for c in cols]].to_numpy(float))
    state = df["ball_state"].astype(str).str.lower().to_numpy()
    owner = df["ball_owning_team_id"].astype(str).to_numpy()
    truth = np.where(np.char.find(state.astype(str), "dead") >= 0, D_,
                     np.where(owner == str(home.team_id), H_, A_)).astype(np.int8)
    m = Match(match_id, ball, xy, np.array(team, np.int8), df["period_id"].to_numpy(np.int8), truth,
              pitch, float(ds.metadata.frame_rate))
    np.savez_compressed(path, ball=m.ball, xy=m.xy, team=m.team, period=m.period, truth=m.truth,
                        pitch=np.array(pitch), fps=m.fps)
    return m


# ----------------------------------------------------------------------------- 2. labels
def runs(a: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end_exclusive, value) of maximal runs."""
    if len(a) == 0:
        return []
    cut = np.flatnonzero(np.diff(a) != 0) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(a)]))
    return [(int(s), int(e), int(a[s])) for s, e in zip(starts, ends)]


def controlled_possession(truth: np.ndarray, period: np.ndarray, min_hold: int) -> np.ndarray:
    """Touch-level H/A/D -> controlled possession. A team run shorter than `min_hold` frames that
    sits inside live play goes back to the team that had the ball before it. Runs right after a
    dead ball are kept (a restart always has a clear possessor)."""
    out = truth.copy()
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        seg = out[idx]
        prev_team, prev_end_dead = None, True
        for s, e, v in runs(seg):
            if v == D_:
                prev_team, prev_end_dead = None, True
                continue
            if (prev_team is not None and v != prev_team and (e - s) < min_hold and not prev_end_dead):
                seg[s:e] = prev_team
                continue
            prev_team, prev_end_dead = v, False
        out[idx] = seg
    return out


def possession_rate(labels: np.ndarray, period: np.ndarray, fps: float) -> tuple[float, float, int]:
    """(possessions per live minute, mean possession seconds, count)."""
    n, frames = 0, 0
    for p in np.unique(period):
        for s, e, v in runs(labels[period == p]):
            if v != D_:
                n += 1
                frames += e - s
    live_min = frames / fps / 60
    return n / max(live_min, 1e-9), frames / fps / max(n, 1), n


def choose_min_hold(matches: list[Match], target_rate: float = 4.2) -> tuple[int, list[tuple[int, float]]]:
    """Smallest min_hold (frames) whose pooled possession rate is at or below the target."""
    table = []
    for secs in (0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8):
        h = int(secs * 25)
        lab = [controlled_possession(m.truth, m.period, h) for m in matches]
        n = sum(possession_rate(l, m.period, m.fps)[2] for l, m in zip(lab, matches))
        live = sum(np.sum(l != D_) for l in lab) / 25 / 60
        table.append((h, n / live))
    best = min(table, key=lambda t: abs(np.log(t[1] / target_rate)))
    return best[0], table


# ----------------------------------------------------------------------------- 3. degrade
@dataclass
class DegradeParams:
    miss_dead: float = 0.64        # P(ball missing | dead ball)  — from profile_gaps: ~2/3 of missing time is dead
    miss_alive: float = 0.22       # P(ball missing | live play)
    gap_median_s: float = 10.7     # profile_atd: missing stretches median 10.7 s
    gap_sigma: float = 0.9         # log-normal spread (p90 ≈ 34 s)
    frag_median_s: float = 35.0    # profile_atd: track fragment median 35.5 s
    frag_sigma: float = 1.3
    dropout_s: float = 4.0         # mean time a player is untracked between fragments
    extra_drop_dead: float = 0.10  # extra chance a player is untracked during dead ball (19 vs 21 tracked)
    jitter_m: float = 0.25


def _lognormal_frames(rng, median_s, sigma, fps, n=1):
    return np.maximum(1, (rng.lognormal(np.log(median_s * fps), sigma, n))).astype(int)


def _ball_missing(m: Match, p: DegradeParams, rng, iters: int = 6) -> np.ndarray:
    """Missing-ball mask. Each stoppage is covered by a gap with probability `cover` (starting a
    little after it begins, sometimes running past the restart, as a tracker re-acquires late);
    live play gets random log-normal gaps at rate `hazard`. `cover` and `hazard` are adjusted a
    few times so that P(missing | dead) and P(missing | live) land on the targets."""
    T, fps = len(m.truth), m.fps
    dead = m.truth == D_
    dead_runs = [(s, e) for s, e, v in runs(dead.astype(np.int8)) if v == 1]
    cover, hazard = p.miss_dead, p.miss_alive / (p.gap_median_s * fps)
    seed = int(rng.integers(1 << 31))
    for _ in range(iters):
        r = np.random.default_rng(seed)
        miss = np.zeros(T, bool)
        for s, e in dead_runs:
            if r.random() < min(cover, 1.0):
                n = e - s
                a = s + int(r.uniform(0, 0.25) * n)
                b = min(T, a + int(n * r.uniform(0.7, 1.15)))
                miss[a:b] = True
        starts = np.flatnonzero(~dead & (r.random(T) < hazard))
        for a in starts:
            miss[a:a + int(_lognormal_frames(r, p.gap_median_s * 0.6, p.gap_sigma, fps)[0])] = True
        got_d = miss[dead].mean() if dead.any() else p.miss_dead
        got_a = miss[~dead].mean() if (~dead).any() else p.miss_alive
        cover = float(np.clip(cover * p.miss_dead / max(got_d, 1e-3), 0.05, 1.0))
        hazard = hazard * p.miss_alive / max(got_a, 1e-3)
    return miss


def degrade(m: Match, p: DegradeParams | None = None, seed: int = 0) -> ATD:
    """A noisy, challenge-like copy of a clean DFL match (labels untouched)."""
    p = p or DegradeParams()
    rng = np.random.default_rng(seed)
    T, fps = len(m.ball), m.fps
    dead = m.truth == D_

    # --- ball: missing stretches tied to stoppages (most missing ball is dead ball), plus random
    # gaps in live play; both rates are calibrated on this match to hit the targets.
    missing = _ball_missing(m, p, rng)
    ball = m.ball.copy()
    ball[missing] = np.nan
    ball += rng.normal(0, p.jitter_m / 2, ball.shape).astype(np.float32)

    # --- players: fragments separated by short dropouts, more dropouts during dead ball
    xy = m.xy.copy()
    for s in range(xy.shape[1]):
        t = 0
        while t < T:
            t += int(_lognormal_frames(rng, p.frag_median_s, p.frag_sigma, fps)[0])
            if t >= T:
                break
            gap = int(rng.exponential(p.dropout_s * fps)) + 1
            xy[t:t + gap, s] = np.nan
            t += gap
    drop = dead[:, None] & (rng.random(xy.shape[:2]) < p.extra_drop_dead)
    xy[drop] = np.nan
    xy += rng.normal(0, p.jitter_m, xy.shape).astype(np.float32)
    return ATD(ball=ball, xy=xy, speed=speed_from_xy(xy, fps), team=m.team)


def degrade_report(m: Match, a: ATD) -> dict:
    dead = m.truth == D_
    miss = np.isnan(a.ball[:, 0])
    pres = ~np.isnan(a.xy[..., 0])
    gaps = [e - s for s, e, v in runs(miss.astype(np.int8)) if v == 1]
    return dict(ball_visible=float(1 - miss.mean()), miss_dead=float(miss[dead].mean()),
                miss_alive=float(miss[~dead].mean()), gap_median_s=float(np.median(gaps) / m.fps) if gaps else 0.0,
                tracked_per_team=float(pres.sum(1).mean() / 2))
