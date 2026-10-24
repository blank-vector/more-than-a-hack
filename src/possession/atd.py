"""Fast reader for the challenge ATD (Metrica FIFA/EPTS export) into numpy arrays.

Why not kloppy: loading 199,319 frames x 46 slots through kloppy objects takes minutes and
gigabytes. The raw format is simple, so we stream it once and cache .npz arrays (outside the
repo). Second load takes about a second.

Raw line:   "<frame_id>:<slot>;<slot>;...;<slot>:<ball>"
each group: "x,y,speed" (players) and "x,y,..." (ball), normalised 0-1, "NaN" when missing.

Output (indexed by VIDEO frame = frame_id - 1, i.e. submission row):
    ball   (T, 2)     metres, origin at pitch centre, x along the length, NaN if missing
    xy     (T, S, 2)  metres, per track slot (S = 46), NaN if the slot is empty
    speed  (T, S)     m/s per slot, as reported by the tracker
    team   (S,)       0 = home (H), 1 = away (A), per slot
Slots are tracklets, not players: they appear, vanish and swap identity.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import challenge as ch

HOME, AWAY = 0, 1


@dataclass
class ATD:
    ball: np.ndarray
    xy: np.ndarray
    speed: np.ndarray
    team: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.ball)

    @property
    def home(self) -> np.ndarray:
        return self.xy[:, self.team == HOME]

    @property
    def away(self) -> np.ndarray:
        return self.xy[:, self.team == AWAY]


def slot_teams(meta_path: str | Path) -> np.ndarray:
    """Team per track slot. First <Team> in the XML is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _xyv(group: str) -> tuple[float, float, float]:
    parts = group.split(",")
    try:
        x, y = float(parts[0]), float(parts[1])
        v = float(parts[2]) if len(parts) > 2 else np.nan
    except (ValueError, IndexError):
        return np.nan, np.nan, np.nan
    return x, y, v


def parse(raw_path: str | Path, meta_path: str | Path, n_frames: int = ch.N_FRAMES,
          pitch: tuple[float, float] = ch.PITCH) -> ATD:
    team = slot_teams(meta_path)
    S = len(team)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    xy = np.full((n_frames, S, 2), np.nan, np.float32)
    speed = np.full((n_frames, S), np.nan, np.float32)
    L, W = pitch

    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= i < n_frames:
                continue
            for s, g in enumerate(parts[1].split(";")[:S]):
                if g and g[0] != "N":
                    xy[i, s, 0], xy[i, s, 1], speed[i, s] = _xyv(g)
            b = parts[2]
            if b and b[0] != "N":
                bx, by, _ = _xyv(b)
                ball[i] = bx, by

    # normalised (0,0 top-left) -> metres, centre origin, y up
    def to_m(a):
        out = np.empty_like(a)
        out[..., 0] = (a[..., 0] - 0.5) * L
        out[..., 1] = (0.5 - a[..., 1]) * W
        return out

    return ATD(ball=to_m(ball), xy=to_m(xy), speed=speed, team=team)


def load(source: str = "tactical", refresh: bool = False) -> ATD:
    """Parse once, then load from the cache in $MTAH_CACHE (outside this repo)."""
    cache = ch.cache_dir() / f"{ch.GAME}_{source}_atd.npz"
    if cache.exists() and not refresh:
        z = np.load(cache)
        return ATD(z["ball"], z["xy"], z["speed"], z["team"])
    meta, raw = ch.games.require(["atd_meta", "atd_raw"], ch.GAME, None, source)
    n = ch.games.source_frames(source, ch.GAME)
    print(f"parsing {Path(raw).name} (one-off, ~1-2 min) ...", flush=True)
    a = parse(raw, meta, n_frames=n)
    np.savez_compressed(cache, ball=a.ball, xy=a.xy, speed=a.speed, team=a.team)
    return a
