"""Load Metrica-format tracking + event data into one canonical per-match structure.

Canonical match (saved as .npz in data/cache):
    period   (T,)      int   period id
    frame    (T,)      int   frame id (as used by the event file)
    ball     (T, 2)    float ball position in metres, origin at pitch centre, NaN if missing
    home     (T, P, 2) float home players in metres, NaN if not on pitch / missing
    away     (T, Q, 2) float away players in metres
    fps      scalar
Events are kept as a small DataFrame (team in {"home","away"}, type, subtype,
period, start_frame, end_frame) and saved next to it as .csv.

Only `load_tracking_df` and `load_events_df` know about file formats. When the
competition data arrives, add a loader there and everything downstream works.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


@dataclass
class Match:
    name: str
    period: np.ndarray
    frame: np.ndarray
    ball: np.ndarray
    home: np.ndarray
    away: np.ndarray
    fps: float
    events: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return len(self.frame)


# --------------------------------------------------------------------------- loaders
def _sample_paths(sample_dir: Path, game: int) -> dict:
    g = sample_dir / "data" / f"Sample_Game_{game}"
    if game in (1, 2):
        return dict(
            kind="csv",
            home=g / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv",
            away=g / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv",
            events=g / f"Sample_Game_{game}_RawEventsData.csv",
        )
    return dict(
        kind="epts",
        meta=g / f"Sample_Game_{game}_metadata.xml",
        raw=g / f"Sample_Game_{game}_tracking.txt",
        events=g / f"Sample_Game_{game}_events.json",
    )


def load_tracking_df(paths: dict) -> tuple[pd.DataFrame, float, str]:
    """Return kloppy's wide tracking DataFrame (Metrica normalised coords), fps,
    and the provider team id of the home team (for mapping events)."""
    from kloppy import metrica

    if paths["kind"] == "csv":
        ds = metrica.load_tracking_csv(
            home_data=str(paths["home"]), away_data=str(paths["away"]), coordinates="metrica"
        )
    else:
        ds = metrica.load_tracking_epts(
            meta_data=str(paths["meta"]), raw_data=str(paths["raw"]), coordinates="metrica"
        )
    home, away = ds.metadata.teams
    df = ds.to_df()
    # Normalise player columns to home_<id>_x / away_<id>_x whatever the provider calls them
    rename = {}
    for side, team in (("home", home), ("away", away)):
        for p in team.players:
            pid = str(p.player_id)
            prefix = pid if pid.startswith(side + "_") else f"{side}_{pid}"
            for suf in ("_x", "_y"):
                if pid + suf in df.columns:
                    rename[pid + suf] = prefix + suf
    df = df.rename(columns=rename)
    return df, float(ds.metadata.frame_rate), home.team_id


def load_events_df(paths: dict, home_team_id: str) -> pd.DataFrame:
    if paths["events"].suffix == ".csv":
        e = pd.read_csv(paths["events"])
        out = pd.DataFrame(
            dict(
                team=e["Team"].str.lower(),
                type=e["Type"].str.upper(),
                subtype=e["Subtype"].fillna("").str.upper(),
                period=e["Period"].astype(int),
                start_frame=e["Start Frame"].astype(int),
                end_frame=e["End Frame"].astype(int),
            )
        )
    else:
        rows = []
        for ev in json.load(open(paths["events"]))["data"]:
            rows.append(
                dict(
                    team="home" if ev["team"]["id"] == home_team_id else "away",
                    type=ev["type"]["name"].upper(),
                    subtype=_subtype(ev.get("subtypes")),
                    period=int(ev["period"]),
                    start_frame=int(ev["start"]["frame"]),
                    end_frame=int(ev["end"]["frame"] or ev["start"]["frame"]),
                )
            )
        out = pd.DataFrame(rows)
    # End frame 0 in Metrica CSV means "same as start"
    out.loc[out.end_frame <= 0, "end_frame"] = out.start_frame
    return out.sort_values(["start_frame", "end_frame"], kind="stable").reset_index(drop=True)


def _subtype(s) -> str:
    if not s:
        return ""
    if isinstance(s, dict):
        s = [s]
    return "-".join(x["name"] for x in s).upper()


def _players(df: pd.DataFrame, side: str) -> np.ndarray:
    ids = sorted({c[: -2] for c in df.columns if c.startswith(side + "_") and c.endswith("_x")})
    x = df[[i + "_x" for i in ids]].to_numpy(float)
    y = df[[i + "_y" for i in ids]].to_numpy(float)
    return _to_metres(np.stack([x, y], axis=-1))


def _to_metres(a: np.ndarray) -> np.ndarray:
    # Metrica: x,y in [0,1], (0,0) top-left. -> metres, centre origin, y up.
    out = np.empty_like(a)
    out[..., 0] = (a[..., 0] - 0.5) * PITCH_LENGTH
    out[..., 1] = (0.5 - a[..., 1]) * PITCH_WIDTH
    return out


def build_match(name: str, paths: dict) -> Match:
    df, fps, home_id = load_tracking_df(paths)
    ball = _to_metres(df[["ball_x", "ball_y"]].to_numpy(float))
    return Match(
        name=name,
        period=df["period_id"].to_numpy(int),
        frame=df["frame_id"].to_numpy(int),
        ball=ball,
        home=_players(df, "home"),
        away=_players(df, "away"),
        fps=fps,
        events=load_events_df(paths, home_id),
    )


# --------------------------------------------------------------------------- cache
def save(m: Match) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / f"{m.name}.npz",
        period=m.period, frame=m.frame, ball=m.ball, home=m.home, away=m.away, fps=m.fps,
    )
    m.events.to_csv(CACHE / f"{m.name}_events.csv", index=False)


def load(name: str) -> Match:
    z = np.load(CACHE / f"{name}.npz")
    ev = pd.read_csv(CACHE / f"{name}_events.csv", keep_default_na=False)
    return Match(name, z["period"], z["frame"], z["ball"], z["home"], z["away"], float(z["fps"]), ev)


def cached_matches() -> list[str]:
    return sorted(p.stem for p in CACHE.glob("*.npz"))


def prepare_samples(sample_dir: Path) -> list[str]:
    names = []
    for g in (1, 2, 3):
        name = f"sample_game_{g}"
        if not (CACHE / f"{name}.npz").exists():
            print(f"loading {name} ...", flush=True)
            save(build_match(name, _sample_paths(sample_dir, g)))
        names.append(name)
    return names
