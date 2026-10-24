"""Bridge to the official challenge repo (metrica-sports/more-than-a-hack-2027).

We import its games.py (paths, frame counts, graded halves) and its evaluation code (the scorer
that decides the competition) instead of copying them, so we always score exactly like the
organisers do.

Where things live (override with env vars):
    MTAH_REPO      official repo        default: ../more-than-a-hack-2027 next to this repo
    MTAH_DATA_DIR  match data bundle    read by games.py itself
    MTAH_CACHE     our parsed arrays    default: $MTAH_DATA_DIR/cache  (outside this repo)

Nothing in this repo ever stores match data. That is a rule of the competition (no
redistribution), and it keeps the data away from any AI tool working in this repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_REPO = Path(__file__).resolve().parents[2]
OFFICIAL = Path(os.environ.get("MTAH_REPO", THIS_REPO.parent / "more-than-a-hack-2027")).resolve()

if not (OFFICIAL / "games.py").exists():
    raise ImportError(
        f"Official challenge repo not found at {OFFICIAL}.\n"
        "Clone it next to this repo, or set MTAH_REPO:\n"
        "  git clone https://github.com/metrica-sports/more-than-a-hack-2027"
    )
for p in (OFFICIAL, OFFICIAL / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import games  # noqa: E402  (official)
import possession_metrics as pm  # noqa: E402  (official)

GAME = games.GAME
N_FRAMES = games.GAMES[GAME]["n_video_frames"]
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]  # (length, width) in metres
PUBLIC_GT = OFFICIAL / "evaluation" / "gt" / f"{GAME}_possession_gt_public.csv"


def cache_dir() -> Path:
    d = Path(os.environ.get("MTAH_CACHE", Path(games.DATA_DIR) / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def graded_mask(n: int = N_FRAMES):
    import numpy as np

    m = np.zeros(n, bool)
    for _, a, b in games.half_video_blocks(GAME):
        m[a: b + 1] = True
    return m


def half_index(n: int = N_FRAMES):
    """0 outside the graded halves, else 1 or 2."""
    import numpy as np

    h = np.zeros(n, np.int8)
    for half, a, b in games.half_video_blocks(GAME):
        h[a: b + 1] = half
    return h


def public_gt():
    """Public 5-minute ground truth as label codes (official H=0, A=1, D=2, X=3)."""
    return pm.label_codes(pm.read_label_csv(str(PUBLIC_GT)))
