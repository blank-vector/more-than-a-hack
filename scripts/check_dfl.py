"""Check that the open DFL/Sportec dataset can serve as training data.

    python scripts/check_dfl.py            # one match, ~2-5 min (downloads ~200 MB once)
    python scripts/check_dfl.py --all      # all 7 matches

This is PUBLIC data (Bassek et al., 2025, Scientific Data), not challenge data, so the output
is safe to share.

The challenge ground truth is "the team the corrected tracking says owns the ball, or D when it
says the ball is dead". DFL tracking carries exactly those two per-frame fields (BallPossession,
BallStatus). This script confirms they load, and compares possession structure with the
challenge's public 5 minutes (17 possessions, mean 14.2 s).
"""
import argparse

import numpy as np

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def spells(labels: np.ndarray) -> list[tuple[str, int]]:
    """Maximal runs, as (label, length)."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], i - start))
            start = i
    return out


def check(match_id: str) -> dict:
    from kloppy import sportec

    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
    df = ds.to_df()
    fps = ds.metadata.frame_rate
    home_id = ds.metadata.teams[0].team_id
    state = df["ball_state"].astype(str).str.lower()
    owner = df["ball_owning_team_id"].astype(str)
    lab = np.where(state.str.contains("dead"), "D", np.where(owner == str(home_id), "H", "A"))

    print(f"\n== {match_id}: {len(df)} frames @ {fps} fps, home = {ds.metadata.teams[0].name}")
    print("ball_state:", state.value_counts().to_dict())
    print("ball_owning_team_id:", df["ball_owning_team_id"].value_counts(dropna=False).to_dict())
    print(f"ball position missing on {df['ball_x'].isna().mean():.1%} of frames")

    poss, dead = [], []
    for p in df["period_id"].unique():
        for label, n in spells(lab[(df["period_id"] == p).to_numpy()]):
            (dead if label == "D" else poss).append(n / fps)
    poss, dead = np.array(poss), np.array(dead)
    mins = (lab != "D").sum() / fps / 60
    print(f"possessions: {len(poss)} ({len(poss) / mins:.1f} per alive minute), "
          f"duration s: mean {poss.mean():.1f}  median {np.median(poss):.1f}  p90 {np.percentile(poss, 90):.1f}")
    print(f"dead-ball stretches: {len(dead)}, duration s: mean {dead.mean():.1f}  median {np.median(dead):.1f}")
    print(f"label share: H {np.mean(lab == 'H'):.1%}  A {np.mean(lab == 'A'):.1%}  D {np.mean(lab == 'D'):.1%}")
    return dict(n_poss=len(poss), mean_poss=poss.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for m in MATCHES if args.all else MATCHES[:1]:
        check(m)
    print("\nChallenge public window for comparison: 17 possessions, mean 14.2 s")
