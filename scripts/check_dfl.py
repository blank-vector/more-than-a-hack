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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession.report import bar, console, grade, insight, save, section, table, title  # noqa: E402

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

    state_counts = state.value_counts()
    section(f"{match_id} · {ds.metadata.teams[0].name} vs {ds.metadata.teams[1].name}",
            f"{len(df):,} frames @ {fps:.0f} fps")
    has_state = state.isin(["alive", "dead"]).mean()
    has_owner = df["ball_owning_team_id"].notna().mean()
    t = table("label field", "present on", "verdict")
    t.add_row("ball alive/dead (BallStatus)", bar(has_state, 14, "green"),
              grade(has_state, 0.95, 0.5, fmt="{:.0%}"))
    t.add_row("owning team (BallPossession)", bar(has_owner, 14, "green"),
              grade(has_owner, 0.95, 0.5, fmt="{:.0%}"))
    t.add_row("ball position", bar(1 - df["ball_x"].isna().mean(), 14, "green"), "")
    console.print(t)

    poss, dead = [], []
    for p in df["period_id"].unique():
        for label, n in spells(lab[(df["period_id"] == p).to_numpy()]):
            (dead if label == "D" else poss).append(n / fps)
    poss, dead = np.array(poss), np.array(dead)
    mins = (lab != "D").sum() / fps / 60
    t = table("", "count", "mean s", "median s", "p90 s")
    t.add_row("possessions", f"{len(poss)}", grade(poss.mean(), 18, 10, fmt="{:.1f}"),
              f"{np.median(poss):.1f}", f"{np.percentile(poss, 90):.1f}")
    t.add_row("dead-ball stretches", f"{len(dead)}", f"{dead.mean():.1f}", f"{np.median(dead):.1f}",
              f"{np.percentile(dead, 90):.1f}")
    console.print(t)
    console.print(f"labels: [red]H {np.mean(lab == 'H'):.1%}[/red]  [blue]A {np.mean(lab == 'A'):.1%}[/blue]  "
                  f"[yellow]D {np.mean(lab == 'D'):.1%}[/yellow]   ·   {len(poss) / mins:.1f} possessions per alive minute   "
                  f"·   raw ball_state values: {dict(state_counts)}")
    return dict(n_poss=len(poss), mean_poss=poss.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    title("Open DFL data as training labels", "public data · safe to share")
    res = [check(m) for m in (MATCHES if args.all else MATCHES[:1])]
    section("Verdict")
    insight("challenge public window: 17 possessions, mean 14.2 s. "
            f"DFL here: mean {np.mean([r['mean_poss'] for r in res]):.1f} s. "
            "Similar means → the two label definitions likely agree; very different → we'll need to adapt.")
    save("check_dfl")
