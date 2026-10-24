"""What happens while the ball is missing? Prints AGGREGATE statistics only (safe to share).

    python scripts/profile_gaps.py

The ball is missing on 41% of graded frames, mostly in long gaps (median ~11 s). Before
designing anything for those gaps we need to know:
  1. WHEN: is missing ball spread over the match, or concentrated in some periods?
  2. LIVE OR DEAD: during a gap, are players running (live play) or standing (dead ball)?
  3. WHERE: where was the ball last seen before it vanished (near a line? in a box?)
  4. HOW FAR: how far is the ball from where it vanished when it reappears?
  5. TRACKING: do player tracks degrade during gaps too (a camera problem) or not?
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402
from possession.report import bar, console, insight, save, section, table, title  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

SLOW = 1.5  # m/s, mean player speed below which play is probably dead (from profile_atd)


def run_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of consecutive True runs."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def zone(x: float, y: float) -> str:
    L, W = ch.PITCH
    if abs(x) > L / 2 or abs(y) > W / 2:
        return "outside the lines"
    if abs(x) > L / 2 - 16.5 and abs(y) < 20.16:
        return "penalty box"
    if abs(y) > W / 2 - 3:
        return "within 3 m of a touchline"
    if abs(x) > L / 2 - 3:
        return "within 3 m of a goal line"
    return "open play"


def main():
    title("What happens while the ball is missing?", "aggregate stats only · safe to share")
    with console.status("loading tracking ..."):
        a = atd_mod.load()
    T, fps = a.n_frames, ch.FPS
    halves = ch.half_index(T)
    graded = halves > 0
    ball_ok = ~np.isnan(a.ball[:, 0])
    pres = ~np.isnan(a.xy[..., 0])
    n_tracked = pres.sum(1)
    speed = np.nanmean(np.where(pres, a.speed, np.nan), axis=1)

    # ------------------------------------------------------------------ 1. when
    section("1 · When is the ball missing?", "Ball visibility per 5 minutes of each half.")
    t = table("block", "ball visible", "players tracked (mean)", "mean player speed")
    for h in (1, 2):
        idx = np.flatnonzero(halves == h)
        step = 5 * 60 * fps
        for k, s in enumerate(range(idx[0], idx[-1] + 1, step)):
            blk = slice(s, min(s + step, idx[-1] + 1))
            t.add_row(f"half {h} · {5 * k:>2}–{5 * k + 5} min", bar(ball_ok[blk].mean(), 16, "green"),
                      f"{n_tracked[blk].mean():.1f}", f"{np.nanmean(speed[blk]):.2f} m/s")
    console.print(t)

    gaps = [(s, e) for h in (1, 2) for s, e in run_bounds(~ball_ok & (halves == h))]
    long_gaps = [(s, e) for s, e in gaps if (e - s) >= 5 * fps]
    console.print(f"gaps ≥ 5 s: [bold]{len(long_gaps)}[/bold] of {len(gaps)}, "
                  f"covering {sum(e - s for s, e in long_gaps) / graded.sum():.1%} of graded frames")

    # ------------------------------------------------------------------ 2. live or dead
    section("2 · Live or dead during gaps?",
            f"Share of each gap where mean player speed < {SLOW} m/s (our dead-ball hint).")
    slow_share = np.array([np.nanmean(speed[s:e] < SLOW) for s, e in long_gaps])
    dur = np.array([(e - s) / fps for s, e in long_gaps])
    t = table("gap is ...", "gaps", "share of long-gap time")
    for name, m in (("mostly running (< 20% slow)", slow_share < 0.2),
                    ("mixed (20–80% slow)", (slow_share >= 0.2) & (slow_share <= 0.8)),
                    ("mostly standing (> 80% slow)", slow_share > 0.8)):
        t.add_row(name, f"{int(m.sum())}", bar(dur[m].sum() / dur.sum() if dur.sum() else 0, 16, "cyan"))
    console.print(t)
    gap_frames = np.zeros(T, bool)
    for s, e in long_gaps:
        gap_frames[s:e] = True
    vis_alive_like = ball_ok & graded
    console.print(f"frames with mean speed < {SLOW} m/s: in long gaps [bold]{np.nanmean(speed[gap_frames] < SLOW):.1%}[/bold]"
                  f"   ·   while ball visible [bold]{np.nanmean(speed[vis_alive_like] < SLOW):.1%}[/bold]")

    # ------------------------------------------------------------------ 3. where
    section("3 · Where was the ball last seen before a long gap?")
    zones = {}
    for s, e in long_gaps:
        if s == 0 or not ball_ok[s - 1]:
            zones["unknown (no sighting before)"] = zones.get("unknown (no sighting before)", 0) + 1
            continue
        z = zone(*a.ball[s - 1])
        zones[z] = zones.get(z, 0) + 1
    t = table("last seen", "gaps", "share")
    for z, n in sorted(zones.items(), key=lambda kv: -kv[1]):
        t.add_row(z, str(n), bar(n / max(len(long_gaps), 1), 16, "magenta"))
    console.print(t)

    # ------------------------------------------------------------------ 4. how far
    section("4 · How far did the ball move across a gap?", "Distance between last sighting and first re-sighting.")
    jumps = np.array([np.linalg.norm(a.ball[e] - a.ball[s - 1]) for s, e in long_gaps
                      if s > 0 and e < T and ball_ok[s - 1] and ball_ok[e]])
    if not long_gaps:
        console.print("[dim]no long gaps[/dim]")
    if len(jumps):
        t = table("", "gaps", "p10", "p25", "p50", "p75", "p90")
        t.add_row("distance (m)", str(len(jumps)),
                  *[f"{np.percentile(jumps, q):.1f}" for q in (10, 25, 50, 75, 90)])
        console.print(t)
        console.print(f"reappears within 5 m of where it vanished: [bold]{np.mean(jumps < 5):.0%}[/bold] of gaps")

    # ------------------------------------------------------------------ 5. tracking
    section("5 · Do player tracks degrade during gaps?")
    t = table("", "players tracked (mean)", "frames with < 18 tracked")
    t.add_row("ball visible", f"{n_tracked[ball_ok & graded].mean():.1f}", bar(np.mean(n_tracked[ball_ok & graded] < 18), 14, "red"))
    t.add_row("ball in a long gap", f"{n_tracked[gap_frames].mean():.1f}", bar(np.mean(n_tracked[gap_frames] < 18), 14, "red"))
    console.print(t)

    mostly_standing = dur[slow_share > 0.8].sum() / dur.sum() if dur.sum() else 0
    insight(f"{mostly_standing:.0%} of long-gap time looks like dead ball (players standing). "
            "The rest is live play where we must infer possession without the ball.")


if __name__ == "__main__":
    main()
    save("profile_gaps")
