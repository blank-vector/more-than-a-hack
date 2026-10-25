"""Profile the challenge tracking. Prints AGGREGATE statistics only.

    python scripts/profile_atd.py

Everything printed is a summary (percentages, percentiles, counts), never frames or positions,
so the output is safe to share. A plain-text copy is saved to outputs/profile_atd.txt.

What we want to learn, and why:
  1. how much ball is missing, and in what shape (many short gaps vs a few long ones)
     -> decides how to fill gaps and where to abstain (X)
  2. how fragmented the player tracks are -> decides team-level vs track-level features
  3. on the public 5 minutes: how ball visibility and player speeds differ between
     possession (H/A) and dead ball (D) -> first hints for a dead-ball detector
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402
from possession.report import bar, console, grade, insight, save, section, table, title, status  # noqa: E402


def runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def pct_row(name: str, a: np.ndarray, qs=(10, 25, 50, 75, 90, 99), fmt="{:.2f}"):
    if len(a) == 0:
        return [name, str(0)] + ["-"] * len(qs)
    return [name, f"{len(a):,}"] + [fmt.format(np.percentile(a, q)) for q in qs]


def main():
    title("Challenge tracking profile", "aggregate stats only · safe to share")
    with status("loading tracking (first run parses the file, ~1-2 min) ..."):
        a = atd_mod.load()
    T = a.n_frames
    graded = ch.graded_mask(T)
    halves = ch.half_index(T)
    fps = ch.FPS
    ball_ok = ~np.isnan(a.ball[:, 0])
    pres = ~np.isnan(a.xy[..., 0])

    # ------------------------------------------------------------------ 1. ball
    section("1 · Ball visibility", "How often the tracker sees the ball, and the shape of the gaps.")
    t = table("", "ball visible")
    t.add_row("all frames", bar(ball_ok.mean(), style="green"))
    t.add_row("graded frames", bar(ball_ok[graded].mean(), style="green"))
    for h in (1, 2):
        t.add_row(f"  half {h}", bar(ball_ok[halves == h].mean(), style="green"))
    console.print(t)

    gaps = np.concatenate([runs(~ball_ok & (halves == h)) for h in (1, 2)]) / fps
    vis = np.concatenate([runs(ball_ok & (halves == h)) for h in (1, 2)]) / fps
    t = table("seconds", "count", "p10", "p25", "p50", "p75", "p90", "p99", title="Stretch lengths (graded)")
    t.add_row(*pct_row("[red]ball missing[/red]", gaps))
    t.add_row(*pct_row("[green]ball visible[/green]", vis))
    console.print(t)

    t = table("gaps longer than", "gaps", "share of all missing time", title="Where the missing time lives")
    missing = gaps.sum()
    for s in (0.2, 0.5, 1, 2, 5, 10, 30):
        share = gaps[gaps > s].sum() / missing if missing else 0
        t.add_row(f"{s} s", f"{int((gaps > s).sum()):,}", bar(share, style="red"))
    console.print(t)
    long_share = gaps[gaps > 2].sum() / missing if missing else 0
    insight(f"{long_share:.0%} of missing-ball time is in gaps over 2 s "
            + ("→ long gaps dominate: we need team-shape signals, not just interpolation."
               if long_share > 0.5 else "→ mostly short gaps: interpolation/bridging should go far."))

    # ------------------------------------------------------------------ 2. players
    section("2 · Player tracks", "Slots are tracklets, not people. How many are visible, and how long do they last?")
    n_home = pres[:, a.team == 0].sum(1)[graded]
    n_away = pres[:, a.team == 1].sum(1)[graded]
    t = table("", "slots", "tracked p5", "p50", "p95", "frames with < 9 tracked")
    for name, n, k in (("[red]home[/red]", n_home, 0), ("[blue]away[/blue]", n_away, 1)):
        t.add_row(name, str(int((a.team == k).sum())), f"{np.percentile(n, 5):.0f}",
                  f"{np.percentile(n, 50):.0f}", f"{np.percentile(n, 95):.0f}",
                  grade((n < 9).mean(), 0.02, 0.10, higher_is_better=False, fmt="{:.1%}"))
    console.print(t)

    frag = np.concatenate([runs(pres[:, s] & graded) for s in range(pres.shape[1])]) / fps
    sp = a.speed[graded][pres[graded]]
    sp = sp[~np.isnan(sp)]
    t = table("", "count", "p10", "p25", "p50", "p75", "p90", "p99")
    t.add_row(*pct_row("track fragment length (s)", frag))
    t.add_row(*pct_row("player speed (m/s)", sp))
    console.print(t)
    insight(f"median track fragment lasts {np.median(frag):.1f} s "
            + ("→ too short to follow individual players; work at team level."
               if np.median(frag) < 30 else "→ long enough for some per-player features."))

    bx = a.ball[graded & ball_ok]
    out = (np.abs(bx[:, 0]) > ch.PITCH[0] / 2) | (np.abs(bx[:, 1]) > ch.PITCH[1] / 2)
    console.print(f"visible ball outside the pitch lines (graded): [bold]{out.mean():.1%}[/bold]")

    # ------------------------------------------------------------------ 3. public window
    section("3 · Public 5 minutes, by true label",
            "What differs between possession (H/A) and dead ball (D)? These are dead-ball detector hints.")
    gt = ch.public_gt()
    mean_speed = np.nanmean(np.where(pres, a.speed, np.nan), axis=1)
    bvel = np.linalg.norm(np.gradient(a.ball, axis=0), axis=1) * fps
    t = table("true label", "frames", "ball visible", "mean player speed", "ball speed (median)")
    styles = {0: ("H home", "red"), 1: ("A away", "blue"), 2: ("D dead", "yellow")}
    for code, (name, st) in styles.items():
        m = gt == code
        if m.any():
            t.add_row(f"[{st}]{name}[/{st}]", f"{m.sum():,}", bar(ball_ok[m].mean(), 12, "green"),
                      f"{np.nanmean(mean_speed[m]):.2f} m/s", f"{np.nanmedian(bvel[m & ball_ok]):.1f} m/s")
    console.print(t)

    alive = (gt == 0) | (gt == 1)
    ok = ~np.isnan(mean_speed)
    t = table("rule: mean player speed <", "catches D (want high)", "false alarms in H/A (want low)",
              title="Can 'everyone slows down' detect dead ball?")
    best = (-1.0, 0.0)
    for thr in (1.0, 1.5, 2.0, 2.5, 3.0):
        slow = (mean_speed < thr) & ok
        recall, fa = slow[gt == 2].mean(), slow[alive].mean()
        best = max(best, (recall - fa, thr))
        t.add_row(f"{thr} m/s", bar(recall, 14, "green"), bar(fa, 14, "red"))
    console.print(t)
    if best[0] > -1:
        insight(f"best separation at {best[1]} m/s (recall − false alarms = {best[0]:.2f}). "
                "Above ~0.4 means speed alone is a useful dead-ball signal.")


if __name__ == "__main__":
    main()
    save("profile_atd")
