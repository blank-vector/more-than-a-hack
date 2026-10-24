"""v2 stage 1: prepare the DFL matches as training data. Public data, output safe to share.

    python scripts/prepare_dfl.py              # all matches it can load (~2-5 min each the first time)

  1. loads each match through kloppy and caches challenge-like arrays + truth
  2. chooses `min_hold` (touch-level -> controlled possession) to match ~4.2 possessions per live minute
  3. degrades one copy of each match and checks it looks like the challenge tracking
"""
import sys
from pathlib import Path

import numpy as np
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import dfl  # noqa: E402
from possession.report import bar, console, grade, insight, save, section, table, title  # noqa: E402

TARGET = dict(ball_visible=0.593, gap_median_s=10.7, tracked_per_team=10.5)


def main():
    title("v2 · stage 1 · DFL training data", "public data · safe to share")
    matches = []
    section("1 · Load")
    for mid in dfl.MATCHES:
        try:
            with console.status(f"loading {mid} ..."):
                m = dfl.load_match(mid)
            matches.append(m)
            r, mean, n = dfl.possession_rate(m.truth, m.period, m.fps)
            console.print(f"[green]✓[/green] {mid}: {len(m.ball):,} frames · {m.xy.shape[1]} players · "
                          f"touch-level {r:.1f} poss/live-min · dead {np.mean(m.truth == dfl.D_):.0%}")
        except Exception as e:  # one bad download must not stop the rest
            console.print(f"[red]✗[/red] {mid}: {type(e).__name__}: {str(e)[:120]}")
    if not matches:
        return

    section("2 · Touch-level → controlled possession", "Pick the minimum hold that brings DFL to the challenge rate (~4.2).")
    hold, sweep = dfl.choose_min_hold(matches)
    t = table("min hold", "possessions per live minute")
    for h, r in sweep:
        t.add_row(f"{h / 25:.1f} s" + ("  ← chosen" if h == hold else ""),
                  Text(f"{r:.1f}", style="bold green" if h == hold else ""))
    console.print(t)
    (dfl.cache_dir() / "min_hold.txt").write_text(str(hold))
    insight(f"min_hold = {hold / 25:.1f} s saved; training labels will use it.")

    section("3 · Degraded copies vs the real challenge tracking")
    t = table("match", "ball visible", "missing | dead", "missing | live", "gap median", "tracked per team")
    for i, m in enumerate(matches):
        r = dfl.degrade_report(m, dfl.degrade(m, seed=i))
        t.add_row(m.name, f"{r['ball_visible']:.0%}", f"{r['miss_dead']:.0%}", f"{r['miss_alive']:.0%}",
                  f"{r['gap_median_s']:.1f} s", f"{r['tracked_per_team']:.1f}")
    t.add_row("[bold]challenge[/bold]", f"{TARGET['ball_visible']:.0%}", "~64%", "~22%",
              f"{TARGET['gap_median_s']} s", f"~{TARGET['tracked_per_team']}")
    console.print(t)
    insight("each degraded match should sit close to the challenge row; if not, adjust DegradeParams.")


if __name__ == "__main__":
    main()
    save("prepare_dfl")
