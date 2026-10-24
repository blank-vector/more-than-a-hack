"""Profile the challenge tracking. Prints AGGREGATE statistics only.

    python scripts/profile_atd.py

Everything printed is a summary (percentages, percentiles, counts), never frames or positions,
so the output is safe to share. It is also saved to outputs/profile_atd.txt.

What we want to learn, and why:
  1. how much ball is missing, and in what shape (many short gaps vs a few long ones)
     -> decides how to fill gaps and where to abstain (X)
  2. how fragmented the player tracks are -> decides team-level vs track-level features
  3. on the public 5 minutes: how ball visibility and player speeds differ between
     possession (H/A) and dead ball (D) -> first hints for a dead-ball detector
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402


def runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def pct(a, qs=(10, 25, 50, 75, 90, 99)):
    if len(a) == 0:
        return "n/a"
    return "  ".join(f"p{q}={np.percentile(a, q):.2f}" for q in qs)


def main():
    a = atd_mod.load()
    T = a.n_frames
    graded = ch.graded_mask(T)
    fps = ch.FPS
    ball_ok = ~np.isnan(a.ball[:, 0])
    pres = ~np.isnan(a.xy[..., 0])

    print("=== 1. Ball visibility")
    print(f"frames {T}, graded {graded.sum()}")
    print(f"ball visible: all {ball_ok.mean():.1%}   graded {ball_ok[graded].mean():.1%}")
    for half in (1, 2):
        h = ch.half_index(T) == half
        print(f"  half {half}: {ball_ok[h].mean():.1%}")

    gaps = []
    for half in (1, 2):
        h = ch.half_index(T) == half
        gaps.append(runs(~ball_ok & h))
    gaps = np.concatenate(gaps) / fps
    print(f"\nball gaps in graded halves: {len(gaps)} gaps, seconds: {pct(gaps)}")
    missing = gaps.sum()
    for s in (0.2, 0.5, 1, 2, 5, 10, 30):
        print(f"  share of missing time in gaps longer than {s:>4}s: {gaps[gaps > s].sum() / missing:.1%}"
              f"   (n={int((gaps > s).sum())})")
    vis = np.concatenate([runs(ball_ok & (ch.half_index(T) == h)) for h in (1, 2)]) / fps
    print(f"ball visible stretches: {len(vis)}, seconds: {pct(vis)}")

    print("\n=== 2. Player tracks")
    n_home = pres[:, a.team == 0].sum(1)
    n_away = pres[:, a.team == 1].sum(1)
    print(f"slots: {len(a.team)} (home {int((a.team == 0).sum())}, away {int((a.team == 1).sum())})")
    print(f"tracked per graded frame: home {pct(n_home[graded], (5, 50, 95))}   away {pct(n_away[graded], (5, 50, 95))}")
    print(f"graded frames with <9 tracked on a team: home {(n_home[graded] < 9).mean():.1%}  away {(n_away[graded] < 9).mean():.1%}")
    frag = np.concatenate([runs(pres[:, s] & graded) for s in range(pres.shape[1])]) / fps
    print(f"track fragments: {len(frag)}, seconds: {pct(frag)}")
    sp = a.speed[graded][pres[graded]]
    print(f"player speed m/s (graded): {pct(sp[~np.isnan(sp)], (5, 50, 95, 99))}")

    bx = a.ball[graded & ball_ok]
    out = (np.abs(bx[:, 0]) > ch.PITCH[0] / 2) | (np.abs(bx[:, 1]) > ch.PITCH[1] / 2)
    print(f"\nvisible ball outside the pitch lines (graded): {out.mean():.1%}")

    print("\n=== 3. Public 5 minutes, by ground-truth label")
    gt = ch.public_gt()
    names = {0: "H", 1: "A", 2: "D"}
    mean_speed = np.nanmean(np.where(pres, a.speed, np.nan), axis=1)
    bvel = np.linalg.norm(np.gradient(a.ball, axis=0), axis=1) * fps
    for code, name in names.items():
        m = gt == code
        if not m.any():
            continue
        print(f"  {name}: frames {m.sum():>5}  ball visible {ball_ok[m].mean():.1%}  "
              f"mean player speed {np.nanmean(mean_speed[m]):.2f} m/s  "
              f"ball speed median {np.nanmedian(bvel[m & ball_ok]):.1f} m/s")
    alive = (gt == 0) | (gt == 1)
    m_ok = ~np.isnan(mean_speed)
    for thr in (1.0, 1.5, 2.0, 2.5):
        slow = (mean_speed < thr) & m_ok
        print(f"  mean player speed < {thr} m/s: in D {slow[gt == 2].mean():.1%}  in H/A {slow[alive].mean():.1%}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    text = buf.getvalue()
    print(text)
    out = ch.THIS_REPO / "outputs"
    out.mkdir(exist_ok=True)
    (out / "profile_atd.txt").write_text(text)
    print(f"saved to {out / 'profile_atd.txt'}")
