"""Build the version-1 submission and check it.

    python scripts/predict_v1.py                       # writes outputs/submissions/v1.csv
    python scripts/predict_v1.py --switch-ha 40        # override any Params field (dashes or underscores)

Prints only aggregates (safe to share): validation, label shares, possession counts, and the
Smart Tagging codes' learned meaning. Then run `python scripts/score.py` for the public window.

Sanity target (not a score): the public window has ~4.2 possessions per minute of live play.
A full-match prediction far from that rate is over- or under-segmenting.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402
from possession import v1  # noqa: E402
from possession.report import bar, console, grade, insight, save, section, table, title  # noqa: E402

TARGET_RATE = 4.2  # possessions per live minute in the public window


def runs(labels: np.ndarray, which: str) -> np.ndarray:
    m = np.concatenate(([False], labels == which, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def possession_rate(labels: np.ndarray) -> tuple[int, float, float]:
    halves = ch.half_index(len(labels))
    poss = np.concatenate([runs(labels[halves == h], L) for h in (1, 2) for L in "HA"]) / ch.FPS
    live_min = np.isin(labels[halves > 0], ["H", "A"]).sum() / ch.FPS / 60
    return len(poss), len(poss) / max(live_min, 1e-9), float(poss.mean()) if len(poss) else 0.0


TARGET_DEAD = 0.45  # dead-ball share: DFL matches 41-52%, typical top-flight football ~40-50%


def decode(ll, halves, p):
    labels = np.full(len(ll), "X", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(halves == h)
        labels[idx] = v1.LETTERS[v1.viterbi(ll[idx], p)]
    return labels


def sweep(a, tags, p):
    section("Sweep: dead-ball threshold × stickiness",
            f"Each cell: possessions per live minute / dead-ball share. Targets ≈ {TARGET_RATE} and ≈ {TARGET_DEAD:.0%}. "
            "No ground truth used.")
    halves = ch.half_index(a.n_frames)
    biases, sticks = (-3.0, -2.0, -1.0, 0.0, 1.0), (1, 2, 4, 8)
    t = table("--d-bias  \\  --stickiness", *[str(s_) for s_ in sticks])
    best = None
    with console.status("decoding 20 combinations (~1 min) ..."):
        for b in biases:
            ev = v1.build_evidence(a, tags, dataclasses.replace(p, d_bias=b))
            ll = v1.emissions(ev)
            cells = []
            for s_ in sticks:
                labels = decode(ll, halves, dataclasses.replace(p, d_bias=b, stickiness=float(s_)))
                _, rate, _ = possession_rate(labels)
                dead = float(np.mean(labels[halves > 0] == "D"))
                miss = abs(np.log(max(rate, 1e-9) / TARGET_RATE)) + 3 * abs(dead - TARGET_DEAD)
                best = min(best or (miss, b, s_), (miss, b, s_))
                style = "bold green" if miss < 0.25 else "yellow" if miss < 0.6 else "red"
                cells.append(Text(f"{rate:.1f} / {dead:.0%}", style=style))
            t.add_row(str(b), *cells)
    console.print(t)
    insight(f"closest to both targets: --d-bias {best[1]} --stickiness {best[2]}\n"
            f"   →  python scripts/predict_v1.py --d-bias {best[1]} --stickiness {best[2]}")


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(v1.Params):
        if f.type in ("float", float):
            ap.add_argument("--" + f.name.replace("_", "-"), type=float, default=None)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--sweep", action="store_true",
                    help="try several --switch-ha values and show the possession rate of each (no ground truth used)")
    args = ap.parse_args()
    p = v1.Params(**{k: val for k, val in vars(args).items() if k not in ("name", "sweep") and val is not None})

    title(f"Version 1 submission · {args.name}", "rule-based evidence + Viterbi · aggregates only")
    with console.status("loading tracking and tags ..."):
        a = atd_mod.load()
        tags = v1.load_tags()
    with console.status("building evidence and decoding ..."):
        labels, ev = v1.predict(a, tags, p)

    if args.sweep:
        sweep(a, tags, p)
        return

    out = ch.THIS_REPO / "outputs" / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.csv"
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)

    frame = ch.pm.read_label_csv(str(path))
    why = ch.pm.validate_submission(frame, ch.N_FRAMES)
    section("Validation")
    console.print("[bold green]✓ valid submission[/bold green]" if why is None else f"[bold red]✗ {why}[/bold red]")
    console.print(f"[dim]{path}[/dim]")

    graded = ch.graded_mask(len(labels))
    g = labels[graded]
    section("What it predicts (full match)")
    console.print(f"dead-ball share: [bold]{np.mean(g == 'D'):.0%}[/bold]  [dim](target ≈ {TARGET_DEAD:.0%})[/dim]")
    t = table("label", "share of graded frames")
    for L, st in (("H", "red"), ("A", "blue"), ("D", "yellow")):
        t.add_row(f"[{st}]{L}[/{st}]", bar(np.mean(g == L), 20, st))
    console.print(t)

    poss = np.concatenate([runs(labels[ch.half_index(len(labels)) == h], L) for h in (1, 2) for L in "HA"]) / ch.FPS
    dead = np.concatenate([runs(labels[ch.half_index(len(labels)) == h], "D") for h in (1, 2)]) / ch.FPS
    live_min = np.isin(g, ["H", "A"]).sum() / ch.FPS / 60
    rate = len(poss) / max(live_min, 1e-9)
    t = table("", "count", "mean s", "median s", "p90 s")
    t.add_row("possessions", str(len(poss)), f"{poss.mean():.1f}", f"{np.median(poss):.1f}", f"{np.percentile(poss, 90):.1f}")
    t.add_row("dead-ball stretches", str(len(dead)), f"{dead.mean():.1f}", f"{np.median(dead):.1f}", f"{np.percentile(dead, 90):.1f}")
    console.print(t)
    console.print(f"possessions per live minute: [bold]{rate:.1f}[/bold]  [dim](public window ≈ {TARGET_RATE})[/dim]")
    if rate > TARGET_RATE * 1.3:
        insight("over-segmenting: raise --stickiness (see --sweep).")
    elif rate < TARGET_RATE * 0.7:
        insight("under-segmenting: lower --stickiness (see --sweep).")
    else:
        insight("possession rate is in the right range.")

    section("Smart Tagging codes", "Does each code's Team tag match the nearest player? (learned, no ground truth)")
    t = table("code", "agreement", "tags used", "read as")
    for code, (agree, n) in sorted(ev.report["code_agree"].items(), key=lambda kv: -kv[1][0]):
        sign = ev.report["code_sign"][code]
        t.add_row(code, bar(agree, 12, "cyan"), str(n),
                  "[green]team in possession[/green]" if sign > 0 else "[red]defending team[/red]" if sign < 0 else "[dim]ignored[/dim]")
    console.print(t)
    console.print(f"[dim]ball visible with a player within {p.near_m} m: {ev.report['frac_near']:.1%} of graded frames[/dim]")
    console.print("\nnext: [bold]python scripts/score.py[/bold]")


if __name__ == "__main__":
    main()
    save("predict_v1")
