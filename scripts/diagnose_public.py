"""Where does a submission go wrong on the public 5 minutes? Truth vs prediction, side by side.

    python scripts/diagnose_public.py                 # outputs/submissions/v1.csv
    python scripts/diagnose_public.py path/to/sub.csv

Shows only labels (the public answers and our predictions), never tracking, so it is safe to share.
Use it to SEE failure modes (late starts, false dead balls, wrong team), not to tune numbers.
"""
import sys
from pathlib import Path

import numpy as np
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import challenge as ch  # noqa: E402
from possession.report import console, insight, save, section, table, title  # noqa: E402

pm = ch.pm
STYLE = {0: ("H", "white on red"), 1: ("A", "white on blue"), 2: ("D", "black on yellow"), 3: ("X", "dim")}
SEC = ch.FPS


def segments(codes: np.ndarray, lo: int, hi: int):
    """(start, end_exclusive, code) runs inside [lo, hi)."""
    out, s = [], lo
    for i in range(lo + 1, hi + 1):
        if i == hi or codes[i] != codes[s]:
            out.append((s, i, int(codes[s])))
            s = i
    return out


def strip(codes: np.ndarray, lo: int, hi: int) -> Text:
    """One character per second, the majority label of that second."""
    t = Text()
    for s in range(lo, hi, SEC):
        c = int(np.bincount(codes[s:s + SEC], minlength=4).argmax())
        t.append(STYLE[c][0], style=STYLE[c][1])
    return t


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ch.THIS_REPO / "outputs" / "submissions" / "v1.csv"
    title(f"Public 5 minutes · truth vs {path.stem}", "labels only · safe to share")
    gt = ch.public_gt()
    pred = pm.label_codes(pm.read_label_csv(str(path)))
    graded = np.flatnonzero(gt != 3)
    lo, hi = int(graded[0]), int(graded[-1]) + 1

    section("Timeline", "1 character = 1 second.  H red · A blue · D yellow.  Each row = 1 minute.")
    for m0 in range(lo, hi, 60 * SEC):
        m1 = min(m0 + 60 * SEC, hi)
        minute = (m0 - lo) // (60 * SEC)
        console.print(Text(f"min {minute}  truth ", style="bold") + strip(gt, m0, m1))
        console.print(Text(f"min {minute}  {path.stem[:5]:<5} ", style="bold") + strip(pred, m0, m1))
        console.print()

    section("Each true segment: what did we predict there?")
    t = table("#", "truth", "start", "length", "we predicted", "best overlap (IoU)", "verdict")
    for i, (s, e, c) in enumerate(segments(gt, lo, hi), 1):
        inside = pred[s:e]
        shares = np.bincount(inside, minlength=4) / len(inside)
        main_c = int(shares.argmax())
        # best IoU with any predicted run of the same label overlapping this segment
        best = 0.0
        for ps, pe, pc in segments(pred, lo, hi):
            if pc == c and pe > s and ps < e:
                best = max(best, (min(e, pe) - max(s, ps)) / (max(e, pe) - min(s, ps)))
        mix = "  ".join(f"{STYLE[k][0]} {shares[k]:.0%}" for k in range(4) if shares[k] >= 0.05)
        verdict = ("[green]match[/green]" if best >= 0.5 else "[yellow]partial[/yellow]" if best >= 0.3
                   else "[red]missed[/red]")
        t.add_row(str(i), Text(STYLE[c][0], style=STYLE[c][1]), f"{(s - lo) / SEC:5.1f} s",
                  f"{(e - s) / SEC:.1f} s", mix, f"{best:.2f}", verdict)
    console.print(t)

    section("Boundary timing", "For each true change of label, how late (+) or early (−) was ours?")
    lags = []
    pred_changes = np.flatnonzero(np.diff(pred[lo:hi]) != 0) + lo + 1
    for s, e, c in segments(gt, lo, hi)[1:]:
        near = pred_changes[np.abs(pred_changes - s) <= 5 * SEC]
        lags.append((near[np.argmin(np.abs(near - s))] - s) / SEC if len(near) else None)
    found = [x for x in lags if x is not None]
    console.print(f"true changes: {len(lags)}   matched within 5 s: {len(found)}   "
                  f"median lag: {np.median(found):+.1f} s" if found else "no boundaries matched")
    extra = len(pred_changes) - len(found)
    console.print(f"our changes with no true change nearby: [bold]{max(extra, 0)}[/bold]")
    insight("look for: red rows (missed possessions), D where the truth is H/A (false stoppages), "
            "and a consistent positive lag (we switch late).")


if __name__ == "__main__":
    main()
    save("diagnose_public")
