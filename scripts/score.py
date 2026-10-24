"""A readable scoreboard, using the OFFICIAL metric code.

    python scripts/score.py                 # official baselines + everything in outputs/submissions/
    python scripts/score.py my.csv other.csv

Scores against the public 5 minutes (17 possessions). Remember: that window checks that a
pipeline is CORRECT; it is far too small to choose between two versions. Gaps under ~0.04
spellF1 mean nothing even on the full match.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import challenge as ch  # noqa: E402
from possession.report import console, grade, insight, save, section, table, title  # noqa: E402

pm = ch.pm
FLOOR = 0.2433  # published full-match floor (baseline_smooth50)


def files_from_args() -> list[Path]:
    if len(sys.argv) > 1:
        return [Path(p) for p in sys.argv[1:]]
    files = sorted((ch.OFFICIAL / "evaluation" / "results").glob("*.csv"))
    files += sorted((ch.THIS_REPO / "outputs" / "submissions").glob("*.csv"))
    return files


def main():
    title("Scoreboard · public 5 minutes", "official metric code · not a leaderboard")
    gt = ch.public_gt()
    runs = pm.valid_runs(gt)
    gt_stats = pm.possession_stats(gt, runs)
    n_gt = len(pm.possession_spells(gt, runs))

    rows, skipped = [], []
    for f in files_from_args():
        try:
            frame = pm.read_label_csv(str(f))
            why = pm.validate_submission(frame, len(gt))
        except Exception as e:  # unreadable file
            why = str(e)
        if why:
            skipped.append((f.stem, why))
            continue
        r = pm.evaluate(gt, pm.label_codes(frame))
        rows.append((f.stem, r))
    rows.sort(key=lambda x: (-x[1]["spell_f1_mean"], -x[1]["macro_f1"]))

    section("Ranking", f"truth: {n_gt} possessions, mean {gt_stats['mean_duration_s']:.1f} s. "
                       "Possession count close to the truth matters as much as the scores.")
    t = table("#", "submission", "spellF1", "macroF1", "possessions", "mean length", "dead-ball F1", "abstained")
    t.columns[1].justify = "left"
    for i, (name, r) in enumerate(rows, 1):
        n = r["n_pred_spells"]
        err = abs(n - n_gt) / max(n_gt, 1)
        poss = grade(err, 0.2, 0.6, higher_is_better=False, fmt="{:.0%}")
        poss.plain = f"{n} vs {n_gt}"
        dur = r["mean_spell_duration_s"]
        t.add_row(str(i), name,
                  grade(r["spell_f1_mean"], 0.5, FLOOR),
                  grade(r["macro_f1"], 0.75, 0.6),
                  poss,
                  f"{dur:.1f} s",
                  grade(r["per_class"]["D"]["f1"], 0.7, 0.4),
                  f"{r['n_abstain_on_graded']:,}")
    console.print(t)
    console.print("[dim]spellF1: green ≥ 0.50, yellow ≥ published floor 0.2433, red below. "
                  "possessions: green within 20% of truth, red off by > 60%.[/dim]")

    if rows:
        best = rows[0]
        insight(f"best here: [bold]{best[0]}[/bold] — but with only {n_gt} possessions, "
                "one missed possession moves spellF1 by ~0.06. Don't pick versions on this window.")

    if skipped:
        section("Skipped (would score zero)")
        for name, why in skipped:
            console.print(f"[bold red]✗[/bold red] {name}: {why}")


if __name__ == "__main__":
    main()
    save("scoreboard")
