"""v2 stage 2: train on DFL, validate honestly, tune the decoder. Public data, output safe to share.

    python scripts/train_v2.py                 # leave-one-match-out over the DFL matches (~10-20 min)
    python scripts/train_v2.py --min-hold 3    # try another label translation (seconds)
    python scripts/train_v2.py --copies 3      # train on 3 differently-degraded copies of each match
    python scripts/train_v2.py --save          # write the tuned decoder to outputs/v2_config.json

For each DFL match: train on the others (degraded), predict it (degraded), decode, and score
with the OFFICIAL spellF1 against its translated labels. v1's rules run on the same degraded
matches as a reference. The decoder costs with the best mean spellF1 are saved for predict_v2.py.
"""
import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import challenge as ch  # noqa: E402
from possession import dfl, v1, v2  # noqa: E402
from possession.report import console, grade, insight, save, section, table, title, status  # noqa: E402

pm = ch.pm
GRID_HA = (10, 20, 40, 80, 160, 320)
GRID_D = (20, 40, 80, 160, 320)


def score(truth, pred, period):
    r = pm.evaluate(v2.to_official(truth, period), v2.to_official(pred, period))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hold", type=float, default=None, help="seconds; default = prepare_dfl's choice")
    ap.add_argument("--matches", type=int, default=0, help="use only the first N matches (quick test)")
    ap.add_argument("--copies", type=int, default=1, help="degraded copies of each training match")
    ap.add_argument("--save", action="store_true", help="save the tuned settings to outputs/v2_config.json")
    args = ap.parse_args()

    title("v2 · stage 2 · train on DFL, validate with the official spellF1", "public data · safe to share")
    cfg_path = ch.THIS_REPO / "outputs" / "v2_config.json"
    cfg0 = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    hold_file = dfl.cache_dir() / "min_hold.txt"
    hold = (int(args.min_hold * 25) if args.min_hold is not None else cfg0.get("min_hold")
            or (int(hold_file.read_text()) if hold_file.exists() else 50))
    console.print(f"label translation: min_hold = {hold / 25:.1f} s · training copies per match: {args.copies}")

    data = []
    with status("loading, degrading and building features ..."):
        ids = dfl.MATCHES[: args.matches or None]
        for i, mid in enumerate(ids):
            try:
                m = dfl.load_match(mid)
            except Exception as e:
                console.print(f"[red]✗[/red] {mid}: {e}")
                continue
            y = dfl.controlled_possession(m.truth, m.period, hold)
            copies = [dfl.degrade(m, seed=100 + i + 1000 * c) for c in range(args.copies)]
            Xs = [v2.features(a, m.period, m.pitch, m.fps) for a in copies]
            # copy 0 is the held-out test version; all copies are used when the match trains others
            data.append(dict(m=m, a=copies[0], y=y, X=Xs[0], Xs=Xs))
    console.print(f"{len(data)} matches · {sum(len(d['y']) for d in data):,} frames · {data[0]['X'].shape[1]} features")

    section("Leave-one-match-out", "Train on the others, predict the held-out match. Costs are tuned on all folds together.")
    folds = []
    with status("training folds ..."):
        for k, d in enumerate(data):
            rest = [e for j, e in enumerate(data) if j != k]
            models = v2.fit(pd.concat([x for e in rest for x in e["Xs"]], ignore_index=True),
                            np.concatenate([e["y"] for e in rest for _ in e["Xs"]]))
            folds.append(v2.proba(models, d["X"]))

    best, grid = None, {}
    with status(f"decoding {len(GRID_HA) * len(GRID_D)} cost settings × {len(data)} matches ..."):
        for ha, dd in itertools.product(GRID_HA, GRID_D):
            dec = v2.Decoder(switch_ha=ha, switch_d=dd)
            s = [score(d["y"], v2.decode(p, d["m"].period, dec), d["m"].period)[0] for d, p in zip(data, folds)]
            grid[(ha, dd)] = float(np.mean(s))
            best = max(best or (grid[(ha, dd)], ha, dd), (grid[(ha, dd)], ha, dd))

    t = table("switch-ha  \\  switch-d", *[str(x) for x in GRID_D], title="mean spellF1 over held-out matches")
    for ha in GRID_HA:
        t.add_row(str(ha), *[grade(grid[(ha, dd)], best[0] - 0.01, best[0] - 0.05) for dd in GRID_D])
    console.print(t)
    dec = v2.Decoder(switch_ha=best[1], switch_d=best[2])

    t = table("held-out match", "spellF1 v2", "spellF1 v1 rules", "macroF1 v2", "possessions v2 / truth")
    rows = []
    for d, p in zip(data, folds):
        s2, f2, n2, ng = score(d["y"], v2.decode(p, d["m"].period, dec), d["m"].period)
        ev = v1.build_evidence(d["a"], [], v1.Params(d_bias=-2.0, stickiness=2, d_cost_ratio=4))
        lab1 = np.full(len(d["y"]), 0, np.int8)
        ll = v1.emissions(ev)
        for q in np.unique(d["m"].period):
            idx = np.flatnonzero(d["m"].period == q)
            lab1[idx] = v1.viterbi(ll[idx], v1.Params(d_bias=-2.0, stickiness=2, d_cost_ratio=4))
        s1 = score(d["y"], lab1, d["m"].period)[0]
        rows.append((s2, s1, f2, ng))
        t.add_row(d["m"].name, grade(s2, 0.5, 0.25), grade(s1, 0.5, 0.25), f"{f2:.3f}", f"{n2} / {ng}")
    m2, m1, mf, mg = np.mean(rows, axis=0)
    t.add_row("[bold]mean[/bold]", grade(m2, 0.5, 0.25), grade(m1, 0.5, 0.25), f"{mf:.3f}", "")
    console.print(t)
    insight(f"v2 {m2:.3f} vs v1 rules {m1:.3f} on {len(data)} held-out matches "
            f"(~{mg:.0f} possessions each). "
            f"Best decoder: switch-ha {best[1]}, switch-d {best[2]}.")

    if args.save:
        cfg = {**cfg0, **dict(min_hold=hold, switch_ha=best[1], switch_d=best[2], copies=args.copies,
                              cv_spell_f1=m2, cv_spell_f1_v1=m1)}
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        console.print("[dim]saved → outputs/v2_config.json (used by predict_v2.py)[/dim]")
    else:
        console.print("[dim]not saved (add --save to make predict_v2.py use these settings)[/dim]")


if __name__ == "__main__":
    main()
    save("train_v2")
