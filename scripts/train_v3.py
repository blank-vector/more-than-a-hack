"""v3: leave-one-match-out on DFL with the change-aware decoder. Public data, output safe to share.

    python scripts/train_v3.py            # same hold and noise seeds as v2d, so scores compare with v2d's 0.657
    python scripts/train_v3.py --save     # store the best settings in configs/v3.json

The grid includes k = 0 (constant switch costs, like v2), so the table shows directly how much the
change detector adds on top of the new features.
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
from possession import dfl, v2, v3  # noqa: E402
from possession.report import console, grade, insight, save, section, status, table, title  # noqa: E402

pm = ch.pm
GRID_HA = (10, 20, 40, 80)
GRID_D = (20, 40, 80)
GRID_K = (0, 1, 2, 4, 8)
V2D_CV = 0.657


def score(truth, pred, period):
    r = pm.evaluate(v2.to_official(truth, period), v2.to_official(pred, period))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hold", type=float, default=2.0)
    ap.add_argument("--matches", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save-as", type=Path, help="write the chosen settings to this config file (e.g. configs/final.json)")
    ap.add_argument("--dead-from-v2", action="store_true",
                    help="hybrid: dead-ball model trained on v2's 62 features, team/change from v3")
    args = ap.parse_args()
    hold = int(round(args.min_hold * 25))
    title("v3 · change-aware decoder · leave-one-match-out on DFL" + (" · HYBRID dead model (v2 features)" if args.dead_from_v2 else ""),
          f"hold {args.min_hold:g} s · same noise seeds as v2d · public data · safe to share")

    data = []
    with status("loading, degrading and building v3 features ..."):
        for i, mid in enumerate(dfl.MATCHES[: args.matches or None]):
            try:
                m = dfl.load_match(mid)
            except Exception as e:
                console.print(f"[red]✗[/red] {mid}: {e}")
                continue
            a = dfl.degrade(m, seed=100 + i)
            data.append(dict(m=m, y=dfl.controlled_possession(m.truth, m.period, hold),
                             X=v3.features(a, m.period, m.pitch, m.fps)))
    console.print(f"{len(data)} matches · {data[0]['X'].shape[1]} features")

    folds = []
    with status("training 3 models per held-out match ..."):
        for k, d in enumerate(data):
            rest = [e for j, e in enumerate(data) if j != k]
            mdl = v3.fit(pd.concat([e["X"] for e in rest], ignore_index=True),
                         np.concatenate([e["y"] for e in rest]), np.concatenate([e["m"].period for e in rest]))
            if args.dead_from_v2:
                v2_cols = [c for c in rest[0]["X"].columns][:62]
                dead = v3.fit_dead_v2(pd.concat([e["X"] for e in rest], ignore_index=True),
                                      np.concatenate([e["y"] for e in rest]), v2_cols)
                folds.append(v3.proba_hybrid(mdl, dead, d["X"]))
            else:
                folds.append(v3.proba(mdl, d["X"]))

    res = {}
    with status(f"decoding {len(GRID_HA) * len(GRID_D) * len(GRID_K)} settings × {len(data)} matches ..."):
        for ha, dd, kk in itertools.product(GRID_HA, GRID_D, GRID_K):
            dec = v3.Decoder(ha, dd, kk)
            res[(ha, dd, kk)] = float(np.mean([score(d["y"], v3.decode(p, pc, d["m"].period, dec), d["m"].period)[0]
                                               for d, (p, pc) in zip(data, folds)]))
    best = max(res, key=res.get)

    section("Does the change detector help?", "Best spellF1 for each k (k = 0 means constant costs, like v2).")
    t = table("k", "best mean spellF1", "at switch-ha / switch-d")
    for kk in GRID_K:
        sub = {key: v for key, v in res.items() if key[2] == kk}
        b = max(sub, key=sub.get)
        t.add_row(str(kk), grade(sub[b], V2D_CV + 0.01, V2D_CV), f"{b[0]} / {b[1]}")
    console.print(t)

    dec = v3.Decoder(*best)
    t = table("held-out match", "spellF1 v3", "macroF1", "possessions v3 / truth")
    rows = []
    for d, (p, pc) in zip(data, folds):
        s, f1, n, ng = score(d["y"], v3.decode(p, pc, d["m"].period, dec), d["m"].period)
        rows.append((s, n / ng))
        t.add_row(d["m"].name, f"{s:.3f}", f"{f1:.3f}", f"{n} / {ng}")
    ms, mr = np.mean(rows, axis=0)
    t.add_row("[bold]mean[/bold]", grade(ms, V2D_CV + 0.01, V2D_CV), "", f"{mr:.0%} of true count")
    console.print(t)
    insight(f"v3 {ms:.3f} vs v2d {V2D_CV:.3f} on the same folds · best: switch-ha {best[0]}, switch-d {best[1]}, k {best[2]}. "
            "Adopt only if the gain is > 0.01.")

    if args.save or args.save_as:
        cfg = dict(name="v3", min_hold=hold, switch_ha=best[0], switch_d=best[1], k=best[2], copies=1, tags=True,
                   cv_spell_f1=ms)
        p = args.save_as or ch.THIS_REPO / "configs" / "v3.json"
        if args.dead_from_v2:
            cfg["note"] = "hybrid dead model in CV; at prediction time pass --dead-model or set dead_model"
        cfg["selected_on"] = "leave-one-match-out official spellF1 on degraded DFL matches only (external data)"
        p.write_text(json.dumps(cfg, indent=2))
        console.print(f"[dim]saved → {p.relative_to(ch.THIS_REPO)}[/dim]")


if __name__ == "__main__":
    main()
    save("train_v3")
