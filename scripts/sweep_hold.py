"""v2 round 2: choose the label hold, tune the decoder and correct its lag. Aggregates only.

    python scripts/sweep_hold.py                    # holds 2, 3, 4 s (~20-40 min)
    python scripts/sweep_hold.py --holds 2.5 3

For each hold (how long a team must keep the ball before DFL labels count it as a possession):
  1. leave-one-match-out on DFL: tune switch costs AND a lag shift by official spellF1
  2. measure how late our changes are vs the truth (before and after the shift)
  3. train on all DFL and predict the challenge match; report its possession rate

CV scores at different holds are graded against DIFFERENT labels, so they are not comparable.
The hold is chosen by the challenge prediction's possession rate (target ≈ 4.2 per live minute);
the decoder settings for that hold come from its own CV. The winner is saved to
outputs/v2_config.json for predict_v2.py.
"""
import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rich.text import Text

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402
from possession import dfl, v1, v2  # noqa: E402
from possession.report import console, insight, save, section, table, title, status  # noqa: E402

pm = ch.pm
GRID_HA, GRID_D, SHIFTS = (20, 40, 80), (20, 40, 80), (0, 6, 12, 18, 25)
TARGET_RATE = 4.2


def spell(truth, pred, period):
    r = pm.evaluate(v2.to_official(truth, period), v2.to_official(pred, period))
    return r["spell_f1_mean"]


def challenge_stats(labels, halves):
    g = labels[halves > 0]
    n_poss = n_dead = 0
    for h in (1, 2):
        seg = labels[halves == h]
        for s, e, v in dfl.runs(seg):
            if v in (v2.H, v2.A):
                n_poss += 1
            elif v == v2.D:
                n_dead += 1
    live_min = np.isin(g, [v2.H, v2.A]).sum() / ch.FPS / 60
    return n_poss / max(live_min, 1e-9), n_dead, float(np.mean(g == v2.D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", type=float, nargs="+", default=[2, 3, 4])
    args = ap.parse_args()
    title("v2 · round 2 · label hold × decoder × lag", "public DFL data + challenge aggregates · safe to share")

    with status("loading and degrading DFL ..."):
        base = []
        for i, mid in enumerate(dfl.MATCHES):
            try:
                m = dfl.load_match(mid)
            except Exception as e:
                console.print(f"[red]✗[/red] {mid}: {e}")
                continue
            a = dfl.degrade(m, seed=100 + i)
            base.append(dict(m=m, X=v2.features(a, m.period, m.pitch, m.fps)))
    with status("challenge features ..."):
        ca = atd_mod.load()
        halves = ch.half_index(ca.n_frames)
        cX = v2.features(ca, halves, ch.PITCH, ch.FPS)
        force_d = v1.build_evidence(ca, v1.load_tags(), v1.Params()).force_d
    console.print(f"{len(base)} DFL matches · challenge {ca.n_frames:,} frames")

    results = []
    for hold_s in args.holds:
        hold = int(round(hold_s * 25))
        section(f"Hold {hold_s:g} s")
        data = [dict(**b, y=dfl.controlled_possession(b["m"].truth, b["m"].period, hold)) for b in base]
        rate_dfl = np.mean([dfl.possession_rate(d["y"], d["m"].period, d["m"].fps)[0] for d in data])

        with status("cross-validation: training 1 model per held-out match ..."):
            probs = []
            for k, d in enumerate(data):
                rest = [e for j, e in enumerate(data) if j != k]
                mdl = v2.fit(pd.concat([e["X"] for e in rest], ignore_index=True), np.concatenate([e["y"] for e in rest]))
                probs.append(v2.proba(mdl, d["X"]))

        best = None
        with status("tuning switch costs and lag shift ..."):
            for ha, dd in itertools.product(GRID_HA, GRID_D):
                raw = [v2.decode(p, d["m"].period, v2.Decoder(ha, dd)) for d, p in zip(data, probs)]
                for k in SHIFTS:
                    sc = np.mean([spell(d["y"], v2.shift_earlier(r, d["m"].period, k), d["m"].period)
                                  for d, r in zip(data, raw)])
                    if best is None or sc > best[0]:
                        best = (sc, ha, dd, k)
        sc, ha, dd, k = best
        raw = [v2.decode(p, d["m"].period, v2.Decoder(ha, dd)) for d, p in zip(data, probs)]
        lag0 = np.concatenate([v2.change_lags(d["y"], r, d["m"].period) for d, r in zip(data, raw)])
        lag1 = np.concatenate([v2.change_lags(d["y"], v2.shift_earlier(r, d["m"].period, k), d["m"].period)
                               for d, r in zip(data, raw)])

        with status("training on all DFL and predicting the challenge ..."):
            mdl = v2.fit(pd.concat([d["X"] for d in data], ignore_index=True), np.concatenate([d["y"] for d in data]))
            p = v2.proba(mdl, cX)
            p[force_d] = [1.0, 0.0, 0.0]
            lab = np.full(ca.n_frames, -1, np.int8)
            g = halves > 0
            lab[g] = v2.decode(p[g], halves[g], v2.Decoder(ha, dd, shift=k))
            rate, stops, dead = challenge_stats(lab, halves)

        console.print(f"DFL labels at this hold: {rate_dfl:.1f} possessions per live minute")
        console.print(f"CV spellF1 [bold]{sc:.3f}[/bold] with switch-ha {ha}, switch-d {dd}, shift {k / 25:.2f} s")
        console.print(f"median lag of our changes: {np.median(lag0) / 25:+.2f} s → {np.median(lag1) / 25:+.2f} s after the shift")
        console.print(f"challenge: [bold]{rate:.1f}[/bold] possessions per live minute · {stops} stoppages · {dead:.0%} dead")
        results.append(dict(hold=hold, hold_s=hold_s, cv=sc, ha=ha, dd=dd, shift=k, lag0=float(np.median(lag0) / 25),
                            lag1=float(np.median(lag1) / 25), rate=rate, stops=stops, dead=dead))

    section("Summary", f"Choose the hold whose CHALLENGE rate is nearest {TARGET_RATE}. CV columns are per-hold, not comparable.")
    pick = min(results, key=lambda r: abs(np.log(r["rate"] / TARGET_RATE)))
    t = table("hold", "challenge poss/live-min", "stoppages", "dead", "CV spellF1 (own labels)", "decoder", "lag before → after")
    for r in results:
        style = "bold green" if r is pick else ""
        t.add_row(Text(f"{r['hold_s']:g} s" + ("  ← chosen" if r is pick else ""), style=style),
                  Text(f"{r['rate']:.1f}", style=style), str(r["stops"]), f"{r['dead']:.0%}", f"{r['cv']:.3f}",
                  f"{r['ha']}/{r['dd']} · shift {r['shift'] / 25:.2f} s", f"{r['lag0']:+.2f} → {r['lag1']:+.2f} s")
    console.print(t)
    cfg = dict(min_hold=pick["hold"], switch_ha=pick["ha"], switch_d=pick["dd"], shift=pick["shift"],
               cv_spell_f1=pick["cv"], challenge_rate=pick["rate"])
    (ch.THIS_REPO / "outputs" / "v2_config.json").write_text(json.dumps(cfg, indent=2))
    insight(f"saved hold {pick['hold_s']:g} s, switch-ha {pick['ha']}, switch-d {pick['dd']}, shift {pick['shift'] / 25:.2f} s "
            "→ outputs/v2_config.json.  Next: python scripts/predict_v2.py --name v2b")


if __name__ == "__main__":
    main()
    save("sweep_hold")
