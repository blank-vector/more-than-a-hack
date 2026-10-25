"""v2 stage 3: train on ALL DFL matches, predict the challenge match, write the submission.

    python scripts/predict_v2.py            # uses outputs/v2_config.json from train_v2.py
    python scripts/predict_v2.py --name v2b --switch-ha 160
    python scripts/predict_v2.py --config configs/v2d.json --models models/v2d --name v2d   # frozen, no training
    python scripts/predict_v2.py --config configs/v2d.json --save-models models/v2d          # (re)train and store models

Prints only aggregates (safe to share). Then: python scripts/score.py && python scripts/diagnose_public.py outputs/submissions/v2.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import atd as atd_mod  # noqa: E402
from possession import challenge as ch  # noqa: E402
from possession import dfl, v1, v2  # noqa: E402
from possession.report import bar, console, insight, save, section, table, title, status  # noqa: E402


def runs_of(labels, which):
    m = np.concatenate(([False], labels == which, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2")
    ap.add_argument("--switch-ha", type=float)
    ap.add_argument("--switch-d", type=float)
    ap.add_argument("--min-hold", type=float, help="seconds")
    ap.add_argument("--shift", type=float, help="seconds to move predicted changes earlier")
    ap.add_argument("--copies", type=int, help="degraded copies of each DFL match for training")
    ap.add_argument("--no-tags", action="store_true", help="skip the Smart Tagging restart prior")
    ap.add_argument("--config", type=Path, help="settings file (default: outputs/v2_config.json)")
    ap.add_argument("--models", type=Path, help="load trained models from this folder instead of training")
    ap.add_argument("--save-models", type=Path, help="after training, store the models in this folder")
    args = ap.parse_args()

    cfg_path = args.config or ch.THIS_REPO / "outputs" / "v2_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else dict(min_hold=100, switch_ha=80, switch_d=80)
    hold = int(args.min_hold * 25) if args.min_hold is not None else cfg["min_hold"]
    shift = int(round(args.shift * 25)) if args.shift is not None else int(cfg.get("shift", 0))
    dec = v2.Decoder(switch_ha=args.switch_ha or cfg["switch_ha"], switch_d=args.switch_d or cfg["switch_d"], shift=shift)
    copies = args.copies or int(cfg.get("copies", 1))
    use_tags = not (args.no_tags or cfg.get("tags") is False)
    title(f"v2 submission · {args.name}" + (" · + Smart Tagging restarts" if use_tags else "") + f" · {copies} copies",
          f"min_hold {hold / 25:.1f} s · switch-ha {dec.switch_ha:g} · switch-d {dec.switch_d:g} · shift {dec.shift / 25:.2f} s")

    if args.models:
        models = v2.load_models(args.models)
        console.print(f"[dim]loaded trained models from {args.models}[/dim]")
    else:
        with status("training on all DFL matches ..."):
            Xs, ys = [], []
            for i, mid in enumerate(dfl.MATCHES):
                try:
                    m = dfl.load_match(mid)
                except Exception as e:
                    console.print(f"[red]✗[/red] {mid}: {e}")
                    continue
                y = dfl.controlled_possession(m.truth, m.period, hold)
                for c in range(copies):
                    a = dfl.degrade(m, seed=100 + i + 1000 * c)
                    Xs.append(v2.features(a, m.period, m.pitch, m.fps))
                    ys.append(y)
            models = v2.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
        if args.save_models:
            v2.save_models(models, args.save_models)
            console.print(f"[dim]saved trained models to {args.save_models}[/dim]")

    with status("predicting the challenge match ..."):
        a = atd_mod.load()
        halves = ch.half_index(a.n_frames)
        X = v2.features(a, halves, ch.PITCH, ch.FPS)
        p = v2.proba(models, X)
        # before each half's kick-off (from Smart Tagging) the ball is always dead
        tags = v1.load_tags()
        ev = v1.build_evidence(a, tags, v1.Params())
        tag_report = None
        if use_tags:
            p, tag_report = v2.apply_tag_prior(p, tags, v2.TagPrior())
        p[ev.force_d] = [1.0, 0.0, 0.0]
        labels = np.full(a.n_frames, "X", dtype="<U1")
        graded = halves > 0
        labels[graded] = v2.LETTERS[v2.decode(p[graded], halves[graded], dec)]

    out = ch.THIS_REPO / "outputs" / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.csv"
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)
    why = ch.pm.validate_submission(ch.pm.read_label_csv(str(path)), ch.N_FRAMES)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    section("Validation")
    console.print("[bold green]✓ valid submission[/bold green]" if why is None else f"[bold red]✗ {why}[/bold red]")
    console.print(f"[dim]{path}[/dim]")
    console.print(f"sha256 [bold]{digest[:16]}[/bold]…  (identical runs must print the same value)")

    section("What it predicts (full match)")
    g = labels[graded]
    t = table("label", "share of graded frames")
    for L, st in (("H", "red"), ("A", "blue"), ("D", "yellow")):
        t.add_row(f"[{st}]{L}[/{st}]", bar(np.mean(g == L), 20, st))
    console.print(t)
    poss = np.concatenate([runs_of(labels[halves == h], L) for h in (1, 2) for L in "HA"]) / ch.FPS
    dead = np.concatenate([runs_of(labels[halves == h], "D") for h in (1, 2)]) / ch.FPS
    live_min = np.isin(g, ["H", "A"]).sum() / ch.FPS / 60
    t = table("", "count", "mean s", "median s", "target")
    t.add_row("possessions", str(len(poss)), f"{poss.mean():.1f}", f"{np.median(poss):.1f}",
              f"{len(poss) / live_min:.1f} per live min (≈ 4.2)")
    t.add_row("dead-ball stretches", str(len(dead)), f"{dead.mean():.1f}", f"{np.median(dead):.1f}", "≈ 105")
    console.print(t)
    console.print(f"dead-ball share: [bold]{np.mean(g == 'D'):.0%}[/bold] (≈ 45%)   ·   "
                  f"Barcelona share of live play: [bold]{np.mean(g == 'H') / max(np.mean(np.isin(g, ['H', 'A'])), 1e-9):.0%}[/bold]")
    codes = np.select([labels == "D", labels == "H", labels == "A"], [v2.D, v2.H, v2.A], -1)
    cov = v2.tag_coverage(codes, tags)
    console.print(f"set-piece tags preceded by predicted dead ball: [bold]{cov:.0%}[/bold]"
                  + (f"   ·   prior applied to {tag_report['set_pieces']} set pieces, {tag_report['goals']} goals"
                     if tag_report else "   ·   (no tag prior)"))
    insight(f"next: python scripts/score.py  ·  python scripts/diagnose_public.py {path.relative_to(ch.THIS_REPO)}")


if __name__ == "__main__":
    main()
    save("predict_v2")
