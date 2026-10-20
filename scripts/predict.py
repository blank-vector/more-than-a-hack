"""Train on all cached matches except the target, predict the target, write a CSV.

    python scripts/predict.py sample_game_3 --out outputs/sample_game_3_pred.csv

The output format (period, frame, possession in {none, home, away}) is a
placeholder until the competition publishes its submission spec.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import data, features, labels, model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("--out", type=Path)
ap.add_argument("--penalty", type=float, default=20.0)
args = ap.parse_args()

train = [n for n in data.cached_matches() if n != args.match and n.startswith("sample_")]
Xs, ys = [], []
for n in train:
    m = data.load(n)
    Xs.append(features.build(m))
    ys.append(labels.frame_labels(m.frame, m.period, m.events))
booster = model.train(pd.concat(Xs), np.concatenate(ys))

m = data.load(args.match)
pred = model.smooth(model.predict_proba(booster, features.build(m)), m.period, args.penalty)
out = args.out or data.ROOT / "outputs" / f"{args.match}_pred.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"period": m.period, "frame": m.frame,
              "possession": np.array(labels.CLASS_NAMES)[pred]}).to_csv(out, index=False)
print(f"wrote {out}")
