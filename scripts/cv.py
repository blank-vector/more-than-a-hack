"""Leave-one-match-out comparison: baseline vs LightGBM vs LightGBM + Viterbi.

    python scripts/cv.py

Each scenario is evaluated on a clean and a "realistic" (noisy, gappy, no
dead-ball NaN leak) version of the ball. The model is trained on both versions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import baseline, data, evaluate as ev, features, labels, model  # noqa: E402

SCENARIOS = {"clean": {}, "realistic": dict(noise_m=0.7, dropout=0.1, fill_dead=True)}
PENALTIES = [2, 5, 10, 20]

names = data.cached_matches()
M = {n: data.load(n) for n in names}
Y = {n: labels.frame_labels(m.frame, m.period, m.events) for n, m in M.items()}
balls = {(n, s): (ev.degrade_ball(M[n].ball, **kw, seed=i) if kw else M[n].ball)
         for i, n in enumerate(names) for s, kw in SCENARIOS.items()}
X = {k: features.build(M[k[0]], b) for k, b in balls.items()}

rows = []
for test in names:
    train = [n for n in names if n != test]
    booster = model.train(pd.concat([X[(n, s)] for n in train for s in SCENARIOS]),
                          np.concatenate([Y[n] for n in train for s in SCENARIOS]))
    for s in SCENARIOS:
        m, y = M[test], Y[test]
        res = {"baseline": baseline.predict(m, 1.0, 3, ball=balls[(test, s)])}
        proba = model.predict_proba(booster, X[(test, s)])
        res["lgbm"] = proba.argmax(1)
        for pen in PENALTIES:
            res[f"lgbm+viterbi({pen})"] = model.smooth(proba, m.period, pen)
        for method, pred in res.items():
            rows.append(dict(test=test, scenario=s, method=method, **ev.score(y, pred)))
    print(f"done {test}", flush=True)

df = pd.DataFrame(rows)
summary = df.groupby(["scenario", "method"])[["acc", "macro_f1", "acc_transitions", "acc_steady", "switches_pred"]].mean()
summary["switches_true"] = df.groupby(["scenario", "method"]).switches_true.mean()
pd.set_option("display.width", 160)
print(summary.round(3))
out = data.ROOT / "outputs"
out.mkdir(exist_ok=True)
df.to_csv(out / "cv_results.csv", index=False)
