# More Than a Hack 2027 — frame-level possession

Starter pipeline for the Initial Challenge (20 Oct – 17 Nov 2026): estimate which team has possession in every frame of a match, using automatic tracking data and a homography.

It's built and validated on Metrica's three public sample matches. The competition data should use a similar format, so on day one only a loader in `src/possession/data.py` needs adapting.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare.py        # clones metrica-sports/sample-data into data/ and caches the 3 matches (~1 min)
```

**Windows (PowerShell)**, one line at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python scripts\prepare.py
```

Then open `notebooks/` in VS Code, choosing the `.venv` kernel.

## Layout

| Path | What |
|---|---|
| `src/possession/data.py` | Loads tracking and events into one canonical `Match` (metres, centre origin), cached as `.npz` |
| `src/possession/labels.py` | Proxy ground truth from events: `none / home / away` per frame |
| `src/possession/baseline.py` | Nearest player within *r* m, plus hysteresis |
| `src/possession/features.py` | ~45 per-frame features: distances, counts near the ball, ball speed and acceleration, time since/until each team was close, rolling windows |
| `src/possession/model.py` | LightGBM 3-class classifier with Viterbi smoothing (a penalty per possession change) |
| `src/possession/evaluate.py` | Metrics, plus `degrade_ball` to stress-test with realistic ball noise |
| `scripts/cv.py` | Leave-one-match-out comparison of all methods |
| `scripts/predict.py` | Train, predict one match, and write a CSV |
| `notebooks/01_explore_data.ipynb` | Pitch plots, label timeline, distance traces |
| `notebooks/02_baseline_vs_model.ipynb` | Interactive version of the CV, with confusion matrix, feature importance and an error viewer |

## Results so far (leave-one-match-out, 3 matches)

| | Clean sample ball | Realistic ball¹ |
|---|---|---|
| Baseline (r = 1 m, 3-frame hysteresis) | 0.839 | 0.703 |
| LightGBM | 0.859 | 0.836 |
| **LightGBM + Viterbi (penalty 20)** | **0.864** | **0.845** |

Figures are frame accuracy. The smoothed model also predicts about the right number of possession changes (~420 vs ~434 true per match); the raw model predicts 4–5× too many.

¹ Ball with 0.7 m of smooth positional noise, 10% of frames dropped in bursts, and the NaN-during-dead-ball signal removed.

## Caveats

- **The labels are a proxy.** They come from Metrica's event data using our own rules (see `labels.py`). Once the official definition of possession is published (dead ball? contested? loose ball?), update `labels.py` first.
- **The sample ball is too clean.** In this public data, the ball sits within 0.1 m of the passer and is NaN whenever play is dead. Judge methods on the realistic scenario, not the clean one.
- **Features use future frames** (centred windows, "time until next close"). That's fine if submissions score whole matches offline. If the challenge requires causal predictions, drop those features.
- **The submission format is a placeholder** (`period, frame, possession`) until the spec is out.

## Ideas to try next

- Pitch control / probabilistic ownership instead of hard nearest-distance
- Better handling of ball height and aerial balls (if `ball_z` is provided)
- Train on the real competition matches with their labels; tune `degrade_ball` to match their noise
- Sequence models (1D CNN / GRU over a ±2 s window) in place of handcrafted rolling features
- Homography pipeline, if tracking arrives in image coordinates: `cv2.perspectiveTransform`
