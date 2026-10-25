# More Than a Hack 2027 — frame-level possession

Our pipeline for [Phase 1 of the challenge](https://github.com/metrica-sports/more-than-a-hack-2027): label every frame of FC Barcelona vs Athletic Club as `H` / `A` / `D` / `X`, ranked on **spellF1**.

**Write-up:** [docs/RESULTS.md](docs/RESULTS.md) (current state) · [docs/JOURNAL.md](docs/JOURNAL.md) (how we got here) · [docs/PIPELINE.md](docs/PIPELINE.md) (data formats and design)

## Layout on disk

```
~/code/mtah/
├─ more-than-a-hack-2027/   official repo: games.py, scorer, baseline  (MTAH_REPO)
├─ more-than-a-hack/        this repo
└─ data/                    match bundle + our cache                   (MTAH_DATA_DIR)
```

**Match data never goes in this repo**, and must never be shown to a hosted AI tool. That's a competition rule. `.claude/settings.json` denies Claude Code read access to it.

## Challenge scripts

```bash
python scripts/score.py         # readable scoreboard on the public 5 minutes (official metric)
python scripts/profile_atd.py   # aggregate stats of the real tracking (safe to share)
python scripts/profile_gaps.py  # what happens while the ball is missing
python scripts/check_dfl.py     # verify open DFL data as training labels (public data)
python scripts/predict_v1.py --sweep   # v1: pick stickiness by possession rate (no ground truth)
python scripts/predict_v1.py --stickiness S   # v1: write outputs/submissions/v1.csv
python scripts/diagnose_public.py            # truth vs prediction timeline on the public 5 min
python scripts/prepare_dfl.py                # v2 stage 1: DFL training data (public)
python scripts/train_v2.py                   # v2 stage 2: leave-one-match-out validation on DFL, tunes the decoder
python scripts/predict_v2.py                 # v2 stage 3: train on all DFL, write outputs/submissions/v2.csv
python scripts/sweep_hold.py                 # v2 round 2: choose label hold, decoder costs and lag shift
```

`src/possession/challenge.py` imports the official `games.py` and `possession_metrics.py` from the official repo, so we always score exactly like the organisers. `src/possession/atd.py` parses the ATD once into a cache in `$MTAH_DATA_DIR/cache`.

## Submission log

| Version | Command | Full-match sanity (rate / dead / stoppages) | Public window (sanity only) |
|---|---|---|---|
| v1.1 | `python scripts/predict_v1.py --d-bias -2.0 --stickiness 2 --d-cost-ratio 4` | 3.6 / 44% / 107 | spellF1 0.400 · macroF1 0.700 · 13 vs 17 possessions |

---

## Prep work on Metrica's public sample (before the brief)

Starter pipeline for the Initial Challenge (20 Oct – 17 Nov 2026): estimate which team has possession in every frame of a match, using automatic tracking data and a homography.

It's built and validated on Metrica's three public sample matches. The competition data should use a similar format, so on day one only a loader in `src/possession/data.py` needs adapting.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # also installs src/possession in editable mode
python scripts/prepare.py        # clones metrica-sports/sample-data into data/ and caches the 3 matches (~1 min)
```

**Windows (PowerShell)**, one line at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt   # also installs src/possession in editable mode
python scripts\prepare.py
```

Then open `notebooks/` in VS Code, choosing the `.venv` kernel.

### Layout

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

### Results so far (leave-one-match-out, 3 matches)

| | Clean sample ball | Realistic ball¹ |
|---|---|---|
| Baseline (r = 1 m, 3-frame hysteresis) | 0.839 | 0.703 |
| LightGBM | 0.859 | 0.836 |
| **LightGBM + Viterbi (penalty 20)** | **0.864** | **0.845** |

Figures are frame accuracy. The smoothed model also predicts about the right number of possession changes (~420 vs ~434 true per match); the raw model predicts 4–5× too many.

¹ Ball with 0.7 m of smooth positional noise, 10% of frames dropped in bursts, and the NaN-during-dead-ball signal removed.

### Caveats

- **The labels are a proxy.** They come from Metrica's event data using our own rules (see `labels.py`). Once the official definition of possession is published (dead ball? contested? loose ball?), update `labels.py` first.
- **The sample ball is too clean.** In this public data, the ball sits within 0.1 m of the passer and is NaN whenever play is dead. Judge methods on the realistic scenario, not the clean one.
- **Features use future frames** (centred windows, "time until next close"). That's fine if submissions score whole matches offline. If the challenge requires causal predictions, drop those features.
- **The submission format is a placeholder** (`period, frame, possession`) until the spec is out.

### Ideas to try next

- Pitch control / probabilistic ownership instead of hard nearest-distance
- Better handling of ball height and aerial balls (if `ball_z` is provided)
- Train on the real competition matches with their labels; tune `degrade_ball` to match their noise
- Sequence models (1D CNN / GRU over a ±2 s window) in place of handcrafted rolling features
- Homography pipeline, if tracking arrives in image coordinates: `cv2.perspectiveTransform`
