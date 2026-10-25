# Pipeline — data, formats and decisions

*How data flows from the raw files to the submission, what every array looks like, and why each choice was made. History: [JOURNAL.md](JOURNAL.md). Numbers: [RESULTS.md](RESULTS.md).*

## 1. The whole flow

```
CHALLENGE (Barça–Athletic)                         DFL OPEN DATA (7 Bundesliga matches)
data/tactical/*_atd.txt + .xml                     kloppy download (Hugging Face)
        │ atd.py (parse once, cache)                        │ dfl.load_match (cache)
        ▼                                                   ▼
   ATD arrays  ◄──────── same shapes ────────►   Match arrays + touch-level truth
        │                                                   │ dfl.controlled_possession (min_hold)
        │                                                   │ dfl.degrade (make it look like ATD)
        ▼                                                   ▼
   v2.features (T × 62)                            v2.features (T × 62) + labels
        │                                                   │
        │                                     v2.fit → 2 LightGBM models
        │                                                   │
        └──────────► v2.proba (T × 3: P(D), P(H), P(A)) ◄───┘
                               │ + Smart Tagging: force D before each kick-off
                               ▼
                     v2.decode: Viterbi per half + lag shift
                               ▼
               outputs/submissions/v2*.csv (199,319 rows: frame,label)
                               ▼
         validate (official) → score.py / diagnose_public.py (public 5 min)
```

v1 follows the same path but replaces the learned models with hand-set evidence (`v1.py`).

## 2. Inputs

### Challenge tracking (ATD) — the main input

| File | Format |
|---|---|
| `BAR-ATH_tactical_atd.txt` | One line per frame: `frame_id:slot;slot;…;slot:ball`. Each group is `x,y,speed`, normalised 0–1, `NaN` when missing |
| `BAR-ATH_tactical_atd.xml` | Metadata: two teams (first = home = Barcelona), 46 player "slots" each tagged with a team |

`atd.py` streams the text file once and caches arrays in `$MTAH_DATA_DIR/cache/BAR-ATH_tactical_atd.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `ball` | (199,319, 2) | Ball x, y in metres, `NaN` when missing (~41% of graded frames) |
| `xy` | (199,319, 46, 2) | Every slot's position in metres, `NaN` when the slot is empty |
| `speed` | (199,319, 46) | Tracker's speed in m/s (v1 uses it; v2 recomputes speed from positions) |
| `team` | (46,) | 0 = home (Barcelona), 1 = away (Athletic) |

**Coordinates:** `x_m = (x − 0.5) × 105.3`, `y_m = (0.5 − y) × 68.0`, giving metres with the origin at the centre spot and y pointing up.
**Frames:** row *i* of every array is video frame *i* = submission row *i* (ATD `frame_id − 1`).
**Slots are not players:** they are tracklets that appear, vanish and swap identity (median 35 s long).

### Smart Tagging — challenge only

`*_smart_tagging.xml` (SportsCode), read with the official loader into a list of 638 dicts:
`{code, start_frame, end_frame, tags: {Team, Type, Half, …}}`, with frames = round(seconds × 25).
Used for: kick-off times (everything before each half's kick-off is forced to `D`) and, in v1, phase-team hints.

### Not used (decision)

Homography and the two videos. The tracking already gives positions in pitch coordinates; using the video would mean building our own ball detector, a large project. We noted it as a possible later lever for the 41% missing ball.

### Public ground truth

`evaluation/gt/BAR-ATH_possession_gt_public.csv`: 199,319 rows `frame,label`, real labels only for frames 11,185–18,684 (first 5 min of play), `X` elsewhere. Used only for sanity checks and diagnosis.

### DFL open data — training data

Loaded with kloppy (`sportec.load_open_tracking_data`, normalised coordinates) and cached in `$MTAH_DATA_DIR/cache/dfl/<id>.npz` as a `Match`:

| Array | Shape | Meaning |
|---|---|---|
| `ball`, `xy`, `team` | as ATD, T ≈ 137–147k, ~31 player columns | Clean positions in metres, ball always present |
| `period` | (T,) | 1 or 2 |
| `truth` | (T,) | 0 = D, 1 = H, 2 = A, from DFL's BallStatus + BallPossession |

## 3. Label codes — a gotcha

Two encodings live side by side:

| | D | H | A | X |
|---|---|---|---|---|
| **Ours** (v1, v2, dfl) | 0 | 1 | 2 | — |
| **Official** (`possession_metrics`) | 2 | 0 | 1 | 3 |

`v2.to_official()` converts ours to official and puts `X` on each period's first frame, so the official scorer treats DFL periods as separate halves.

## 4. Steps and decisions

### Step 0 — Ground rules
| Decision | Why | Rejected alternative |
|---|---|---|
| Import the official `games.py` and scorer, don't copy them | Score exactly like the organisers, pick up their fixes | Re-implementing spellF1 (risk of silent differences) |
| Match data outside the repo; Claude Code denied read access; share aggregates only | Competition rule: no data to hosted AI tools, no redistribution | — |
| Public window for correctness only | 17 possessions; organisers show it picks the wrong settings | Tuning on it |

### Step 1 — Profiling (`profile_atd.py`, `profile_gaps.py`)
Measured how the ball goes missing (long blocks, mostly during dead ball), track quality (35 s fragments), and what dead ball looks like (slow players, ball outside lines, ball reappears ~11 m away). **Decision:** split the problem into "is play dead?" and "who has it?".

### Step 2 — v1: hand-set evidence + decoder (`v1.py`, `predict_v1.py`)
Per frame, two numbers:
- **dead-ball logit**: bias + slow players + ball missing + ball outside lines + few players tracked + set-piece window + ball standing still − ball in play − ball moving
- **team logit**: ± strong if a player is within 2 m of the visible ball, ± weak for the nearest team otherwise, ± a Smart Tagging phase nudge (sign learned per tag code from agreement with the nearest player)

These become log-probabilities for D, H, A (T × 3). **Viterbi** then picks the single best label sequence, paying a cost for each change of state. That is what prevents flicker.

| Decision | Why |
|---|---|
| Tune with targets that use no answers (4.2 poss/live-min, 45% dead, ~105 stoppages) | Honest tuning without the hidden truth |
| Scale both switch costs together ("stickiness") | The decoder was dodging costly team switches via a fake brief `D` |
| Separate stoppage cost (`d-cost-ratio`) | Fake short stoppages split possessions |

### Step 3 — DFL as training data (`dfl.py`, `prepare_dfl.py`)
| Decision | Why |
|---|---|
| **Translate labels** with a minimum hold: a team run shorter than `min_hold` inside live play goes back to the previous team | DFL counts touches (6.5–11 poss/live-min); the challenge counts control (~4.2) |
| **Degrade** each match: ball gaps tied to stoppages, calibrated to P(missing \| dead) ≈ 64% and P(missing \| live) ≈ 22%; tracks cut into ~35 s fragments; jitter | Train and validate on data that looks like the real input |
| Recompute speed from positions for both datasets | So "speed" means the same thing on both sides |

### Step 4 — v2: learned models (`v2.py`)
**Features** (T × 62, identical code for both datasets):
- ball: visible, outside lines, speed, standing still, distance to touchline and goal line, position relative to home's attacking direction
- nearest players: nearest home and away distances (1st, 2nd), their difference, players within 5 m
- team shape per side: players tracked, centroid along the attacking direction, front-most player, spread, mean speed, forward velocity
- time since / until: ball seen, ball out, home close to the ball, away close to the ball (capped at 10 s)
- rolling windows of 1, 3 and 10 s (centred) of speed, ball visibility, "home/away close", distance difference, centroid difference

**Attacking direction** is inferred per period: a team stands on average in its own half. This makes features mean the same in both halves and both datasets.

**Models** (LightGBM, every 5th frame since neighbours are near-duplicates):
- dead vs alive, on all frames
- home vs away, on live frames only

Combined: P(D) = p_dead, P(H) = (1 − p_dead) × p_home, P(A) = (1 − p_dead) × (1 − p_home).

**Decoder:** Viterbi with two costs (H↔A switch, entering/leaving D), then an optional **lag shift** that moves every change earlier.

### Step 5 — Validation and tuning (`train_v2.py`, `sweep_hold.py`)
| Decision | Why |
|---|---|
| **Leave-one-match-out on DFL** with the official spellF1 | ~1,450 scored possessions instead of 17; the first comparison we can trust |
| Score v1 rules on the same matches | Like-for-like baseline (0.429 vs v2 0.723) |
| Choose `min_hold` by the challenge prediction's possession rate, not by CV | Changing the hold changes the answer key, so CV scores at different holds aren't comparable |
| Tune switch costs and lag shift by CV spellF1 | Measured on held-out matches |

## 5. Output

`outputs/submissions/<name>.csv`:
```
frame,label
0,X
…
11185,A
…
199318,X
```
Exactly 199,319 rows, `frame` 0…199,318 in order, labels `H`/`A`/`D` on graded frames (11,185–90,651 and 110,808–190,197) and `X` elsewhere. Every script validates the file with the official `validate_submission` before anything else.

## 6. Scripts at a glance

| Script | Reads | Writes | Shares |
|---|---|---|---|
| `profile_atd.py`, `profile_gaps.py` | ATD, public GT | — | aggregates |
| `check_dfl.py`, `prepare_dfl.py` | DFL | DFL cache, `min_hold.txt` | DFL stats |
| `predict_v1.py` | ATD, tags | `submissions/v1.csv` | aggregates |
| `train_v2.py` | DFL | `outputs/v2_config.json` | CV scores |
| `sweep_hold.py` | DFL, ATD, tags | `outputs/v2_config.json` | CV + challenge aggregates |
| `predict_v2.py` | DFL, ATD, tags, config | `submissions/<name>.csv` | aggregates |
| `score.py` | submissions, public GT | — | public scores |
| `diagnose_public.py` | one submission, public GT | — | labels on the public 5 min |

Every script saves a plain-text copy of its output to `outputs/<script>.txt`.
