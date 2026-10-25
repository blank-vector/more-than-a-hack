# Results — current state

*More Than a Hack 2027, Phase 1: frame-level ball possession on FC Barcelona vs Athletic Club.*
*Last updated: 25 Oct 2026. The story of how we got here is in [JOURNAL.md](JOURNAL.md).*

## The task in one paragraph

Label each of the 199,319 frames of the tactical video `H` (Barcelona has the ball), `A` (Athletic), `D` (dead ball) or `X` (abstain). Only the two halves are graded (158,857 frames). The ranking metric is **spellF1**: possessions are matched as whole intervals by temporal IoU (0.3–0.7), so flicker and missed short possessions are punished hard; macroF1 is the tie-break. Published floor: **spellF1 0.2433** on the full match. Deadline: **17 Nov 2026, 23:59 CET**. Team names must be registered by **10 Nov**.

## What the data looks like

Measured by `scripts/profile_atd.py` and `scripts/profile_gaps.py` (aggregates only).

| Fact | Value | Consequence |
|---|---|---|
| Ball visible, graded frames | 59.3% | The ball is the main signal and is often missing |
| Ball gaps | 150 gaps, median 10.7 s, 44% of missing time in gaps > 30 s | Interpolation can't fill them |
| Players slow (< 1.5 m/s) in long gaps | 80.5% of frames (vs 27.8% when the ball is visible) | **Most missing ball is dead ball** |
| Long gaps that start at or outside a line | ~30% | Out of play → throw-in, corner, goal kick |
| Ball moves between vanishing and reappearing | median 11.4 m | Restarts happen elsewhere |
| Player track fragments | median 35.5 s, p90 4.7 min | Per-track features are usable |
| Players tracked per team | median 11, p5 6–7 | Fewer during stoppages (19.2 vs 21.4 total) |
| Mean player speed by true label (public 5 min) | H 2.21 · A 2.61 · **D 1.42** m/s | Speed < 1.5 m/s catches 78% of D at 15% false alarms |
| Public window ball visibility | H 99.9% · A 85% · D 66% | **The public 5 min are much easier than the rest** |

### External training data: DFL open data (Bundesliga, public)

`scripts/check_dfl.py`: 6 matches loaded, per-frame alive/dead and owning team on **100%** of frames.

| | Challenge (public 5 min) | DFL (6 matches) |
|---|---|---|
| Possessions per live minute | ~4.2 | 7.1–11.0 |
| Mean possession | 14.2 s | 5.5–8.4 s |
| Dead-ball share | 20% (opening 5 min only) | 41–52% |
| Dead-ball stretches per match | — | 97–113, mean 23–27 s |

DFL counts possession on any touch; the challenge counts *controlled* possession. DFL labels need translating before training (see v2).

## Submissions

All public-window numbers are **sanity checks only**: 17 possessions, ~0.06 spellF1 per possession.

| Version | Command | Full match: poss/live-min · dead share · stoppages | Public: spellF1 · macroF1 · possessions (truth 17) |
|---|---|---|---|
| baseline_smooth50 (official) | `python starter/baseline.py` | — | 0.326 · 0.621 · 26 |
| v1 | `predict_v1.py --d-bias -1.0 --stickiness 2` | 4.2 · 47% · 171 | 0.315 · 0.680 · 16 |
| v1.1 | `predict_v1.py --d-bias -2.0 --stickiness 2 --d-cost-ratio 4` | 3.6 · 44% · 107 | 0.400 · 0.700 · 13 |
| v2 | `train_v2.py` then `predict_v2.py` (min_hold 4.0 s, switch-ha 40, switch-d 40) | 3.2 · 41% · 106 | 0.571 · 0.759 · 11 |
| **v2b (current)** | `sweep_hold.py` then `predict_v2.py --name v2b` (min_hold 2.0 s, switch-ha 20, switch-d 40, no shift) | 3.8 · 41% · 103 | **0.600 · 0.764 · 11** |

### Round 2: label hold (each hold has its own CV labels, so CV is not comparable across rows)

| Hold | DFL label rate | CV spellF1 | Decoder | Median lag on DFL | Challenge poss/live-min |
|---|---|---|---|---|---|
| **2 s (chosen)** | 5.5 | 0.656 | 20 / 40 | 0.00 s | **3.8** |
| 3 s | 4.6 | 0.699 | 20 / 40 | 0.00 s | 3.5 |
| 4 s | 4.0 | 0.723 | 40 / 40 | 0.00 s | 3.2 |

**Training copies (hold 2 s):** 1 copy 0.657 → **3 copies 0.671** mean CV spellF1 (v1 rules at this hold: 0.373). Adopted.

No lag on DFL, so no shift; the +0.6 s seen on the public window rests on 10 matched changes.

### Honest validation: leave-one-match-out on 7 degraded DFL matches (official spellF1, ~207 possessions each)

| Held-out match | v2 | v1 rules |
|---|---|---|
| J03WMX | 0.708 | 0.461 |
| J03WN1 | 0.708 | 0.386 |
| J03WOH | 0.708 | 0.381 |
| J03WOY | 0.669 | 0.459 |
| J03WPY | 0.768 | 0.433 |
| J03WQQ | 0.759 | 0.448 |
| J03WR9 | 0.743 | 0.434 |
| **Mean** | **0.723** | **0.429** |

v2 wins on every match. Absolute levels are optimistic (synthetic noise, translated labels); the gap between methods is the reliable part. Best decoder setting (40, 40) sits inside the grid, on a smooth plateau.

Full-match targets (no ground truth used): ~4.2 possessions per live minute, ~45% dead time, ~105 stoppages.

## Where v2 still fails (public window, `scripts/diagnose_public.py`)

| Failure mode | Segments | Status vs v1.1 |
|---|---|---|
| Short possessions (0.8–3.5 s) | 3, 11, 13, 14, 17 missed · 8 partial (0.49) | Better: 8 no longer merges 7 and 9 (now 0.80 and 0.95) |
| Stoppages with the ball carried | 15, 20 | Still missed |
| Short real stoppage | 6 | Still missed |
| Boundary timing | — | Slightly late: median +1.0 s |
| Under-segmenting | 11 vs 17 possessions; 3.2 per live minute on the full match | New focus |

Verdicts for v2: 9 match · 3 partial · 9 missed out of 21 segments.

## Where v1.1 failed (kept for reference)

| Failure mode | Segments | Status |
|---|---|---|
| Short opposing possessions (0.8–3.5 s) swallowed; each miss also merges its neighbours | 3, 8, 11, 13, 17 | Open → v2 |
| Stoppages with the ball carried to the spot look like possession | 15, 20 | Open → v2 |
| False short stoppages splitting a possession | 16 | **Fixed in v1.1** (IoU 0.30 → 0.76) |
| A real 5.5 s stoppage removed by the higher stoppage cost | 6 | New in v1.1 → v2 |
| Boundary timing | all | OK (median lag −0.5 s) |

Verdicts for v1.1: 8 match · 4 partial · 9 missed out of 21 segments.

## In progress: version 2 (learned from DFL)

| Step | What | Status |
|---|---|---|
| 1 | Load DFL, translate touch-level → controlled possession, degrade to challenge-like tracking | **Done** · all 7 matches · `min_hold` 4.0 s → 3.9 poss/live-min · degraded ball visible 52–63% (challenge 59%) |
| 2 | 62 shared features (ball, nearest players, team shape with inferred attacking direction, time since/until events, rolling windows) | **Built** · `src/possession/v2.py` |
| 3 | LightGBM: dead vs alive, and home vs away on live frames | **Built** |
| 4 | Decoder costs tuned by **official spellF1, leave-one-match-out on DFL**; v1 rules scored on the same matches for reference | **Built** · run `scripts/train_v2.py` |
| 5 | Train on all DFL, predict the challenge → `outputs/submissions/v2.csv` | **Built** · run `scripts/predict_v2.py` |

## Open questions and risks

- **Team**: solo for now; teams of < 3 can't advance past Phase 1. Registration closes 10 Nov.
- **Label mismatch**: DFL's possession definition differs from the challenge's; the `min_hold` translation is an approximation.
- **Domain gap**: degraded DFL (Bundesliga 2, synthetic noise) may not behave like the real Metrica tracking.
- **The public window is unrepresentative** (ball almost always visible), so it can't validate behaviour in long gaps.
