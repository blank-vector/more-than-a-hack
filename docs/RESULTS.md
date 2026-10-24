# Results — current state

*More Than a Hack 2027, Phase 1: frame-level ball possession on FC Barcelona vs Athletic Club.*
*Last updated: 24 Oct 2026. The story of how we got here is in [JOURNAL.md](JOURNAL.md).*

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
| **v1.1 (current)** | `predict_v1.py --d-bias -2.0 --stickiness 2 --d-cost-ratio 4` | 3.6 · 44% · 107 | **0.400 · 0.700 · 13** |

Full-match targets (no ground truth used): ~4.2 possessions per live minute, ~45% dead time, ~105 stoppages.

## Where v1.1 still fails (public window, `scripts/diagnose_public.py`)

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
| 1 | Load DFL, translate touch-level → controlled possession (`min_hold` chosen to hit ~4.2/live-min), degrade to challenge-like tracking (calibrated ball gaps tied to stoppages, fragmented tracks) | **Built** · run `scripts/prepare_dfl.py` |
| 2 | Shared features for DFL and challenge | Next |
| 3 | LightGBM: dead vs alive, and which team | — |
| 4 | Decoder + **honest validation**: official spellF1 on held-out DFL matches | — |
| 5 | Predict the challenge match → `v2.csv` | — |

## Open questions and risks

- **Team**: solo for now; teams of < 3 can't advance past Phase 1. Registration closes 10 Nov.
- **Label mismatch**: DFL's possession definition differs from the challenge's; the `min_hold` translation is an approximation.
- **Domain gap**: degraded DFL (Bundesliga 2, synthetic noise) may not behave like the real Metrica tracking.
- **The public window is unrepresentative** (ball almost always visible), so it can't validate behaviour in long gaps.
