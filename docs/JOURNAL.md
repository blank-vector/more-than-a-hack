# Journal

*How the solution was built, step by step: what we tried, what we found, what we decided and why.*
*Current numbers live in [RESULTS.md](RESULTS.md).*

---

## 1. Preparation before the brief (14–20 Oct)

**Goal:** have a working possession pipeline before the challenge opened.

- Built a pipeline on Metrica's three public sample matches: loaders, possession labels derived from events, a nearest-player baseline with hysteresis, a LightGBM frame classifier with Viterbi smoothing, and leave-one-match-out validation.
- Learned early that the sample's ball is unrealistically clean (it is `NaN` exactly when play is dead), so we added a stress test (`degrade_ball`) and judged methods on degraded data.
- Result (frame accuracy, degraded ball): baseline 0.70 → LightGBM + Viterbi 0.845.

**Lesson:** a high score on easy data can come from a quirk of the data. Always test with the shortcut removed.

## 2. Reading the challenge (22 Oct)

The brief changed several assumptions:

| Assumed | Actual |
|---|---|
| Possession from events | Ground truth is read off **corrected tracking**: owning team, or `D` when dead |
| Clean ball | Ball missing on **41%** of graded frames |
| Player IDs | **None**: 46 anonymous track slots |
| Accuracy | Ranked on **spellF1**: possessions as intervals |

**Decisions:**
- Import the official `games.py` and `possession_metrics.py` instead of copying them, so we always score exactly like the organisers.
- **No match data near hosted AI tools** (a competition rule): data lives outside the repo, Claude Code is denied read access, and only aggregate statistics are shared.
- Use the public 5 minutes only to check correctness, never to choose between versions (the organisers show it picks the wrong smoothing window even for their own baseline).

## 3. Profiling the tracking (24 Oct)

`profile_atd.py` and `profile_gaps.py`, aggregates only.

**Findings:**
1. The ball goes missing in **long blocks** (median 10.7 s), not flickers.
2. During those blocks players mostly stand still (80% of frames under 1.5 m/s): **most missing ball is dead ball**.
3. Track fragments last a median **35 s**, much better than feared, so per-track features are viable.
4. Mean player speed separates dead ball from play (1.4 vs 2.2–2.6 m/s).
5. The public window has near-perfect ball visibility during play, so it is **not representative** of the full match.

**Decision:** treat "is it dead?" and "who has it?" as two problems. The first leans on player speed, ball position and Smart Tagging restarts; the second on nearest-track geometry and, in gaps, team shape.

## 4. Checking external training data (24 Oct)

DFL/Sportec open data, 7 Bundesliga matches: per-frame owning team and alive/dead flag on 100% of frames.

**Finding:** DFL changes possession 2× as often as the challenge (7–11 vs ~4.2 per live minute), because it counts any touch.
**Decision:** usable, but its possession labels must be translated to "controlled possession" before training.

## 5. Version 1: rules + decoder (24 Oct)

**Design:** per-frame evidence for dead ball (slow players, missing ball, ball outside the lines, few players tracked, Smart Tagging set pieces) and for team (nearest player within 2 m, weak nearest-team hint, Smart Tagging phase team), decoded by a Viterbi over H/A/D where every state change pays a cost.

**Smart Tagging, learned without ground truth:** for each tag code, check how often its Team tag agrees with the nearest player. Attacking phases agree 61–85% (read as "team in possession"), defending phases 15–31% (read as "defending team"), the rest are ignored.

**Tuning without the answers:** a grid sweep picks settings whose *full-match* output hits targets that use no ground truth: ~4.2 possessions per live minute, ~45% dead time, ~105 stoppages.

**Iterations:**

| Step | Problem seen | Change | Effect |
|---|---|---|---|
| v1.0 | Dead-ball share 71% | Visible ball inside the lines and a moving ball count as live-play evidence; tune the dead-ball threshold | Dead share 47%, possession rate 4.2 |
| v1.0 | Decoder dodged expensive team switches through a brief `D` | Scale both switch costs together (`stickiness`) | Sweep behaves as expected |
| v1.0 | 171 stoppages (DFL: 97–113) | — | Public spellF1 0.315 |
| v1.1 | False short stoppages; dead balls with the ball held look like possession | "Ball standing still" evidence; separate stoppage cost (`--d-cost-ratio 4`) | 107 stoppages; public spellF1 **0.400**, macroF1 **0.700** |

**Diagnosis tool:** `diagnose_public.py` draws truth vs prediction as coloured per-second timelines and grades each true segment. It showed that a missed 3-second possession also merges its two neighbours, costing **three** matches. That explains why spellF1 punishes stickiness.

**Where rules stop:** short possessions and carried-ball stoppages need evidence weighed more finely than hand-set rules allow.

## 6. Version 2: learning from DFL (in progress, 24 Oct)

**Stage 1 (built):** `prepare_dfl.py`
- Load each DFL match into the same arrays as the challenge tracking.
- Translate labels: a team run shorter than `min_hold` inside live play goes back to the previous team; `min_hold` is chosen so DFL reaches ~4.2 possessions per live minute.
- Degrade: missing-ball gaps tied to stoppages and calibrated per match to P(missing | dead) ≈ 64% and P(missing | live) ≈ 22%; track fragments of ~35 s; jitter. Player speed is recomputed from positions for both datasets, so the feature means the same thing on both sides.

**Next:** shared features → LightGBM (dead/alive, team) → decoder tuned by **official spellF1 on held-out DFL matches**, the first honest validation set (~400 possessions per match).
