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

**Stage 1 results (real DFL, 7 matches):** touch-level 6.5–11.0 possessions per live minute; a 4.0 s hold brings the pool to 3.9 (3.0 s gives 4.5). Degraded copies: ball visible 52–63%, missing 59–72% in stoppages and 21–24% in live play, ~10 players tracked per team — close to the challenge's 59% / ~64% / ~22% / ~10.5. Gaps run slightly long in three matches (14–16 s vs 10.7 s).

**Caveat noted:** a 4 s hold erases short real possessions from the training labels, and the challenge truth does contain some (0.8 s, 1.3 s). To revisit once validation can measure it.

**Stage 2–3 (built):** 62 features computed identically on both datasets (speeds recomputed from positions; attacking direction inferred per period from where each team stands), two LightGBM models (dead vs alive; home vs away on live frames), and a Viterbi decoder whose two switch costs are chosen by the official spellF1 on held-out DFL matches.

**Results (25 Oct):**
- Leave-one-match-out on 7 DFL matches: **v2 0.723 vs v1 rules 0.429** mean spellF1, v2 ahead on every match. First comparison measured on ~1,450 possessions instead of 17.
- Decoder grid: smooth plateau, best at switch-ha 40 / switch-d 40, not on an edge.
- Challenge match: 202 possessions (3.2 per live minute, below the ~4.2 target), 106 stoppages, 41% dead, Barcelona 56% of live play.
- Public window (sanity): spellF1 **0.571**, macroF1 0.759, dead-ball F1 0.561, 11 vs 17 possessions. Segments 7 and 9 now match (0.80, 0.95): catching segment 8 partially stopped the merge.

**Remaining problems:** v2 under-segments (short possessions still lost), switches about 1 s late, and still misses carried-ball stoppages.

**A subtlety about `min_hold`:** changing it changes the *labels* the validation scores against, so cross-validation scores at different holds are not comparable. The hold has to be chosen from the definition (matching the challenge's possession rate), not from the CV score.

### Round 2 (built, 25 Oct): hold × decoder × lag — `scripts/sweep_hold.py`
For holds of 2, 3 and 4 s: leave-one-match-out on DFL tunes the switch costs **and a lag shift** (every predicted change moved earlier by 0–1 s, kept only if it raises spellF1); the median lag of our changes is measured before and after; then the model is retrained on all DFL and the challenge match is predicted. The hold is picked by the challenge prediction's possession rate (≈ 4.2 per live minute), never by CV score.

**Round 2 results:** a 2 s hold brings the challenge prediction to 3.8 possessions per live minute (4 s: 3.2), with 103 stoppages and 41% dead. No systematic lag on DFL (median 0.00 s), so no shift. Public window: spellF1 0.600 (v2: 0.571); segment 8 now a full match (0.64). Still 11 vs 17 possessions: the sub-4-second possessions and three of the four public stoppages are still missed.

**Why not chase 4.2 exactly:** it comes from 17 possessions, so it is uncertain by roughly ±25% (about ±1 per minute). 3.8 is inside that band.

### Round 3 (built, 25 Oct): training copies + Smart Tagging restarts
- **Training copies:** each DFL match can be degraded several times with different random gaps (`--copies N`), so the model sees more varied missing-ball patterns. Measurable on DFL: same labels, so CV scores at different copy counts *are* comparable.
- **Smart Tagging restart prior** (challenge only, applied to the probabilities before decoding): dead ball boosted for 6 s before each set-piece window through its middle; live play and the restarting team (when named) boosted for 3 s after it; dead ball forced from each goal to the next kick-off. Can't be validated on DFL (no tags); judged by the full-match stoppage count, the share of set pieces preceded by predicted dead ball, and the public timeline.
- **Copies result (hold 2 s, 7 held-out matches):** 1 copy → mean spellF1 **0.657** (decoder 10/40); 3 copies → **0.671** (decoder 20/40). +0.014, better on 5 of 7 matches, one equal, one −0.004. Adopted. Both still under-segment on DFL (e.g. 251 vs 298 possessions), the same pattern as on the challenge.
- **Tag prior result (v2c vs v2c_notags):** set pieces preceded by predicted D 83% → 90%; stoppages 93 → 96; public spellF1 0.586 → 0.600, dead-ball F1 0.482 → 0.527. Adopted.
- **A tension:** with 3 copies the challenge prediction drops to 3.3–3.4 possessions per live minute (1 copy: 3.8). DFL says 3 copies is better; the challenge's sanity targets say it under-segments more. Next: compare 1 copy + tags (v2d) on the full-match checks.
- Safety fix: `train_v2.py` now reads the current config and only saves with `--save`, so it can't silently reset the hold chosen by the sweep.

**Earlier plan (kept):** (dead/alive, team) → decoder tuned by **official spellF1 on held-out DFL matches**, the first honest validation set (~400 possessions per match).
