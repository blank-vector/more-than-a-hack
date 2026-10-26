# Frame-level ball possession from automatic tracking

*More Than a Hack 2027, Phase 1 · FC Barcelona vs Athletic Club · technical document*
*Submission: v2d · sha256 `84307b106cc63d00083e155094f1af1546acf93fdf5541487632cec329a0beb6`*

## 1. Summary

We treat the task as **possession segmentation** rather than frame classification. Two gradient-boosted models, trained on the public DFL/Sportec open dataset after it is degraded to resemble the challenge's automatic tracking, give per-frame probabilities of dead ball, home and away possession. A Viterbi decoder turns them into whole possessions, with switch costs tuned by the official spellF1 on held-out DFL matches. Smart Tagging set pieces nudge the decoder at restarts.

On 7 held-out DFL matches (~2,000 scored possessions) the learned model reaches a mean spellF1 of 0.657, against 0.373 for our best hand-tuned rule system on the same matches. On the public 5-minute window it scores spellF1 0.600 / macroF1 0.762, against 0.326 / 0.621 for the published baseline.

## 2. What the data told us

All statistics are aggregates computed locally (`scripts/profile_atd.py`, `scripts/profile_gaps.py`).

- The ball is visible on 59.3% of graded frames, and goes missing in **long blocks** (150 gaps, median 10.7 s, 44% of missing time in gaps over 30 s). Interpolation cannot bridge them.
- **Most missing ball is dead ball.** In gaps over 5 s, 80.5% of frames have a mean player speed under 1.5 m/s (27.8% when the ball is visible); about 30% of gaps begin at or outside a line; the ball reappears a median 11.4 m from where it vanished.
- Player tracklets are long (median 35.5 s), with about 11 players tracked per team.
- In the public window, mean player speed is 1.42 m/s during dead ball vs 2.21–2.61 m/s in play.
- The public window is unusually easy (ball visible 99.9% of the time during home possession), so it cannot validate behaviour in long gaps.

This split the problem in two: **is play dead?** (speeds, ball position, missing ball, restarts) and **who has it?** (nearest players, team shape).

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | Main input. Parsed once into arrays in metres, centre origin; speed recomputed from positions |
| Smart Tagging | Kick-off times (dead ball before each half's kick-off); set-piece windows as a decoder prior |
| Public 5-minute ground truth | Correctness checks and qualitative diagnosis only, never for choosing settings |
| **DFL/Sportec open data** (7 Bundesliga matches, Bassek et al., *Scientific Data*, 2025) | Training and validation labels: per-frame ball status and owning team |
| Homography, both videos, broadcast ATD | **Not used.** Positions are already in pitch coordinates; recovering the missing ball from video would need our own detector |

No challenge data was annotated by hand or shown to any hosted AI tool.

## 4. Method

### 4.1 Training data from another league
DFL tracking is clean and labelled, but differs from the challenge in two ways, which we correct.

**Label definition.** DFL changes possession on any touch (6.5–11 possessions per live minute); the challenge counts controlled possession (about 4.2 in the public window). A team run shorter than a **minimum hold** inside live play is returned to the team that had the ball before it; runs right after a dead ball are kept, because a restart has a clear possessor. The hold (2 s) was chosen so that our *challenge* prediction runs closest to the challenge's possession rate. It cannot be chosen by validation score, because each hold defines a different answer key.

**Input noise.** Each DFL match is degraded: missing-ball stretches are tied to stoppages and calibrated per match to P(missing | dead) ≈ 64% and P(missing | live) ≈ 22%; live-play gaps are log-normal; player tracks are cut into ~35 s fragments with short dropouts, more during stoppages; positions get jitter. The degraded matches show 52–63% ball visibility, against 59% for the challenge.

### 4.2 Features (62 per frame, identical code for both datasets)
- **Ball:** visible, outside the lines, speed, standing still, distance to touchline and goal line, position along the home attacking direction.
- **Nearest players:** distance of the nearest and second-nearest home and away player, their difference, players within 5 m.
- **Team shape:** players tracked, centroid along the attacking direction, front-most player, spread, mean speed, forward velocity. The attacking direction is inferred per period from where each team stands on average.
- **Context:** time since and until the ball was seen, the ball was out, and each team was close to the ball (capped at 10 s); centred rolling means over 1, 3 and 10 s.

### 4.3 Models and decoder
Two LightGBM classifiers (every 5th frame, deterministic): dead vs alive on all frames, and home vs away on live frames. They combine into P(D), P(H), P(A).

A **Viterbi decoder** picks the single most likely label sequence per half, paying a cost for each H↔A change and for entering or leaving D. Without it, per-frame decisions flicker; the official raw baseline has 131 possessions in the public window against 17 true. The costs (20 and 40 log units) were chosen by the mean official spellF1 over leave-one-match-out folds on DFL. A post-hoc lag shift was tested and rejected: the median lag of our changes on DFL is 0.0 s.

### 4.4 Smart Tagging prior (challenge only)
Before decoding: +2 log-odds for dead ball from 6 s before each set-piece window to its middle; −1.5 against dead ball and +1 for the restarting team (when tagged) for 3 s after it. Dead ball is forced before each half's kick-off. This raised the share of set pieces preceded by predicted dead ball from 83% to 88–90%, and the public dead-ball F1 from 0.48 to 0.53–0.56.

## 5. Possession rules implemented

| Situation | Our treatment |
|---|---|
| Loose or contested ball | Stays with the last team in possession (decoder stickiness, and the hold in training labels) |
| Touch or deflection without control | Not a change; shorter than the 2 s hold in training labels and penalised by the decoder |
| Restart (throw-in, corner, free kick, goal kick, kick-off) | Dead ball before it; possession after it, favouring the tagged restarting team |
| Ball out of play | Visible ball outside the lines and ball missing with players slow are strong dead-ball evidence |
| Before each kick-off | Always dead ball |
| Outside graded frames | `X` |

## 6. Validation

| Evaluation | Measure | Result |
|---|---|---|
| Leave-one-match-out, 7 degraded DFL matches | mean spellF1 (official code) | **0.657** (v1 rules: 0.373) |
| Challenge, full match (no ground truth) | possessions per live minute / stoppages / dead share | 3.9 / 106 / 43% (targets ≈ 4.2 / ≈ 105 / ≈ 45%) |
| Challenge, public 5 minutes | spellF1 / macroF1 / dead-ball F1 | 0.600 / 0.762 / 0.561 (baseline_smooth50: 0.326 / 0.621 / 0.323) |

Choices were made on DFL folds or on full-match targets that use no ground truth. The public window was used to check that the pipeline is correct and to look at failure cases.

## 7. Limitations and next steps

- **Short possessions are missed.** In the public window, possessions under ~4 s (0.8–3.6 s) are usually absorbed by their neighbours, and the model predicts 11 possessions against 17. On DFL it under-counts too (about 85% of the true count).
- **Stoppages with the ball carried to the spot** can look like possession (two of four public stoppages are missed).
- **Domain gap.** Simulated noise and translated labels approximate, but are not, the challenge's tracking and definition. Training on three degraded copies scored higher on DFL (0.671) but under-segmented more on the challenge (3.4 per live minute), so we kept one.
- **Not explored:** ball detection from the tactical video to fill gaps, sequence models (temporal CNN or RNN) instead of hand-made context features, and a learned goal-to-kick-off rule.

## 8. Reproduction

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git clone <this repository>
cd <this repository>
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/match/data        # the bundle with tactical/
python scripts/predict_v2.py --config configs/v2d.json --models models/v2d --name v2d
# → outputs/submissions/v2d.csv, sha256 84307b10…
```

The committed models in `models/v2d/` were trained on public DFL data only. To retrain them: `python scripts/predict_v2.py --config configs/v2d.json --save-models models/v2d` (downloads the DFL matches through kloppy). Validation: `python scripts/train_v2.py`.
