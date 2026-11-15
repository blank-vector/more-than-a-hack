# Frame-level ball possession from automatic tracking

*More Than a Hack 2027, Phase 1 · FC Barcelona vs Athletic Club · technical document*

*Submission: `final` · sha256 `565501e7603f55a97e0824af956c033bf42060c9cac238afabf38b2d1f6aeae1`*

## 1. Summary

We treat the task as **possession segmentation** rather than frame classification. We train gradient-boosted models on the public DFL/Sportec open dataset, degraded to resemble the challenge's automatic tracking. Per frame, they give the probabilities of dead ball, home possession and away possession, plus the probability that possession changes at that moment. A Viterbi decoder turns these into whole possessions, with switch costs that drop where a change is likely. Smart Tagging set pieces nudge the decoder at restarts.

**Every learned or tuned element was fitted and selected on DFL data only**, as the organisers' clarification of 3 November requires (details in `docs/COMPLIANCE.md`). This includes the label definition, the model weights, the decoder costs and the choice of variant.

On 7 held-out DFL matches (about 1,800 scored possessions), the submission reaches a mean spellF1 of **0.704**. At the earlier label definition, our best hand-tuned rule system scored 0.373 and our first learned version 0.657 on the same matches. On the public 5-minute window, which we used only as a sanity check, it scores spellF1 0.600 and macroF1 0.726. The published baseline scores 0.326 and 0.621 there.

## 2. What the data told us

All statistics are aggregates computed locally (`scripts/profile_atd.py`, `scripts/profile_gaps.py`).

- The ball is visible on 59.3% of graded frames. It goes missing in **long blocks**: 150 gaps, median 10.7 s, with 44% of missing time in gaps over 30 s. Interpolation cannot bridge them.
- **Most missing ball is dead ball.**
  - In gaps over 5 s, 80.5% of frames have a mean player speed under 1.5 m/s, against 27.8% when the ball is visible.
  - About 30% of gaps begin at or outside a line.
  - The ball reappears a median 11.4 m from where it vanished.
- Player tracklets are long (median 35.5 s), with about 11 players tracked per team.

This split the problem in two:

- **Is play dead?** Speeds, ball position, missing ball and restarts.
- **Who has it?** Nearest players and team shape.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | Main input. Parsed once into arrays in metres with the origin at the centre; speed recomputed from positions |
| Smart Tagging | Kick-off times (dead ball before each half's kick-off); set-piece windows as a decoder prior |
| Public 5-minute ground truth | Wiring checks only (frame alignment, file format) and looking at failure cases; never used to fit, tune or choose anything |
| **DFL/Sportec open data** (7 Bundesliga matches, Bassek et al., *Scientific Data*, 2025) | All training and selection. Tracking with per-frame ball status and owning team; event data used to choose the label definition |
| Homography, both videos, broadcast ATD | **Not used.** Positions are already in pitch coordinates |

No challenge data was annotated by hand or uploaded to any AI service.

## 4. Method

### 4.1 Training data from another league
DFL tracking is clean and labelled, but it differs from the challenge in two ways, which we correct.

**Label definition.** DFL changes possession on any touch, giving 6.5–11 possessions per live minute, whereas the challenge counts controlled possession. A team's run shorter than a **minimum hold** inside live play is given back to the team that had the ball before it. Runs right after a dead ball are kept, because a restart has a clear possessor.

The hold is chosen on DFL alone (`scripts/choose_hold_external.py`):

- We count possessions in DFL's own **event** data the way event providers do. A possession is a run of controlled actions (passes, shots, recoveries) by one team, ended by the other team's controlled action or by a dead ball.
- We then pick the hold at which the tracking labels reproduce that count.

At **2.5 s** the tracking labels give 1.03 times the event count, with a mean |log ratio| of 0.062 over the 7 matches. The next best holds are 3 s (0.080) and 2 s (0.138). The hold cannot be chosen by validation score, because each hold defines a different answer key.

**Input noise.** Each DFL match is degraded to look like the challenge tracking:

- Missing-ball stretches are tied to stoppages and calibrated per match to P(missing | dead) ≈ 64% and P(missing | live) ≈ 22%. Live-play gaps are log-normal.
- Player tracks are cut into ~35 s fragments with short dropouts, more of them during stoppages.
- Positions get jitter.

The degraded matches show 52–63% ball visibility, against 59% for the challenge. These rates were measured as aggregate statistics of the provided tracking, without any labels.

### 4.2 Features (81 per frame, identical code for both datasets)

- **Ball:** whether it is visible, whether it is outside the lines, its speed, whether it is standing still, its distance to the touchline and goal line, and its position along the home team's attacking direction.
- **Nearest players:** distance of the nearest and second-nearest home and away players, their difference, and the number of players within 5 m.
- **Team shape:** players tracked, centroid along the attacking direction, front-most player, spread, mean speed and forward velocity. The attacking direction is inferred per period from where each team stands on average.
- **Context:** time since and until the ball was last seen, the ball was out, and each team was close to the ball (capped at 10 s). Also centred rolling means over 1, 3 and 10 s.
- **Short-range:**
    - ball velocity along the attacking direction, sideways speed and turning;
    - how fast each team's nearest player closes on the ball;
    - tight (≤ 1 m) control over 0.2 and 0.5 s;
    - past-vs-future contrasts (over 0.5, 1 and 2 s) of the distance difference and of player speed.

    At a fixed label definition (2 s hold), these features raised DFL validation from 0.657 to 0.684.

### 4.3 Models and decoder
Three deterministic LightGBM classifiers, each using all 81 features:

- dead vs alive;
- home vs away, on live frames;
- a **change detector**: P(the label changes within ±0.24 s).

The first two combine into P(D), P(H) and P(A).

A **Viterbi decoder** picks the single most likely label sequence for each half. It pays a cost for each H↔A change and for entering or leaving D. Without it, per-frame decisions flicker: in the public window, the official raw baseline has 131 possessions against 17 true ones. Costs vary per frame, `cost_t = max(0, base − k · logit(p_change_t))`, so switching is cheaper where a change is likely.

All three settings were chosen by mean official spellF1 over leave-one-match-out folds on DFL. The grid had 60 settings (H↔A ∈ {10, 20, 40, 80}, D ∈ {20, 40, 80}, k ∈ {0, 1, 2, 4, 8}). The best was **20 (H↔A), 80 (dead ball), k = 1**. The change detector adds almost nothing: the best constant-cost decoder (k = 0) scores 0.703, against 0.704. We also tested a post-hoc lag shift and rejected it, because the median lag of our changes on DFL is 0.0 s.

### 4.4 Smart Tagging prior (challenge only)
Before decoding, we apply fixed, hand-set adjustments from football knowledge:

- +2 log-odds for dead ball, from 6 s before each set-piece window to its middle;
- −1.5 against dead ball and +1 for the restarting team (when tagged) for 3 s after it;
- dead ball forced before each half's kick-off.

These values were not tuned. The prior can't be validated on DFL, which has no tags. It was kept on a check that uses only the provided data: with the prior, 86% of tagged set pieces are preceded by predicted dead ball.

## 5. Possession rules implemented

| Situation | Our treatment |
|---|---|
| Loose or contested ball | Stays with the last team in possession (decoder stickiness, and the hold in the training labels) |
| Touch or deflection without control | Not a change: shorter than the 2.5 s hold in the training labels, and penalised by the decoder |
| Restart (throw-in, corner, free kick, goal kick, kick-off) | Dead ball before it; possession after it, favouring the tagged restarting team |
| Ball out of play | A visible ball outside the lines, or a missing ball while players are slow, is strong dead-ball evidence |
| Before each kick-off | Always dead ball |
| Outside graded frames | `X` |

## 6. Validation

| Evaluation | Measure | Result |
|---|---|---|
| Leave-one-match-out, 7 degraded DFL matches, 2.5 s hold | mean spellF1 (official code) | **0.704** (per match 0.639–0.755; 89% of true possessions found) |
| Same folds, earlier 2 s hold (for comparison between methods) | mean spellF1 | this model 0.688 · first learned version 0.657 · hand-tuned rules 0.373 |
| Challenge, full match (no ground truth) | possessions / per live minute / stoppages / dead share | 223 / 3.5 / 93 / 40% |
| Challenge, public 5 minutes (sanity check only) | spellF1 / macroF1 / dead-ball F1 | 0.600 / 0.726 / 0.463 (baseline_smooth50: 0.326 / 0.621 / 0.323) |

Two runs from the committed models produce byte-identical files.

## 7. Limitations and next steps

- **Short possessions are missed.** On DFL the model finds 89% of the true possessions. In the public window, possessions under ~4 s are usually absorbed by their neighbours (11 predicted against 17).
- **Stoppages are under-detected.** The challenge prediction has 93 dead-ball stretches, while each DFL match has 97–113. With the ball carried to the spot, a stoppage can look like possession. We think our simulated noise makes stoppages easier to see than they are in the real tracking, which pushes the DFL-chosen dead-ball cost up (80). Earlier development variants lowered this cost or swapped the dead-ball model by looking at the challenge itself. We dropped those choices to comply with the rules.
- **Domain gap.** Simulated noise and translated labels approximate, but are not, the challenge's tracking and definition. The most promising fix within the rules is a better noise model: one fitted to licensed tracking data with known errors, or one that reproduces the provided tracking's aggregate gap statistics more faithfully.
- **Not explored:** ball detection from the tactical video to fill gaps; sequence models (temporal CNN or RNN) instead of hand-made context features; a learned goal-to-kick-off rule.

## 8. Reproduction

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git clone https://github.com/blank-vector/more-than-a-hack
cd more-than-a-hack
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/match/data        # the bundle with tactical/
python scripts/predict_v3.py --config configs/final.json --models models/final --name final
# → outputs/submissions/final.csv
# sha256 565501e7603f55a97e0824af956c033bf42060c9cac238afabf38b2d1f6aeae1
```

The committed models in `models/final/` were trained on public DFL data only. To rebuild every choice from scratch (the scripts download the DFL matches through kloppy):

```bash
# label hold from DFL events → 2.5 s
python scripts/choose_hold_external.py
# decoder costs by DFL CV
python scripts/train_v3.py --min-hold 2.5 --save-as configs/final.json
# train on all DFL matches and predict
python scripts/predict_v3.py --config configs/final.json --save-models models/final --name final
```
