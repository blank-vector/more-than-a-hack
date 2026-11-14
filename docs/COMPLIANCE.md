# Compliance with the rules

*Against the organisers' clarification of 3 November 2026, which takes precedence over earlier documentation:*
*"If your algorithm has parameters that need to be tuned, you MUST tune them on external data, never on the ground-truth possession data or on the provided video."*

## What the final submission is fitted, tuned and selected on

| Element | How it was set | Data used |
|---|---|---|
| Model weights (dead/alive, home/away, change detector) | Trained | DFL/Sportec open data only (7 Bundesliga matches, degraded) |
| Training labels: minimum hold (controlled-possession rule) | Chosen so that DFL tracking labels reproduce the possession count from DFL's own **event** data (Opta-style controlled actions) — `scripts/choose_hold_external.py` | DFL only |
| Decoder costs and change-detector strength | Best mean official spellF1, leave-one-match-out | DFL only — `scripts/train_v3.py` |
| Choice between model variants | Best mean official spellF1, leave-one-match-out | DFL only |
| Smart Tagging restart prior | Fixed, hand-set values from football knowledge (dead ball before a restart, play after it); not tuned | Applied as an input, as allowed |
| Dead ball before each half's kick-off | Read from Smart Tagging kick-off tags | Input |
| Noise simulation (ball gaps, track fragments) | Rates measured on the provided automatic tracking: aggregate statistics only, no labels | Provided tracking (input data, not ground truth or video) |

## What was not used to fit, tune or select anything

- **The public ground truth.** It was used to check that the pipeline is wired correctly (frame alignment, file format) and, during development, to look at failure cases. During development, an earlier label hold (2 s) was chosen to match the public window's possession rate, and earlier variants were compared on it. After the 3 November clarification, every such choice was **redone on DFL alone** as listed above.
- **The video.** It is not used at all, neither as input nor for checking.
- **Hand labels.** None were made on the provided match.
- **AI services.** No provided data was uploaded to any. Development used aggregate statistics only.

## How to verify

```bash
python scripts/choose_hold_external.py                          # hold from DFL events vs DFL tracking
python scripts/train_v3.py --min-hold <hold> --save-as configs/final.json   # decoder + variant from DFL CV
python scripts/predict_v3.py --config configs/final.json --models models/final --name final
```
