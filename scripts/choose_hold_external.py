"""Choose the label hold from EXTERNAL data only (DFL events), as the rules require.

    python scripts/choose_hold_external.py

Why: the hold turns DFL's touch-level possession into "controlled" possession. It used to be
chosen to match the public window's possession rate, which is calibrating on the ground truth
(not allowed, see the organisers' clarification of 3 Nov). Here it is chosen on DFL alone:

  1. From DFL's own EVENT data (passes, shots, dribbles, recoveries, restarts), count possessions
     the Opta way: a possession is a run of CONTROLLED actions by one team; it ends when the other
     team makes a controlled action. Like the challenge's spellF1, a dead ball also ends it
     (a restart starts a new possession, even for the same team).
  2. From DFL's TRACKING labels, count possessions after applying each candidate hold.
  3. Pick the hold whose tracking count matches the event count best, pooled over the matches.

Public data only; output is safe to share.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import dfl  # noqa: E402
from possession.report import console, insight, save, section, status, table, title  # noqa: E402

CONTROLLED = {"PASS", "SHOT", "TAKE_ON", "CARRY", "RECOVERY", "GOALKEEPER"}
DEAD = {"BALL_OUT", "FOUL_COMMITTED"}
HOLDS_S = (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5)


def is_set_piece(ev) -> bool:
    try:
        from kloppy.domain import SetPieceQualifier
        return ev.get_qualifier_value(SetPieceQualifier) is not None
    except Exception:
        return False


def event_possessions(match_id: str) -> tuple[int, Counter]:
    from kloppy import sportec
    ds = sportec.load_open_event_data(match_id=match_id)
    types = Counter()
    n, team, period = 0, None, None
    for ev in ds.events:
        t = ev.event_type.name if ev.event_type is not None else "NONE"
        types[t] += 1
        if ev.period is not None and ev.period.id != period:
            period, team = ev.period.id, None          # new half: next controlled action starts a possession
        if t in DEAD:
            team = None                                 # dead ball ends the possession
            continue
        if t not in CONTROLLED or ev.team is None:
            continue
        if is_set_piece(ev):
            team = None                                 # restart: new possession even for the same team
        if ev.team.team_id != team:
            n += 1
            team = ev.team.team_id
    return n, types


def tracking_possessions(m, hold: int) -> int:
    lab = dfl.controlled_possession(m.truth, m.period, hold)
    return dfl.possession_rate(lab, m.period, m.fps)[2]


def main():
    title("Label hold chosen from external data (DFL events)", "public data · safe to share")
    rows, all_types = [], Counter()
    for mid in dfl.MATCHES:
        try:
            with status(f"{mid}: events + tracking labels ..."):
                n_ev, types = event_possessions(mid)
                m = dfl.load_match(mid)
                counts = {h: tracking_possessions(m, int(round(h * 25))) for h in HOLDS_S}
            all_types.update(types)
            rows.append((mid, n_ev, counts))
        except Exception as e:
            console.print(f"[red]✗[/red] {mid}: {type(e).__name__}: {str(e)[:150]}")
    if not rows:
        return

    section("Event types seen in DFL (sanity check)")
    console.print(", ".join(f"{k} {v:,}" for k, v in all_types.most_common()))

    section("Possessions per match: events (Opta-style) vs tracking labels at each hold")
    t = table("match", "events", *[f"{h:g} s" for h in HOLDS_S])
    for mid, n_ev, counts in rows:
        t.add_row(mid, f"[bold]{n_ev}[/bold]", *[str(counts[h]) for h in HOLDS_S])
    console.print(t)

    err = {h: float(np.mean([abs(np.log(c[h] / max(n, 1))) for _, n, c in rows])) for h in HOLDS_S}
    best = min(err, key=err.get)
    t = table("hold", "mean |log(tracking / events)|", "tracking ÷ events")
    for h in HOLDS_S:
        ratio = np.mean([c[h] / max(n, 1) for _, n, c in rows])
        style = "bold green" if h == best else ""
        t.add_row(Text(f"{h:g} s" + ("  ← chosen" if h == best else ""), style=style),
                  Text(f"{err[h]:.3f}", style=style), f"{ratio:.2f}")
    console.print(t)
    insight(f"hold chosen from DFL alone: {best:g} s. Use it with: python scripts/train_v3.py --min-hold {best:g}")


if __name__ == "__main__":
    main()
    save("choose_hold_external")
