"""Shared terminal styling for our scripts (rich).

Every script prints through `console`, which also records, so `save()` writes a plain-text copy
to outputs/ for sharing.

Colour convention, used everywhere:
    green  = good / what we want       yellow = watch this       red = problem
    cyan   = section titles            dim    = context, explanations
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(record=True, highlight=False)
ROOT = Path(__file__).resolve().parents[2]


def title(text: str, subtitle: str = "") -> None:
    console.print()
    body = Text(text, style="bold white")
    if subtitle:
        body += Text("\n" + subtitle, style="dim")
    console.print(Panel(body, border_style="cyan", expand=False, padding=(0, 2)))


def section(text: str, note: str = "") -> None:
    console.print()
    console.rule(f"[bold cyan]{text}", align="left", style="cyan")
    if note:
        console.print(f"[dim]{note}[/dim]")


def table(*columns: str, **kw) -> Table:
    t = Table(header_style="bold", border_style="grey50", **kw)
    for i, c in enumerate(columns):
        t.add_column(c, justify="left" if i == 0 else "right")
    return t


def bar(frac: float, width: int = 20, style: str = "cyan") -> Text:
    """A small horizontal bar for a 0-1 share, e.g. ████████░░░░ 62%."""
    frac = 0.0 if frac != frac else max(0.0, min(1.0, frac))  # NaN-safe
    n = round(frac * width)
    return Text("█" * n, style=style) + Text("░" * (width - n), style="grey35") + Text(f" {frac:6.1%}")


def grade(value: float, good: float, ok: float, higher_is_better: bool = True, fmt: str = "{:.3f}") -> Text:
    """Colour a number green/yellow/red against two thresholds."""
    s = fmt.format(value)
    if value != value:
        return Text("n/a", style="dim")
    better = (lambda a, b: a >= b) if higher_is_better else (lambda a, b: a <= b)
    style = "bold green" if better(value, good) else "yellow" if better(value, ok) else "bold red"
    return Text(s, style=style)


def insight(text: str) -> None:
    """A one-line takeaway under a section."""
    console.print(f"[bold magenta]→[/bold magenta] {text}")


def save(name: str) -> Path:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    path = out / f"{name}.txt"
    path.write_text(console.export_text())
    console.print(f"\n[dim]plain-text copy saved to {path}[/dim]")
    return path
