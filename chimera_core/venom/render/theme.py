"""Shared visual language for Venom screens: one palette, one set of glyphs."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "brand": "bold #5eead4",
    "brand.dim": "#2dd4bf",
    "head": "bold #e2e8f0",
    "label": "bold #94a3b8",
    "muted": "#64748b",
    "text": "#cbd5e1",
    "ok": "#4ade80",
    "warn": "#fbbf24",
    "high": "bold #f87171",
    "medium": "#fb923c",
    "low": "#93c5fd",
    "info": "#a5b4fc",
    "running": "#4ade80",
    "stopped": "#94a3b8",
    "scheduled": "#c4b5fd",
    "configured": "#7dd3fc",
    "exempt": "#a8a29e",
    "risk.READ": "#94a3b8",
    "risk.WRITE": "#fbbf24",
    "risk.EXTERNAL": "#7dd3fc",
    "risk.IDENTITY": "#f0abfc",
    "risk.EXEC": "#f87171",
    "risk.SPEND": "#fb923c",
    "risk.DESTRUCTIVE": "bold #f87171",
    "risk.UNCLASSIFIED": "#e879f9",
    "bar.full": "#2dd4bf",
    "bar.empty": "#334155",
    "code": "#e2e8f0 on #1e293b",
    "selected": "on #134e4a",
})

STATE_GLYPH = {"running": "●", "stopped": "○", "scheduled": "◷", "configured": "◇", "unknown": "·", "exempt": "⊘"}
SEV_GLYPH = {"high": "▲", "medium": "■", "low": "●", "info": "·"}
LAYER_LABEL = {"code": "code", "config": "config", "triggers": "triggers", "runtime": "runtime", "history": "history", "policies": "policies"}


def make_console(no_color: bool = False, width: int | None = None, record: bool = False, file=None) -> Console:
    from ..probe import no_color_requested

    no_color = no_color or no_color_requested()
    return Console(
        theme=THEME, no_color=no_color, color_system=None if no_color else "auto", width=width,
        record=record, file=file, highlight=False, soft_wrap=False,
    )
