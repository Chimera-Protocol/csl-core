"""Counted words for screens: "1 agent", "2 agents"."""

from __future__ import annotations

IRREGULAR = {"policy": "policies"}


def n(count, word: str) -> str:
    """`count` and `word`, plural unless the count is one."""
    return f"{count} {word if count == 1 else IRREGULAR.get(word, word + 's')}"
