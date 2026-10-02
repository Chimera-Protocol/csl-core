"""The CSL editor widget: Textual's TextArea with CSL highlighting and problem lines."""

from __future__ import annotations

import re
from typing import Iterable, Set

from rich.style import Style
from textual.widgets import TextArea
from textual.widgets.text_area import TextAreaTheme

KEYWORDS = r"\b(CONFIG|DOMAIN|VARIABLES|STATE_CONSTRAINT|WHEN|THEN|ALWAYS|MUST|NOT|BE|MAY|AND|OR|True|False|ENFORCEMENT_MODE|CHECK_LOGICAL_CONSISTENCY|ENABLE_FORMAL_VERIFICATION|ENABLE_CAUSAL_INFERENCE|INTEGRATION|POLICY_ID|POLICY_VERSION|BLOCK|WARN|LOG|TRUE|FALSE)\b"
TOKENS = re.compile(
    r'(?P<comment>//[^\n]*)|(?P<string>"[^"\n]*")|(?P<number>\b\d+(?:\.\d+)?\b)|(?P<operator>==|!=|<=|>=|<|>|\.\.)|'
    + rf"(?P<keyword>{KEYWORDS})"
)
RULE_NAME = re.compile(r"\bSTATE_CONSTRAINT\s+(\w+)")
VAR_DECL = re.compile(r"^\s*(\w+)\s*:")

THEME = TextAreaTheme(
    name="venom",
    base_style=Style(color="#cbd5e1", bgcolor="#0f172a"),
    gutter_style=Style(color="#475569", bgcolor="#0f172a"),
    cursor_style=Style(color="#0f172a", bgcolor="#5eead4"),
    cursor_line_style=Style(bgcolor="#162033"),
    cursor_line_gutter_style=Style(color="#5eead4", bgcolor="#162033"),
    bracket_matching_style=Style(bgcolor="#134e4a", bold=True),
    selection_style=Style(bgcolor="#1e3a5f"),
    syntax_styles={
        "comment": Style(color="#64748b", italic=True),
        "string": Style(color="#86efac"),
        "number": Style(color="#fbbf24"),
        "operator": Style(color="#f0abfc"),
        "keyword": Style(color="#5eead4", bold=True),
        "function": Style(color="#e2e8f0", bold=True),
        "variable": Style(color="#7dd3fc"),
        "problem": Style(bgcolor="#3b1219", underline=True),
    },
)


class CslEditor(TextArea):
    """TextArea that highlights CSL with regular expressions (no grammar package needed)."""

    def __init__(self, text: str = "", **kwargs) -> None:
        super().__init__(text, language=None, show_line_numbers=True, tab_behavior="indent", soft_wrap=False, **kwargs)
        self.register_theme(THEME)
        self.theme = "venom"
        self.problem_lines: Set[int] = set()

    def mark_problems(self, lines: Iterable[int]) -> None:
        """1-based line numbers to underline as problems."""
        self.problem_lines = {l - 1 for l in lines if l and l > 0}
        self._build_highlight_map()
        self.refresh()

    def _build_highlight_map(self) -> None:  # noqa: D401  (TextArea hook)
        self._line_cache.clear()
        highlights = self._highlights
        highlights.clear()
        doc = getattr(self, "document", None)
        if doc is None:
            return
        for row in range(doc.line_count):
            line = doc.get_line(row)
            enc = line.encode("utf-8")
            if not enc:
                continue

            def b(i: int) -> int:
                return len(line[:i].encode("utf-8"))

            for m in TOKENS.finditer(line):
                kind = m.lastgroup
                if kind:
                    highlights[row].append((b(m.start()), b(m.end()), kind))
            r = RULE_NAME.search(line)
            if r:
                highlights[row].append((b(r.start(1)), b(r.end(1)), "function"))
            v = VAR_DECL.match(line)
            if v and v.group(1) not in ("ENFORCEMENT_MODE", "CHECK_LOGICAL_CONSISTENCY", "INTEGRATION", "POLICY_ID",
                                        "POLICY_VERSION", "ENABLE_FORMAL_VERIFICATION", "ENABLE_CAUSAL_INFERENCE"):
                highlights[row].append((b(v.start(1)), b(v.end(1)), "variable"))
            if row in getattr(self, "problem_lines", set()):
                highlights[row].append((0, len(enc), "problem"))
