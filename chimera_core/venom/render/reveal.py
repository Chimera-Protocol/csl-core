"""
The discovery animation: a scanning sweep while the layers run, then the findings arrive.

Real progress only: each layer appears when the scanner reports it (held on screen for a
moment so the eye can follow), counts are the real ones, agents are the ones found. Any key
skips to the end. Never shown for --json, --check, CI, NO_COLOR, non-terminals or --no-anim.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..model import Inventory
from .theme import SEV_GLYPH, STATE_GLYPH, THEME
from .web import AgentMark, Web

LAYERS = [("code", "reading source code"), ("config", "reading assistant and MCP configs"),
          ("triggers", "reading schedules and services"), ("runtime", "looking at running processes"),
          ("policies", "reading policies"), ("history", "counting past runs")]
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MIN_LAYER_S = 0.28
AGENT_DROP_S = 0.07
RISK_SHORT = {"EXTERNAL": "EXT", "DESTRUCTIVE": "DESTR", "IDENTITY": "IDENT", "UNCLASSIFIED": "UNCL"}
RISK_RANK = {c: i for i, c in enumerate(["READ", "WRITE", "EXTERNAL", "IDENTITY", "UNCLASSIFIED", "EXEC", "SPEND", "DESTRUCTIVE"])}
WEB_MIN_WIDTH = 92  # below this the web would crowd the lists; the lists alone are shown


def _hex(style_name: str, fallback: str = "#94a3b8") -> str:
    st = THEME.styles.get(style_name)
    return st.color.triplet.hex if st is not None and st.color is not None and st.color.triplet else fallback


def top_risk(agent):
    return max(agent.tools, key=lambda x: RISK_RANK.get(x.risk_class, 0), default=None)


def sweep(width: int, phase: float) -> Text:
    """A scanning beam moving across the width, fading behind its head."""
    width = max(10, width)
    head = int(phase * (width + 12)) - 6
    ramp = "░▒▓█"
    out = Text()
    for i in range(width):
        d = head - i
        if 0 <= d < 4:
            out.append(ramp[3 - d], style="brand")
        elif 4 <= d < 10:
            out.append("░", style="brand.dim")
        else:
            out.append("─", style="bar.empty")
    return out


def _shimmer(text: str, k: float, base: Optional[str], glow: str, base_text: Optional[Text] = None) -> Text:
    """A band of light passing over the text once (k = 0..1); base_text keeps its own colours."""
    head = k * (len(text) + 8) - 4
    out = base_text.copy() if base_text is not None else Text(text, style=base or "")
    for i in range(len(text)):
        if abs(i - head) < 2.5:
            out.stylize(glow, i, i + 1)
    return out


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


class Reveal:
    def __init__(self, console: Console, version: str) -> None:
        self.console = console
        self.version = version
        self.events: Dict[str, Tuple[str, str, float]] = {}  # layer -> (status, detail, time reported)
        self.shown_at: Dict[str, float] = {}
        self.t0 = time.monotonic()
        self.done = False
        self.skipped = False
        self.inv: Optional[Inventory] = None
        self.lock = threading.Lock()
        self.web: Optional[Web] = None

    # the scanner's event callback (called from the scan thread)
    def event(self, layer: str, status: str, detail: str) -> None:
        with self.lock:
            self.events[layer] = (status, detail, time.monotonic())

    # ------------------------------------------------------------------------------
    def _layer_rows(self, now: float) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(width=1, no_wrap=True)
        t.add_column(width=9, no_wrap=True)
        t.add_column(overflow="ellipsis", no_wrap=True)
        prev = self.t0
        for layer, doing in LAYERS:
            status, detail, _when = self.events.get(layer, ("pending", "", 0.0))
            # a finished layer is revealed no sooner than MIN_LAYER_S after the previous one
            ready = status in ("done", "unavailable") and (self.skipped or now >= prev + MIN_LAYER_S)
            if ready and layer not in self.shown_at:
                self.shown_at[layer] = now
            if layer in self.shown_at:
                prev = self.shown_at[layer]
                age = now - prev
                if status == "unavailable":
                    t.add_row(Text("–", style="muted"), Text(layer, style="muted"), Text(detail or "not available here", style="muted"))
                else:
                    t.add_row(Text("✓", style="ok"), Text(layer, style="text"), self._tick(detail, age))
            elif status == "start" or (status in ("done", "unavailable") and layer not in self.shown_at):
                frame = SPINNER[int(now * 12) % len(SPINNER)]
                t.add_row(Text(frame, style="brand"), Text(layer, style="head"), Text(doing + " ...", style="muted"))
                prev = now + 1e9  # later layers wait
            else:
                t.add_row(Text("·", style="muted"), Text(layer, style="muted"), Text(""))
        return t

    def _tick(self, detail: str, age: float) -> Text:
        """Numbers in the detail count up for a moment after the layer appears."""
        import re

        k = 1.0 if self.skipped else ease(age / 0.45)
        def up(m):
            raw = m.group(0).replace(",", "")
            return f"{int(round(int(raw) * k)):,}"
        return Text(re.sub(r"\d[\d,]*", up, detail), style="muted")

    def _all_shown(self) -> bool:
        return all(l in self.shown_at for l, _ in LAYERS if l in self.events) and self.done

    def _agents(self, now: float, rows: int) -> Group:
        inv = self.inv
        if inv is None or not self._all_shown():
            return Group()
        start = max(self.shown_at.values()) + 0.25 if self.shown_at else now
        n = len(inv.agents) if self.skipped else max(0, int((now - start) / AGENT_DROP_S))
        if n == 0:
            return Group()
        lines: List[Text] = [Text("found", style="label")]
        for a in inv.agents[: min(n, rows)]:
            top = top_risk(a)
            line = Text(STATE_GLYPH.get(a.state, "·") + " ", style=a.state if a.state in ("running", "stopped", "scheduled", "configured") else "muted")
            line.append(a.display_name, style="head")
            if top is not None:
                line.append("  " + RISK_SHORT.get(top.risk_class, top.risk_class), style=f"risk.{top.risk_class}")
            lines.append(line)
        if n > rows and len(inv.agents) > rows:
            lines.append(Text(f"+ {len(inv.agents) - rows} more", style="muted"))
        return Group(*lines)

    def _counters(self, now: float) -> Text:
        inv = self.inv
        t = Text()
        if inv is None or not self._all_shown():
            return t
        start = max(self.shown_at.values()) + 0.25 + min(len(inv.agents), 10) * AGENT_DROP_S if self.shown_at else now
        if not self.skipped and now < start:
            return t
        k = 1.0 if self.skipped else ease((now - start) / 0.5)
        tools = sum(len(a.tools) for a in inv.agents)
        t.append(f"{int(len(inv.agents) * k)} agents", style="head")
        t.append(f" · {int(tools * k)} tools", style="text")
        for s in ("high", "medium", "low"):
            c = sum(1 for f in inv.findings if f.severity == s)
            if c:
                t.append(f"   {SEV_GLYPH[s]} {int(c * k)} {s}", style=s)
        landed = self._landed_at()
        if landed is not None and not self.skipped and landed + 0.3 <= now <= landed + 1.2:
            return _shimmer(t.plain, (now - landed - 0.3) / 0.9, None, "bold #f0fdfa", base_text=t)
        return t

    def _creep(self, i: int, now: float) -> float:
        """How far a layer's vein has grown while its layer runs (it never reaches the node on its own)."""
        start = self.t0 + 0.09 * i
        if now <= start:
            return 0.0
        return 0.1 + 0.62 * (1 - math.exp(-(now - start) / 1.3))

    def _web(self, now: float) -> Text:
        if self.web is None:
            self.web = Web([l for l, _ in LAYERS], w=40, h=13, seed=self.version)
        growth, lit = {}, {}
        for i, (layer, _d) in enumerate(LAYERS):
            shown = self.shown_at.get(layer)
            if shown is None:
                growth[layer] = 1.0 if self.skipped else self._creep(i, now)
            else:
                g0 = self._creep(i, shown)
                growth[layer] = g0 + (1 - g0) * ease((now - shown) / 0.35)
                lit[layer] = now - shown
        marks: List[AgentMark] = []
        if self.inv is not None and self._all_shown():
            start = max(self.shown_at.values()) + 0.25 if self.shown_at else now
            for i, a in enumerate(self.inv.agents):
                top = top_risk(a)
                color = _hex(f"risk.{top.risk_class}") if top is not None else "#94a3b8"
                glyph = "◆" if top is not None and top.risk_class in ("EXEC", "SPEND", "DESTRUCTIVE") else STATE_GLYPH.get(a.state, "●")
                age = 9.0 if self.skipped else now - (start + i * AGENT_DROP_S)
                marks.append(AgentMark(glyph, color, age, a.state == "running"))
        flash = None
        landed = self._landed_at()
        if landed is not None and now >= landed:
            flash = 1.0 if self.skipped else min(1.0, (now - landed) / 0.7)
        return self.web.render(now - self.t0, growth, lit, marks, complete=self.skipped and self.done, flash=flash)

    def _landed_at(self) -> Optional[float]:
        """When the last agent has landed on the web (the closing wave starts then)."""
        if self.inv is None or not self._all_shown() or not self.shown_at:
            return None
        start = max(self.shown_at.values()) + 0.25
        return start + min(len(self.inv.agents), 12) * AGENT_DROP_S + 0.3

    def _status(self, now: float) -> Text:
        """What is being read right now, with a band of light passing over it."""
        done = sum(1 for l, _ in LAYERS if l in self.shown_at)
        t = Text()
        if not self._all_shown():
            doing = next((d for l, d in LAYERS if l not in self.shown_at), "finishing")
            t.append(SPINNER[int(now * 12) % len(SPINNER)] + " ", style="brand")
            t.append_text(_shimmer(doing, (now * 0.8) % 1.0, "#94a3b8", "bold #f0fdfa"))
            t.append(f"   {done}/{len(LAYERS)} · {now - self.t0:3.1f}s · read-only", style="muted")
        else:
            t.append("◆ ", style="brand")
            t.append("read-only · nothing on this host was changed", style="muted")
        return t

    def _beam(self, width: int, now: float) -> Text:
        if self._all_shown():
            return Text("━" * width, style="brand")
        return sweep(width, ((now - self.t0) / 1.6) % 1.0)

    def frame(self, now: float) -> Panel:
        width = min(self.console.width, 100) - 4
        rows = Table.grid(expand=True)
        wide = self.console.width >= 76
        if self.console.width >= WEB_MIN_WIDTH:
            right = Group(self._layer_rows(now), Text(""), self._agents(now, 7))
            rows.add_column(width=41, no_wrap=True)
            rows.add_column(ratio=1)
            rows.add_row(self._web(now), right)
            body = Group(self._status(now), Text(""), rows, Text(""), self._counters(now))
        elif wide:
            rows.add_column(ratio=3)
            rows.add_column(ratio=2)
            rows.add_row(self._layer_rows(now), self._agents(now, 8))
            body = Group(self._beam(width, now), Text(""), rows, Text(""), self._counters(now))
        else:
            body = Group(self._beam(width, now), Text(""), self._layer_rows(now), Text(""),
                         self._agents(now, 5), Text(""), self._counters(now))
        title = Text.assemble((" CSL-Core Venom ", "brand"), (self.version + " ", "muted"),
                              ("· discovering " if not self._all_shown() else "· discovered ", "muted"))
        sub = Text(" any key skips ", style="muted") if not self.skipped else None
        return Panel(body, title=title, title_align="left", subtitle=sub, subtitle_align="right",
                     box=box.ROUNDED, border_style="brand.dim", padding=(0, 1), width=min(self.console.width, 100))

    def finished(self, now: float) -> bool:
        if not self.done or self.inv is None:
            return False
        if self.skipped:
            return True
        if not all(l in self.shown_at for l in self.events):
            return False
        landed = self._landed_at()
        return landed is not None and now >= landed + 1.3


def enabled(console: Console, args) -> bool:
    from ..probe import animation_disabled

    if not console.is_terminal or console.no_color or console.color_system is None:
        return False
    if any(getattr(args, a, False) for a in ("json", "check", "compact", "no_anim")):
        return False
    return not animation_disabled()


def run(console: Console, version: str, scan: Callable[[Callable], object]):
    """Run `scan(on_event)` in a thread while the animation plays; returns scan's result."""
    import sys

    reveal = Reveal(console, version)
    box_: Dict[str, object] = {}

    def work():
        try:
            box_["result"] = scan(reveal.event)
        except BaseException as e:  # surface it in the main thread
            box_["error"] = e
        finally:
            reveal.done = True

    th = threading.Thread(target=work, daemon=True)
    th.start()
    keys = None
    try:
        if sys.stdin.isatty():
            from .keys import Keys
            keys = Keys().__enter__()
        with Live(reveal.frame(time.monotonic()), console=console, refresh_per_second=24, transient=True) as live:
            while True:
                if keys is not None and keys.read(0.04) is not None:
                    reveal.skipped = True
                elif keys is None:
                    time.sleep(0.04)
                if reveal.done and reveal.inv is None and "result" in box_:
                    reveal.inv = getattr(box_["result"], "inventory", None)
                now = time.monotonic()
                live.update(reveal.frame(now))
                if "error" in box_ or reveal.finished(now):
                    break
    finally:
        if keys is not None:
            keys.__exit__(None, None, None)
    th.join()
    if "error" in box_:
        raise box_["error"]  # type: ignore[misc]
    return box_["result"]
