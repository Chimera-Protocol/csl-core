"""
Scan screen and the agent detail view.

Short, dense, professional: the terminal shows what matters, reports carry the rest.
Works at 80 columns; NO_COLOR output has no escape codes.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..model import SENSITIVE, Agent, Inventory, Tool
from .theme import LAYER_LABEL, SEV_GLYPH, STATE_GLYPH
from .words import n as _count

LAYER_ORDER = ["code", "config", "triggers", "runtime", "history", "policies"]
RISK_RANK = {c: i for i, c in enumerate(["READ", "WRITE", "EXTERNAL", "IDENTITY", "UNCLASSIFIED", "EXEC", "SPEND", "DESTRUCTIVE"])}
SPARK = "▁▂▃▄▅▆▇█"
RISK_SHORT = {"EXTERNAL": "EXT", "DESTRUCTIVE": "DESTR", "IDENTITY": "IDENT", "UNCLASSIFIED": "UNCL"}


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------

def _section(label: str) -> Text:
    return Text(f"  {label:<11} ", style="label")


LABEL_WIDTH = 14


def labeled(line):
    """A '  LABEL      content' line as a two-column grid, so long content wraps under itself."""
    if not isinstance(line, Text) or len(line.plain) <= LABEL_WIDTH:
        return line
    t = Table.grid()
    t.add_column(width=LABEL_WIDTH, no_wrap=True)
    t.add_column(overflow="fold")
    t.add_row(line[:LABEL_WIDTH], line[LABEL_WIDTH:])
    return t


def _n(v: Optional[int]) -> str:
    return "n/a" if v is None else f"{v:,}"


def _when(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def top_tool(a: Agent) -> Optional[Tool]:
    tools = [t for t in a.tools if t.coverage != "exempt"] or a.tools
    if not tools:
        return None
    return max(tools, key=lambda t: (RISK_RANK.get(t.risk_class, 0), t.coverage != "guarded", t.name))


def access_label(a: Agent) -> Text:
    t = top_tool(a)
    if t is None:
        return Text("-", style="muted")
    target = t.name
    if t.name.endswith("/*"):
        target = t.name[:-2]
    if a.access.fs_roots and t.risk_class in ("WRITE", "READ"):
        target = "fs:" + a.access.fs_roots[0]
    out = Text()
    out.append(f"{RISK_SHORT.get(t.risk_class, t.risk_class):<6}", style=f"risk.{t.risk_class}")
    out.append(target, style="text")
    return out


def guard_label(a: Agent, wide: bool = True) -> Text:
    if a.exempt is not None and a.exempt.status == "approved":
        return Text("exempt", style="exempt")
    g = a.guard
    if g.status == "none":
        return Text("none", style="high" if any(t.risk_class in SENSITIVE for t in a.tools) else "muted")
    label = g.mechanism or "guard"
    if g.mode:
        label += f" · {g.mode}"
    if g.status == "wired_no_rule":
        return Text(f"{label} · no rule" if wide else "no rule", style="warn")
    return Text(label, style="ok")


def state_label(a: Agent) -> Text:
    if a.exempt is not None and a.exempt.status == "approved":
        return Text(f"{STATE_GLYPH['exempt']} exempt", style="exempt")
    glyph = STATE_GLYPH.get(a.state, "·")
    if a.state == "scheduled":
        sched = next((t.schedule for t in a.triggers if t.type == "time" and t.schedule), None)
        return Text(f"{glyph} {sched or 'scheduled'}", style="scheduled")
    return Text(f"{glyph} {a.state}", style=a.state if a.state in ("running", "stopped", "configured") else "muted")


def coverage_bar(ratio: Optional[float], width: int = 22) -> Text:
    t = Text()
    if ratio is None:
        t.append("░" * width, style="bar.empty")
        return t
    full = int(round(ratio * width))
    t.append("█" * full, style="bar.full")
    t.append("░" * (width - full), style="bar.empty")
    return t


def sparkline(values: Optional[List[int]]) -> Text:
    if not values:
        return Text("n/a", style="muted")
    hi = max(values) or 1
    return Text("".join(SPARK[min(7, int(v / hi * 7))] if v else "·" for v in values), style="brand.dim")


def agent_sort_key(a: Agent, inv: Inventory) -> Tuple:
    sev_rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    worst = min((sev_rank.get(f.severity, 4) for f in inv.findings if f.agent_id == a.id), default=5)
    state = {"running": 0, "scheduled": 1, "stopped": 2, "configured": 3}.get(a.state, 4)
    return (worst, state, -len(a.tools), a.display_name.lower())


# ---------------------------------------------------------------------------
# scan screen
# ---------------------------------------------------------------------------

def header(inv: Inventory, version: str, subtitle: Optional[str] = None) -> Panel:
    h = inv.host
    when = _when(h.scanned_at) if h.scanned_at else ""
    dur = f"{h.duration_ms / 1000:.1f}s"
    scope = "read-only"
    if h.mode == "folder":
        scope = f"folder {h.scope} · read-only"
    elif h.mode == "fixture":
        scope = "fixture host · read-only"
    line = Text(" · ".join(x for x in [h.name, h.os, when, dur, scope] if x), style="text")
    if h.partial:
        line.append(" · partial (time budget)", style="warn")
    title = Text.assemble((" CSL-Core Venom ", "brand"), (version + " ", "muted"))
    if subtitle:
        title.append(f"· {subtitle} ", style="muted")
    return Panel(line, title=title, title_align="left", box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))


def discovery_line(inv: Inventory) -> Group:
    t = _section("Discovery")
    for layer in LAYER_ORDER + (["mcp"] if "mcp" in inv.host.layers_run else []):
        if layer in inv.host.layers_run:
            t.append("✓ ", style="ok")
            t.append(LAYER_LABEL.get(layer, layer) + "  ", style="text")
        else:
            t.append("– ", style="muted")
            t.append(LAYER_LABEL.get(layer, layer) + "  ", style="muted")
    t.rstrip()
    t = labeled(t)
    sub = Text(" " * 14)
    def n(count: int, word: str) -> str:
        return f"{count:,} {word}" + ("" if count == 1 else "s")

    sub.append(f"{n(inv.host.files_scanned, 'file')} · {n(inv.host.parse_errors, 'parse error')} · "
               f"{n(inv.host.not_readable, 'path')} not readable", style="muted")
    return Group(t, labeled(sub))


def agents_line(inv: Inventory) -> Text:
    counts = {s: 0 for s in ("running", "stopped", "scheduled", "configured")}
    exempt = 0
    for a in inv.agents:
        if a.exempt is not None and a.exempt.status == "approved":
            exempt += 1
        if a.state in counts:
            counts[a.state] += 1
    t = _section("AGENTS")
    t.append(f"{len(inv.agents)} total", style="head")
    for s in ("running", "stopped", "scheduled", "configured"):
        if counts[s]:
            t.append(f"   {STATE_GLYPH[s]} {counts[s]} {s}", style=s)
    if exempt:
        t.append(f"   {STATE_GLYPH['exempt']} {exempt} exempt", style="exempt")
    return t


def agents_table(inv: Inventory, width: int, limit: int = 10) -> Table:
    if width < 80:
        return _agents_table_narrow(inv, width, limit)
    wide = width >= 110
    days = inv.agents[0].runs.window if inv.agents else "7d"
    tbl = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 1, 0, 0), header_style="label", expand=False)
    # (header, width, justify); widths are fixed so the table never wraps at 80 columns
    cols = [("  Agent", 28 if wide else 21, "left")]
    if wide:
        cols.append(("Kind", 10, "left"))
    cols += [("State", 16 if wide else 13, "left"), (f"Runs {days}", 8 if wide else 7, "right"), ("Tools", 6 if wide else 5, "right"),
             ("  Top access", 26 if wide else 16, "left"), ("Guard", 18 if wide else 10, "left")]
    for head, w, just in cols:
        tbl.add_column(head, width=w, min_width=w, max_width=w, no_wrap=True, overflow="ellipsis", justify=just)  # type: ignore[arg-type]
    agents = sorted(inv.agents, key=lambda a: agent_sort_key(a, inv))[:limit]
    for a in agents:
        runs = Text(_n(a.runs.count), style="muted" if a.runs.count is None else "text")
        row = [Text("  " + a.display_name, style="head")]
        if wide:
            row.append(Text(a.kind, style="muted"))
        row += [state_label(a), runs, Text(str(len(a.tools)), style="text"), Text("  ") + access_label(a), guard_label(a, wide)]
        tbl.add_row(*row)
    return tbl


def _agents_table_narrow(inv: Inventory, width: int, limit: int) -> Table:
    """Under 80 columns: agent, state, tools, guard."""
    tbl = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 1, 0, 0), header_style="label", expand=False)
    name_w = max(12, width - 2 - 13 - 5 - 10 - 3)
    for head, w, just in (("  Agent", name_w + 2, "left"), ("State", 13, "left"), ("Tools", 5, "right"), ("Guard", 10, "left")):
        tbl.add_column(head, width=w, min_width=w, max_width=w, no_wrap=True, overflow="ellipsis", justify=just)  # type: ignore[arg-type]
    for a in sorted(inv.agents, key=lambda a: agent_sort_key(a, inv))[:limit]:
        tbl.add_row(Text("  " + a.display_name, style="head"), state_label(a), Text(str(len(a.tools)), style="text"), guard_label(a, False))
    return tbl


def coverage_line(inv: Inventory, width: int = 100) -> Text:
    c = inv.coverage
    t = _section("COVERAGE")
    if c.tools_total == 0:
        t.append("no tools discovered", style="muted")
        return t
    t.append_text(coverage_bar(c.ratio, 22 if width >= 100 else (16 if width >= 80 else 8)))
    pct = "n/a" if c.ratio is None else f"{c.ratio * 100:.0f}%"
    t.append(f"  {pct:>4}   ", style="head")
    t.append(f"{c.guarded} of {_count(c.tools_total - c.exempt, 'tool')} guarded", style="text")
    if c.wired_no_rule:
        t.append(f" · {c.wired_no_rule} wired without a rule" if width >= 100 else f" · {c.wired_no_rule} no rule", style="warn")
    return t


def findings_block(inv: Inventory, width: int, limit: int = 5, report_hint: Optional[str] = None) -> Group:
    counts = {s: 0 for s in ("high", "medium", "low", "info")}
    for f in inv.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    t = _section("FINDINGS")
    if not inv.findings:
        t.append("none", style="ok")
    for s in ("high", "medium", "low", "info"):
        if counts[s]:
            t.append(f"{SEV_GLYPH[s]} {counts[s]} {s}   ", style=s)
    t.rstrip()
    lines: List = [labeled(t)]
    room = max(20, width - 11)
    for f in inv.findings[:limit]:
        row = Text("   ")
        row.append(SEV_GLYPH[f.severity] + " ", style=f.severity)
        row.append(f"{f.id}  ", style="label")
        row.append(f.summary if len(f.summary) <= room else f.summary[: room - 1] + "…", style="text")
        lines.append(row)
    more = len(inv.findings) - limit
    if more > 0:
        lines.append(Text(f"          {more} more in {report_hint or 'the report'}", style="muted"))
    if inv.exempted:
        lines.append(Text(f"          {len(inv.exempted)} exempted by operator", style="exempt"))
    return Group(*lines)


def reach_block(inv: Inventory) -> Optional[Group]:
    """The strongest reach chain, step by step, and how many more there are (the open core shows one)."""
    from ..reach import build

    g = build(inv)
    if not g.chains:
        return _direct_block(g)
    top = g.top
    head = _section("REACH")
    for i, n in enumerate(top.nodes):
        if i:
            head.append("  →  ", style="muted")
        node = g.nodes[n]
        head.append(node.label, style="bold #f0abfc" if node.kind != "impact" else "high")
    lines = [labeled(head)]
    for e in top.edges:
        step = Text(" " * LABEL_WIDTH)
        step.append("· ", style="#a21caf")
        step.append(e.evidence, style="muted")
        lines.append(labeled(step))
    more = len(g.chains) - 1
    tail = Text(" " * LABEL_WIDTH)
    tail.append(f"{top.confidence} · {top.hops} steps", style="muted")
    if more:
        tail.append(f" · {more} more reach chain{'s' if more != 1 else ''} on this host", style="muted")
    tail.append(" · see it spread: ", style="muted")
    tail.append("cslcore venom map", style="brand")
    lines.append(labeled(tail))
    from ..reach import strongest_direct
    d = strongest_direct(g)
    if d is not None:  # a single agent that already does the worst on its own is shown too
        line = Text(" " * LABEL_WIDTH)
        line.append("direct: ", style="muted")
        for i, n in enumerate(d):
            if i:
                line.append("  →  ", style="muted")
            line.append(g.nodes[n].label, style="high" if g.nodes[n].kind == "impact" else "bold #f0abfc")
        lines.append(labeled(line))
    return Group(*lines)


SINCE_LINES = 4


def since_block(d) -> Group:
    """What opened or closed since the last scan: a scan is a snapshot, this is what moved."""
    from .report import _when

    head = _section("SINCE")
    head.append(f"last scan {_when(d.since)}", style="text")
    if not d.changed:
        head.append(" · no path opened or closed", style="ok")
        return Group(labeled(head))
    parts = []
    if d.opened:
        parts.append((f"{len(d.opened)} path{'s' if len(d.opened) != 1 else ''} opened", "high"))
    if d.closed:
        parts.append((f"{len(d.closed)} closed", "ok"))
    if d.new_chains:
        parts.append((f"{len(d.new_chains)} new reach chain{'s' if len(d.new_chains) != 1 else ''}", "bold #f0abfc"))
    if d.new_agents:
        parts.append((f"{len(d.new_agents)} new agent{'s' if len(d.new_agents) != 1 else ''}", "text"))
    if d.gone_agents:
        parts.append((f"{len(d.gone_agents)} gone", "muted"))
    for text, style in parts:
        head.append(" · ", style="muted")
        head.append(text, style=style)
    lines = [labeled(head)]
    rows = [("+ ", "high", e) for e in d.opened] + [("− ", "ok", e) for e in d.closed]
    for mark, style, e in rows[:SINCE_LINES]:
        line = Text(" " * LABEL_WIDTH)
        line.append(mark, style=style)
        line.append(d.labels.get(e.src, e.src), style="bold #f0abfc")
        line.append("  →  ", style="muted")
        line.append(d.labels.get(e.dst, e.dst), style="high" if e.dst.startswith("impact:") else "bold #f0abfc")
        if mark == "+ ":
            line.append(f"  {e.evidence}", style="muted")
        lines.append(labeled(line))
    if len(rows) > SINCE_LINES:
        lines.append(labeled(Text(" " * LABEL_WIDTH + f"{len(rows) - SINCE_LINES} more in the report", style="muted")))
    return Group(*lines)


def _direct_block(g) -> Optional[Group]:
    """No chain across agents: say so, and show the most serious exposure a single agent has."""
    from ..reach import direct, strongest_direct

    d = strongest_direct(g)
    if d is None:
        return None
    src, agent, impact = d
    head = _section("REACH")
    head.append("no chain across agents", style="ok")
    head.append(" · direct: ", style="muted")
    for i, n in enumerate((src, agent, impact)):
        if i:
            head.append("  →  ", style="muted")
        head.append(g.nodes[n].label, style="high" if g.nodes[n].kind == "impact" else "bold #f0abfc")
    count = len({a for _s, a, _i in direct(g)})
    tail = Text(" " * LABEL_WIDTH)
    tail.append(f"{count} agent{'s' if count != 1 else ''} take untrusted input and act on it without a rule",
                style="muted")
    tail.append(" · see it: ", style="muted")
    tail.append("cslcore venom map", style="brand")
    return Group(labeled(head), labeled(tail))


def next_step(inv: Inventory) -> Text:
    t = _section("NEXT")
    active = [p for p in inv.policies if p.status == "active"]
    unguarded = inv.coverage.unguarded + inv.coverage.wired_no_rule
    if not active and unguarded:
        t.append("cslcore setup", style="brand")
        t.append(f"  first install: {_count(unguarded, 'tool')} need{'s' if unguarded == 1 else ''} a policy", style="text")
    elif any(d.kind in ("unknown_value", "coercion") for d in inv.drift):
        t.append("cslcore policy fix", style="brand")
        t.append("  apply drift suggestions, shown as a diff first", style="text")
    elif unguarded:
        t.append("cslcore setup", style="brand")
        t.append(f"  continue: {_count(unguarded, 'tool')} still without a rule", style="text")
    else:
        t.append("cslcore map --test", style="brand")
        t.append("  check every mapping for fail-open cases", style="text")
    target = _studio_target(inv)
    if target:
        t.append("\n")
        t.append(f"cslcore studio --agent {target}", style="brand")
        t.append("  write its policy, verify (Z3, TLA+), go live", style="text")
    return t


def _studio_target(inv: Inventory) -> Optional[str]:
    """The agent to write a policy for first: running before idle, then the most risky tools
    without a rule, then the most runs."""
    from ..policy.draft import agent_key

    def open_risk(a) -> int:
        return sum(1 for t in a.tools if t.risk_class != "READ" and t.coverage in ("unguarded", "wired_no_rule"))

    active = {"running": 0, "scheduled": 1}
    ranked = sorted((a for a in inv.agents if open_risk(a)),
                    key=lambda a: (active.get(a.state, 2), -open_risk(a), -(a.runs.count or 0), agent_key(a)))
    return agent_key(ranked[0]) if ranked else None


def scan_screen(inv: Inventory, version: str, width: int, compact: bool = False,
                report_hint: Optional[str] = None, show_findings: bool = True, since=None) -> Group:
    parts: List = [header(inv, version)]
    if not compact:
        parts += [discovery_line(inv), Text()]
    parts.append(labeled(agents_line(inv)))
    if not compact and inv.agents:
        parts += [Text(), agents_table(inv, width)]
        hidden = len(inv.agents) - 10
        if hidden > 0:
            parts.append(Text(f"  {hidden} more agents: cslcore venom report", style="muted"))
    parts += [Text(), labeled(coverage_line(inv, width)), Text()]
    if compact:
        counts = {s: sum(1 for f in inv.findings if f.severity == s) for s in ("high", "medium", "low")}
        t = _section("FINDINGS")
        t.append("   ".join(f"{SEV_GLYPH[s]} {n} {s}" for s, n in counts.items() if n) or "none",
                 style="text" if any(counts.values()) else "ok")
        parts.append(labeled(t))
    elif show_findings:
        parts += [findings_block(inv, width, report_hint=report_hint), Text()]
        reach = reach_block(inv)
        if reach is not None:
            parts += [reach, Text()]
        if since is not None:
            parts += [since_block(since), Text()]
        parts.append(labeled(next_step(inv)))
        if report_hint:
            t = _section("REPORT")
            t.append(report_hint, style="text")
            t.append("  (and .json beside it)" if report_hint.endswith(".md") else "", style="muted")
            parts.append(labeled(t))
    else:
        counts = {s: sum(1 for f in inv.findings if f.severity == s) for s in ("high", "medium", "low")}
        t = _section("FINDINGS")
        t.append("   ".join(f"{SEV_GLYPH[s]} {n} {s}" for s, n in counts.items() if n) or "none", style="text")
        t.append("   explained in the next step", style="muted")
        parts.append(labeled(t))
    return Group(*parts)


# ---------------------------------------------------------------------------
# live progress
# ---------------------------------------------------------------------------

class ScanProgress:
    """Live per-layer progress while the scanner runs (transient; the scan screen follows)."""

    def __init__(self, console: Console, version: str) -> None:
        self.console = console
        self.version = version
        self.state = {l: "pending" for l in LAYER_ORDER}
        self.detail = {l: "" for l in LAYER_ORDER}
        self.spinner = Spinner("dots", style="brand")
        self.live: Optional[Live] = None

    def __enter__(self):
        if self.console.is_terminal:
            self.live = Live(self.render(), console=self.console, refresh_per_second=12, transient=True)
            self.live.__enter__()
        return self

    def __exit__(self, *exc):
        if self.live:
            self.live.__exit__(*exc)

    def event(self, layer: str, status: str, detail: str) -> None:
        if layer in self.state:
            self.state[layer] = status
            self.detail[layer] = detail
        if self.live:
            self.live.update(self.render())

    def render(self):
        rows = Table.grid(padding=(0, 2))
        rows.add_column(width=2)
        rows.add_column(width=10)
        rows.add_column()
        for l in LAYER_ORDER:
            st = self.state[l]
            if st == "start":
                icon = self.spinner
            elif st == "done":
                icon = Text("✓", style="ok")
            elif st == "unavailable":
                icon = Text("–", style="muted")
            else:
                icon = Text("·", style="muted")
            rows.add_row(icon, Text(l, style="text" if st != "pending" else "muted"), Text(self.detail[l], style="muted"))
        title = Text.assemble((" CSL-Core Venom ", "brand"), (self.version + " ", "muted"), ("· discovering ", "muted"))
        return Panel(rows, title=title, title_align="left", box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))


# ---------------------------------------------------------------------------
# agent detail
# ---------------------------------------------------------------------------

def _pad(r, left: int = 2):
    from rich.padding import Padding
    return Padding(r, (0, 0, 0, left))


def _issue_grid(rows: List[Tuple[Text, Text, Text, Optional[str]]]) -> Table:
    """Glyph, id, text (+ recommendation) in a grid so wrapped lines stay aligned."""
    g = Table.grid(padding=(0, 1))
    g.add_column(width=1, no_wrap=True)
    g.add_column(no_wrap=True)
    g.add_column(ratio=1)
    for glyph, fid, text, rec in rows:
        body = Text.assemble(text, ("\n" + rec, "muted")) if rec else text
        g.add_row(glyph, fid, body)
    return g


def agent_detail(inv: Inventory, a: Agent, version: str, width: int) -> Group:
    parts: List = [header(inv, version, subtitle=a.display_name)]
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="label", no_wrap=True)
    meta.add_column(style="text", overflow="fold")
    meta.add_row("Agent", Text(a.display_name, style="head"))
    meta.add_row("Id", Text(a.id, style="muted"))
    meta.add_row("Kind", a.kind)
    meta.add_row("State", state_label(a))
    if a.framework:
        meta.add_row("Framework", ", ".join(a.framework))
    if a.model_ids:
        meta.add_row("Models", ", ".join(a.model_ids[:4]))
    if a.entrypoint:
        meta.add_row("Entrypoint", a.entrypoint)
    if a.process_user:
        meta.add_row("Runs as", Text(a.process_user + ("  (elevated)" if a.access.elevated else ""),
                                     style="high" if a.access.elevated else "text"))
    if a.access.permission_mode:
        meta.add_row("Permissions", Text(a.access.permission_mode, style="high" if a.access.permission_mode == "bypass" else "text"))
    if a.system_prompt.present:
        meta.add_row("System prompt", f"present · {a.system_prompt.length} chars · sha256 {a.system_prompt.sha256}")
    g = guard_label(a)
    if a.guard.policy_ids:
        g.append("  " + ", ".join(a.guard.policy_ids[:3]), style="muted")
    meta.add_row("Guard", g)
    runs = Text(f"{_n(a.runs.count)} in {a.runs.window}  ", style="text")
    if a.runs.per_day:
        runs.append_text(sparkline(a.runs.per_day))
    if a.runs.source:
        runs.append(f"  {a.runs.source}", style="muted")
    meta.add_row("Runs", runs)
    parts += [Text(), _pad(meta), Text()]

    tt = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, padding=(0, 2, 0, 0),
               border_style="muted", expand=False)
    tt.add_column("Tool", style="head", no_wrap=True, max_width=28, overflow="ellipsis")
    tt.add_column("Class", no_wrap=True)
    tt.add_column("Coverage", no_wrap=True)
    tt.add_column("Params", style="muted", overflow="fold", max_width=max(16, width - 58))
    cov_style = {"guarded": "ok", "wired_no_rule": "warn", "unguarded": "high", "exempt": "exempt"}
    for t in sorted(a.tools, key=lambda t: (-RISK_RANK.get(t.risk_class, 0), t.name.lower())):
        params = ", ".join(f"{p.name}: {p.type or 'any'}" for p in t.params) or "-"
        tt.add_row(t.name, Text(t.risk_class, style=f"risk.{t.risk_class}"),
                   Text((t.coverage or "-").replace("_", " "), style=cov_style.get(t.coverage or "", "muted")), params)
    parts += [_section("TOOLS").append(str(len(a.tools)), style="head"), _pad(tt, 4)]

    if a.access.credentials or a.access.fs_roots or a.access.network_listen:
        acc = Table.grid(padding=(0, 2))
        acc.add_column(style="muted", no_wrap=True)
        acc.add_column(style="text", overflow="fold")
        acc.add_column(style="muted", overflow="fold")
        for c in a.access.credentials[:8]:
            acc.add_row("credential", c.name, f"{c.kind} · {c.file}")
        if len(a.access.credentials) > 8:
            acc.add_row("", f"{len(a.access.credentials) - 8} more names", "")
        for r in a.access.fs_roots:
            acc.add_row("filesystem", r, "")
        for x in a.access.network_listen:
            acc.add_row("listens", x, "")
        parts += [_section("ACCESS"), _pad(acc, 4), Text()]
    if a.triggers:
        tr = Table.grid(padding=(0, 2))
        tr.add_column(style="text", no_wrap=True)
        tr.add_column(style="head", no_wrap=True)
        tr.add_column(style="muted", overflow="fold")
        for t in a.triggers:
            tr.add_row(t.type.replace("_", " "), t.schedule or "", t.source or "")
        parts += [_section("TRIGGERS"), _pad(tr, 4), Text()]
    fs = [f for f in inv.findings if f.agent_id == a.id]
    if fs:
        rows = [(Text(SEV_GLYPH[f.severity], style=f.severity), Text(f.id, style="label"), Text(f.summary, style="text"), f.recommendation)
                for f in fs]
        parts += [_section("FINDINGS"), _pad(_issue_grid(rows), 4), Text()]
    if a.evidence:
        ev = Table.grid(padding=(0, 2))
        ev.add_column(style="muted", no_wrap=True)
        ev.add_column(style="text", overflow="fold")
        ev.add_column(style="muted", overflow="fold")
        for e in a.evidence[:8]:
            ev.add_row(e.layer, e.path + (f":{e.line}" if e.line else ""), e.detail or "")
        parts += [_section("EVIDENCE"), _pad(ev, 4)]
    return Group(*parts)
