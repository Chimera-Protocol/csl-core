"""
`cslcore venom --share`: a card of this machine's reach map, made to be posted.

What is on it: the counts, the reach map, and the strongest chain (or the most serious direct
exposure). What is never on it: the host name, user names, file paths, credential names.
`--anonymize` also replaces agent names with their kind ("assistant 1", "code 2").

Written as SVG (always) and PNG when a converter is installed (cairosvg or rsvg-convert).
"""

from __future__ import annotations

import copy
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..model import Inventory
from ..reach import build, direct, strongest_direct
from .theme import SEV_GLYPH, THEME
from .topo import Topo

CARD_WIDTH = 104
REPO = "github.com/Chimera-Protocol/csl-core"
PATHISH = re.compile(r"(/[\w.@~-]+){2,}")


def anonymized(inv: Inventory) -> Inventory:
    """A copy whose agents are named by kind and number only."""
    out = copy.deepcopy(inv)
    counts = {}
    for a in out.agents:
        counts[a.kind] = counts.get(a.kind, 0) + 1
        a.display_name = f"{a.kind} {counts[a.kind]}"
    return out


def _clean(label: str) -> str:
    """No file paths on a card that leaves the machine."""
    return PATHISH.sub("…", label)


def card(inv: Inventory, version: str, anonymize: bool = False) -> Panel:
    inv = anonymized(inv) if anonymize else inv
    g = build(inv)
    for n in g.nodes.values():
        n.label = _clean(n.label)
    findings = {s: sum(1 for f in inv.findings if f.severity == s) for s in ("high", "medium", "low")}
    tools = sum(len(a.tools) for a in inv.agents)
    head = Text.assemble((f"{len(inv.agents)} agents", "bold #e2e8f0"), (f" · {tools} tools", "#cbd5e1"))
    for s, n in findings.items():
        if n:
            head.append(f"   {SEV_GLYPH[s]} {n} {s}", style=s)
    head.append(f"   {len(g.chains)} reach chain{'s' if len(g.chains) != 1 else ''}", style="bold #f0abfc")

    topo = Topo(g, w=58, h=17, max_agents=10, seed=version)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=59, no_wrap=True)
    grid.add_column(width=CARD_WIDTH - 59 - 8)
    grid.add_row(topo.render(30.0, complete=True, labels=True), Group(*topo.legend(99.0, complete=True)))

    story: List[Text] = []
    if g.top is not None:
        story.append(Text.assemble(("STRONGEST CHAIN  ", "label"),
                                   ("  →  ".join(g.nodes[n].label for n in g.top.nodes), "bold #f0abfc")))
    else:
        d = strongest_direct(g)
        if d is not None:
            n = len({a for _s, a, _i in direct(g)})
            story.append(Text.assemble(("DIRECT  ", "label"), ("  →  ".join(g.nodes[x].label for x in d), "bold #f0abfc"),
                                       (f"   {n} agent{'s' if n != 1 else ''} act on untrusted input without a rule",
                                        "#94a3b8")))
        else:
            story.append(Text("No agent here acts on untrusted input without a rule.", style="ok"))
    foot = Text.assemble(("pip install csl-core", "bold #5eead4"), ("  ·  ", "#475569"), ("cslcore venom", "bold #5eead4"),
                         ("  ·  ", "#475569"), (REPO, "#94a3b8"))
    title = Text.assemble((" CSL-Core Venom ", "bold #5eead4"), ("· what can reach what on this machine ", "#94a3b8"))
    body = Group(head, Text(""), grid, Text(""), *story, Text(""), foot)
    return Panel(body, title=title, title_align="left", box=box.ROUNDED, border_style="#2dd4bf",
                 padding=(1, 2), width=CARD_WIDTH)


def export(inv: Inventory, version: str, ws, anonymize: bool = False) -> Tuple[Path, Optional[Path]]:
    """Write the card into the workspace as SVG, and PNG when a converter is available.
    Returns (svg, png or None). Files and converters go through the workspace."""
    from datetime import datetime

    console = Console(theme=THEME, record=True, width=CARD_WIDTH, file=io.StringIO(), force_terminal=True,
                      color_system="truecolor")
    console.print(card(inv, version, anonymize))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    svg = ws.venom / "share" / f"venom-{stamp}.svg"
    ws.write_text(svg, console.export_svg(title="CSL-Core Venom"))
    return svg, ws.svg_to_png(svg)
