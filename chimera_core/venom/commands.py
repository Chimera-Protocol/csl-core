"""
Handlers for `cslcore venom ...` and `cslcore setup`. Imported lazily by chimera_core.cli_venom.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from . import VENOM_VERSION, redact
from .analysis.coverage import blocking_drift
from .layers.history import parse_window
from .model import Inventory
from .probe import probe_for
from .render import report as report_mod
from .render.screen import ScanProgress, agent_detail, scan_screen
from .render.theme import make_console
from .scanner import ScanResult, Scanner
from .workspace import Workspace

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CHECK_FAILED = 3


def console_for(args):
    return make_console(no_color=getattr(args, "no_color", False))


def workspace_for(args) -> Workspace:
    return Workspace(getattr(args, "workspace", None) or os.getcwd(), plan_only=getattr(args, "plan_only", False))


def run_scan(args, console, *, live: bool = True) -> ScanResult:
    root = getattr(args, "root", None)
    if root and not os.path.isdir(root):
        raise SystemExit(f"--root {root}: not a folder")
    probe, roots = probe_for(root)
    ws = workspace_for(args)
    window = parse_window(getattr(args, "since", None) or "7d")
    live_mcp = bool(getattr(args, "probe", False)) and confirm_probe(args, console, probe, roots)

    def make(on_event):
        return Scanner(probe, roots, ws, window_days=window, budget_s=getattr(args, "budget", 120.0),
                       on_event=on_event, tool_version=VENOM_VERSION, live_mcp=live_mcp)

    from .render import reveal
    if live and reveal.enabled(console, args):
        return reveal.run(console, VENOM_VERSION, lambda on_event: make(on_event).run())
    if live:
        progress = ScanProgress(console, VENOM_VERSION)
        with progress:
            return make(progress.event).run()
    return make(None).run()


def confirm_probe(args, console, probe, roots) -> bool:
    """--probe starts stdio MCP servers: show exactly what will run, then ask."""
    from rich.text import Text

    from .layers import config as cfg_layer
    from .layers.mcp_live import servers_to_probe
    from .policy.workbench import confirm

    cfg = cfg_layer.scan_config(probe, roots, probe.mode in ("host", "fixture"))
    servers = servers_to_probe(cfg)
    if not servers:
        console.print("  [muted]--probe: no MCP servers configured[/muted]")
        return False
    console.print(Text("  --probe will ask these MCP servers for their tool lists:", style="label"))
    for s in servers:
        what = s.url if s.url else redact.text(" ".join([s.command or ""] + s.args))
        kind = "query (loopback only)" if s.url else "start, list tools, stop"
        console.print(Text.assemble(("    ", ""), (s.name.ljust(16), "head"), (kind.ljust(24), "muted"), (what, "text")))
    return confirm(console, "Start / query them now?", bool(getattr(args, "yes", False)), default=False)


def save_report(ws: Workspace, inv: Inventory) -> Optional[str]:
    stamp = inv.host.scanned_at.replace("-", "").replace(":", "").replace("T", "-")[:13] or "latest"
    data = report_mod.to_json(inv)
    js = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    md = report_mod.to_markdown(inv)
    ws.write_text(ws.reports / f"report-{stamp}.json", js)
    ws.write_text(ws.reports / f"report-{stamp}.md", md)
    ws.write_text(ws.reports / "latest.json", js)
    ws.write_text(ws.reports / "latest.md", md)
    ws.save_inventory(data, stamp)
    return ws.rel(ws.reports / "latest.md")


def check_failed(inv: Inventory, fail_on: str) -> bool:
    levels = {"high": ["high"], "medium": ["high", "medium"], "low": ["high", "medium", "low"]}[fail_on]
    return any(f.severity in levels for f in inv.findings) or bool(blocking_drift(inv.drift))


# ---------------------------------------------------------------------------
# scan / report
# ---------------------------------------------------------------------------

def cmd_scan(args) -> int:
    console = console_for(args)
    as_json = bool(getattr(args, "json", False))
    result = run_scan(args, console, live=not as_json and not args.check)
    inv = result.inventory
    ws = workspace_for(args)
    hint = None
    if not getattr(args, "no_save", False):
        hint = save_report(ws, inv)
    if as_json:
        sys.stdout.write(report_mod.json_text(inv))
    else:
        console.print(scan_screen(inv, VENOM_VERSION, console.width, compact=args.compact, report_hint=hint))
        console.print()
    if args.check:
        failed = check_failed(inv, args.fail_on)
        if not as_json:
            msg = "check failed" if failed else "check passed"
            console.print(f"  [label]CHECK[/label]       [{'high' if failed else 'ok'}]{msg}[/] "
                          f"[muted](fail on {args.fail_on} findings or vocabulary drift)[/muted]")
        return EXIT_CHECK_FAILED if failed else EXIT_OK
    return EXIT_OK


def _load_inventory(args, console) -> Inventory:
    ws = workspace_for(args)
    data = ws.latest_inventory()
    if data is None or getattr(args, "rescan", False):
        return run_scan(args, console).inventory
    return Inventory.from_dict(data)


def cmd_map_view(args) -> int:
    """`cslcore venom map`: the reach map, full screen and interactive."""
    from .render import mapview

    console = console_for(args)
    inv = _load_inventory(args, console)
    return mapview.run(console, inv, seed=VENOM_VERSION, once=bool(getattr(args, "once", False)))


def cmd_report(args) -> int:
    console = console_for(args)
    inv = _load_inventory(args, console)
    if args.agent:
        a = inv.agent(args.agent) or next((x for x in inv.agents if args.agent in x.id), None)
        if a is None:
            console.print(f"[high]no agent matches '{args.agent}'[/high]. Known agents:")
            for x in inv.agents:
                console.print(f"  {x.display_name}  [muted]{x.id}[/muted]")
            return EXIT_USAGE
        console.print(agent_detail(inv, a, VENOM_VERSION, console.width))
        return EXIT_OK
    if args.format == "json":
        sys.stdout.write(report_mod.json_text(inv))
    elif args.format == "md":
        sys.stdout.write(report_mod.to_markdown(inv))
    else:
        console.print(scan_screen(inv, VENOM_VERSION, console.width))
        from .render.screen import agents_table
        if len(inv.agents) > 10:
            console.print()
            from rich.text import Text
            console.print(Text("  ALL AGENTS", style="label"))
            console.print(agents_table(inv, console.width, limit=len(inv.agents)))
    return EXIT_OK
