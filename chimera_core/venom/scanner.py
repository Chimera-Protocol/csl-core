"""
Scanner: runs the discovery layers, resolves agents, and analyzes the inventory.

Layers report progress through an event callback so the terminal can show live
per-layer status. A time budget stops long scans with partial (flagged) results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from . import exemptions as ex
from .analysis import coverage as cov_mod
from .analysis import findings as fnd
from .layers import code, config, governance, processes, triggers
from .model import Host, Inventory
from .probe import HostProbe
from .resolve import Listener, resolve

LAYERS = ["code", "config", "triggers", "runtime", "history", "policies"]

Event = Callable[[str, str, str], None]  # (layer, status: start | done | unavailable, detail)


class Budget:
    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self.partial = False

    def exceeded(self) -> bool:
        return time.monotonic() > self.deadline


@dataclass
class ScanResult:
    inventory: Inventory
    listeners: List[Listener] = field(default_factory=list)
    links: Dict[str, List[str]] = field(default_factory=dict)
    exemptions_applied: List[ex.Exemption] = field(default_factory=list)
    probe: Optional[HostProbe] = None


def _n(count: int, word: str, plural: str = "") -> str:
    return f"{count:,} {word if count == 1 else (plural or word + 's')}"


def _noop(layer: str, status: str, detail: str) -> None:
    pass


def _adopt(inv: Inventory, probe, adopted: Dict[str, str]) -> None:
    """Policies the operator adopted in place (0.5.1 integrations) count as active, without copying."""
    by_path = {p.path: p for p in inv.policies}
    for agent_id, path in adopted.items():
        ref = by_path.get(path)
        if ref is None:
            text = probe.read_text(path)
            if text is None:
                continue
            ref = governance.read_policy(path, text, "active")
            inv.policies.append(ref)
            by_path[path] = ref
        ref.status = "active"
        agent = next((a for a in inv.agents if a.id == agent_id), None)
        if agent is not None and agent.guard.status != "none" and path not in agent.guard.policy_ids:
            agent.guard.policy_ids.append(path)


class Scanner:
    def __init__(self, probe: HostProbe, roots: List[str], workspace=None, window_days: int = 7,
                 budget_s: float = 120.0, on_event: Optional[Event] = None, tool_version: str = "",
                 live_mcp: bool = False) -> None:
        self.probe = probe
        self.roots = [str(r) for r in roots]
        self.workspace = workspace
        self.window_days = window_days
        self.budget = Budget(budget_s)
        self.on_event = on_event or _noop
        self.tool_version = tool_version
        self.live_mcp = live_mcp
        self.config_scan = None

    def _apply_adopted(self, inv: Inventory, adopted: Dict[str, str]) -> None:
        _adopt(inv, self.probe, adopted)

    def run(self) -> ScanResult:
        t0 = time.monotonic()
        probe = self.probe
        host_level = probe.mode in ("host", "fixture")
        ev = self.on_event
        inv = Inventory(tool_version=self.tool_version)
        inv.host = Host(
            name=probe.hostname(), os=probe.os_name(), scanned_at=probe.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            scope=", ".join(self.roots) if probe.mode == "folder" else "host", mode=probe.mode,
        )

        trig = rt = None
        extra_roots: List[str] = []
        if host_level:
            ev("triggers", "start", "")
            trig = triggers.scan_triggers(probe)
            inv.host.layers_unavailable.update({f"triggers.{k}": v for k, v in trig.unavailable.items()})
            inv.host.layers_run.append("triggers")
            ev("triggers", "done", _n(len(trig.records), "entry", "entries"))

            ev("runtime", "start", "")
            rt = processes.scan_runtime(probe)
            if "processes" in rt.sources_ok:
                inv.host.layers_run.append("runtime")
                running = sum(1 for p in rt.processes if processes.assistant_product(p.args) or processes.AGENT_CLI.search(p.args))
                ev("runtime", "done", f"{_n(len(rt.processes), 'process', 'processes')}, {running} agent-like")
            else:
                ev("runtime", "unavailable", rt.unavailable.get("processes", ""))
            inv.host.layers_unavailable.update({f"runtime.{k}": v for k, v in rt.unavailable.items()})

            # Code referenced by triggers and processes joins the code scan (host mode only).
            refs = [p for r in trig.records for p in triggers.referenced_paths(r.command)]
            refs += [p for pr in rt.processes for p in triggers.referenced_paths(pr.args)
                     if processes.AGENT_CLI.search(pr.args) or "agent" in pr.args.lower()]
            for p in refs:
                folder = p if not p.endswith(".py") else str(PurePosixPath(p).parent)
                if probe.is_dir(folder) and not any(folder.startswith(r.rstrip("/") + "/") or folder == r for r in self.roots + extra_roots):
                    if folder.rstrip("/") not in ("", probe.home().rstrip("/")):
                        extra_roots.append(folder)
        else:
            inv.host.layers_unavailable["triggers"] = "folder scope"
            inv.host.layers_unavailable["runtime"] = "folder scope"

        ev("code", "start", "")
        code_roots = self.roots + extra_roots
        cs = code.scan_code(probe, code_roots, self.budget)
        inv.host.files_scanned = cs.files_scanned
        inv.host.parse_errors = cs.parse_errors
        inv.host.layers_run.append("code")
        ev("code", "done", f"{_n(cs.files_scanned, 'file')}, {_n(len(cs.files), 'agent file')}")

        ev("config", "start", "")
        cfg = config.scan_config(probe, code_roots, host_level, self.window_days)
        self.config_scan = cfg
        inv.host.layers_run.append("config")
        ev("config", "done", f"{_n(len(cfg.assistants), 'assistant')}, {_n(cfg.files_read, 'file')}")

        ev("policies", "start", "")
        ws_items = self.workspace.policy_items() if self.workspace is not None else []
        inv.policies = governance.scan_policies(probe, code_roots, ws_items)
        inv.host.layers_run.append("policies")
        ev("policies", "done", f"{sum(1 for p in inv.policies if p.status == 'active')} active, {len(inv.policies)} total")

        ev("history", "start", "")
        res = resolve(probe, code_roots, cs.files, cfg, trig, rt, self.window_days,
                      str(self.workspace.root) if self.workspace is not None else None)
        inv.agents = res.agents
        if self.live_mcp:
            from .layers import mcp_live
            ev("mcp", "start", "")
            probe.allow_spawn = True
            try:
                live = mcp_live.enumerate_live(probe, cfg, inv.agents)
            finally:
                probe.allow_spawn = False
            ok = sum(1 for n in live.servers.values() if n >= 0)
            inv.host.layers_run.append("mcp")
            failed = [s for s, n in live.servers.items() if n < 0]
            if failed:
                inv.host.layers_unavailable["mcp"] = "no answer from: " + ", ".join(sorted(failed))
            ev("mcp", "done", f"{_n(ok, 'server')} listed live")
        inv.host.layers_run.append("history")
        with_runs = sum(1 for a in inv.agents if a.runs.count is not None)
        ev("history", "done", f"run counts for {with_runs} of {_n(len(inv.agents), 'agent')}")

        # exemptions, coverage, drift, findings
        state = self.workspace.load_state() if self.workspace is not None else {}
        self._apply_adopted(inv, state.get("adopted") or {})
        items = self.workspace.load_exemptions() if self.workspace is not None else []
        applied, expired = ex.apply(inv.agents, items, probe.now().date())
        inv.coverage, inv.drift, links = cov_mod.analyze(inv.agents, inv.policies)
        ctx: Dict[str, Any] = {
            "home": probe.home(), "now": probe.now(), "listeners": res.listeners, "links": links,
            "expired": expired, "state": state,
        }
        findings, not_eval = fnd.evaluate(inv, ctx)
        inv.findings, inv.exempted = fnd.split_exempt(findings, inv.agents)
        inv.rules_not_evaluated = not_eval
        inv.host.not_readable = probe.not_readable
        inv.host.partial = self.budget.partial
        inv.host.duration_ms = int((time.monotonic() - t0) * 1000)
        inv.host.layers_run = [l for l in LAYERS + ["mcp"] if l in inv.host.layers_run]
        return ScanResult(inv, res.listeners, links, applied, probe)
