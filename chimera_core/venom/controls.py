"""
Control plane: enforcement mode and kill switches, shared by the CLI, the watch panel and
every running guard.

    state.json  "modes":    {agent: {"mode": "log" | "block", "since": ts}}
                "controls": {agent: {"disabled": bool, "disabled_tools": [...], "at": ts}}
    .csl/venom/audit.jsonl  one line per change (who, what, when)

Guards re-read the state when the file changes, so a change applies to the next tool call
without restarting the agent. A disabled agent or tool is blocked in every mode: the kill
switch is enforcement, not observation.
"""

from __future__ import annotations

import getpass
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .workspace import StateUnreadable, Workspace

MODES = ("log", "block")
NEW_ACTIVATION_MODE = "block"  # an agent whose first policy is activated starts here, unless a mode was chosen


def mode_on_activation(ws: Workspace, agent: str, first: bool, chosen: Optional[str] = None) -> str:
    """The mode an agent runs in once a policy is activated for it. A chosen mode (--mode, a
    question answered, mode = in csl-limits.ini) is set. Otherwise only a first activation sets one:
    block, as the agent's own mode. An agent that already had a policy keeps the mode it has, so
    agents running in log mode are never switched by a later activation."""
    controls = Controls(ws)
    with ws.state_lock():
        has_own = isinstance((ws.load_state().get("modes") or {}).get(agent), dict)
        mode = chosen if chosen in MODES else (NEW_ACTIVATION_MODE if first and not has_own else None)
        if mode is None:
            return controls.get(agent).mode
        if not ws.plan_only and (not has_own or controls.get(agent).mode != mode):
            controls.set_mode(agent, mode)
    return mode


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "operator"


@dataclass
class AgentControl:
    mode: str = "log"
    disabled: bool = False
    disabled_tools: Set[str] = field(default_factory=set)
    exempt: bool = False  # fully trusted by the operator: never blocked, still recorded


class Controls:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    # reading --------------------------------------------------------------------------
    def get(self, agent: str, default_mode: str = "log") -> AgentControl:
        state = self.ws.load_state()
        return self._from_state(state, agent, default_mode)

    @staticmethod
    def _from_state(state: Dict, agent: str, default_mode: str = "log") -> AgentControl:
        """Agent setting, else the workspace default (`cslcore mode --all`), else `default_mode`."""
        m = (state.get("modes") or {}).get(agent)
        d = (state.get("defaults") or {}).get("mode")
        if isinstance(m, dict) and m.get("mode") in MODES:
            mode = m["mode"]
        elif d in MODES:
            mode = d
        else:
            mode = default_mode
        c = (state.get("controls") or {}).get(agent) or {}
        return AgentControl(mode, bool(c.get("disabled")), set(c.get("disabled_tools") or []), bool(c.get("exempt")))

    def all(self) -> Dict[str, AgentControl]:
        state = self.ws.load_state()
        keys = set((state.get("modes") or {})) | set((state.get("controls") or {}))
        return {k: self._from_state(state, k) for k in sorted(keys)}

    # writing --------------------------------------------------------------------------
    def _audit(self, agent: str, action: str, detail: str = "") -> None:
        rec = {"ts": _now(), "by": _who(), "agent": agent, "action": action, "detail": detail}
        if not self.ws.plan_only:
            self.ws.append_line(self.ws.venom / "audit.jsonl", json.dumps(rec, sort_keys=True))

    def set_mode(self, agent: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.ws.transaction() as state:
            state.setdefault("modes", {})[agent] = {"mode": mode, "since": _now()}
        self._audit(agent, f"mode {mode}")

    def default_mode(self) -> Optional[str]:
        d = (self.ws.load_state().get("defaults") or {}).get("mode")
        return d if d in MODES else None

    def set_default(self, mode: str) -> None:
        """Workspace default for agents without their own mode; per-agent modes are kept."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.ws.transaction() as state:
            state.setdefault("defaults", {})["mode"] = mode
        self._audit("*", f"default mode {mode}")

    def set_all(self, mode: str) -> List[str]:
        """Every agent follows `mode`: sets the workspace default and drops per-agent modes.
        Returns the agents whose explicit mode changed."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.ws.transaction() as state:
            changed = [k for k, v in (state.get("modes") or {}).items() if isinstance(v, dict) and v.get("mode") != mode]
            state.setdefault("defaults", {})["mode"] = mode
            state["modes"] = {}
        self._audit("*", f"mode {mode} (all)", ", ".join(changed))
        return changed

    def set_many(self, agents: List[str], mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self.ws.transaction() as state:
            for a in agents:
                state.setdefault("modes", {})[a] = {"mode": mode, "since": _now()}
        for a in agents:
            self._audit(a, f"mode {mode}")

    def set_disabled(self, agent: str, disabled: bool) -> None:
        with self.ws.transaction() as state:
            c = state.setdefault("controls", {}).setdefault(agent, {})
            c["disabled"] = disabled
            c["at"] = _now()
        self._audit(agent, "disable agent" if disabled else "enable agent")

    def set_exempt(self, agent: str, exempt: bool, reason: str = "") -> None:
        """Agent-wide exemption from the panel. Recorded in exemptions.yaml and the audit log."""
        from .model import Exemption

        with self.ws.transaction() as state:
            c = state.setdefault("controls", {}).setdefault(agent, {})
            c["exempt"] = exempt
            c["at"] = _now()
        items = [e for e in self.ws.load_exemptions() if not (e.scope == "agent" and e.agent == agent)]
        if exempt:
            items.append(Exemption(agent=agent, scope="agent", reason=reason, approved_by=_who(), status="approved"))
        self.ws.save_exemptions(items)
        self._audit(agent, "exempt agent" if exempt else "end exemption", reason)

    def set_tool(self, agent: str, tool: str, disabled: bool) -> None:
        with self.ws.transaction() as state:
            c = state.setdefault("controls", {}).setdefault(agent, {})
            tools = set(c.get("disabled_tools") or [])
            (tools.add if disabled else tools.discard)(tool)
            c["disabled_tools"] = sorted(tools)
            c["at"] = _now()
        self._audit(agent, "disable tool" if disabled else "enable tool", tool)

    def audit_tail(self, n: int = 20) -> List[Dict]:
        text = self.ws.read(self.ws.venom / "audit.jsonl") or ""
        out = []
        for line in text.splitlines()[-n:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


class LiveControls:
    """Cached view for a running guard: re-reads state.json only when it changed."""

    def __init__(self, ws: Workspace, agent: str, default_mode: str) -> None:
        import threading

        self.ws = ws
        self.agent = agent
        self.default_mode = default_mode  # what an agent with no mode set anywhere runs in
        self._version: Any = object()  # never equal to a real version: the first call reads
        self.current: Optional[AgentControl] = None  # nothing known yet
        self._lock = threading.Lock()

    def refresh(self) -> AgentControl:
        """The agent's control (mode, freeze, tool switches, exemption) for this call, fail closed.

        Tool calls run in parallel threads (LangGraph's ToolNode, async agents): the state is read
        and only then marked as seen, under a lock, so no call gets a control before it was read.
        A state that cannot be read keeps the last control read; with none read yet, the call
        gets block mode with the agent frozen. A permissive default is never handed out."""
        with self._lock:
            version = self.ws.state_mtime()
            if version != self._version or self.current is None:
                try:
                    state = self.ws.load_state_strict()
                except StateUnreadable:
                    state = None
                if state is not None:
                    self.current = Controls._from_state(state, self.agent, self.default_mode)
                    self._version = version
                elif self.current is None:
                    return AgentControl("block", disabled=True)  # nothing known: stop (and read again next call)
            return self.current
