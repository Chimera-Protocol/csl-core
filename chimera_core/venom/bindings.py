"""
Bindings: which policy guards which agent, kept explicitly in the workspace.

    state.json  "bindings": {agent: {"policy": path, "mapping": path, "at": ts}}

Paths are relative to the workspace, or absolute for adopted policies that live in the
operator's own repository. One policy can guard many agents; every agent has its own mapping
(its own tool names onto the shared policy). Running guards created with
`venom_guard(agent)` look their policy up here and follow changes live.

Older workspaces without explicit bindings resolve by convention: policies/<agent>.csl, or the
adopted policy recorded by setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .workspace import Workspace


@dataclass
class Binding:
    agent: str
    policy: str  # workspace-relative or absolute
    mapping: Optional[str]
    explicit: bool


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def mapping_rel(agent: str, ws: Optional[Workspace] = None) -> str:
    """Where an agent's generated mapping lives, relative to the workspace (beside its policies)."""
    base = ws.rel(ws.policies) if ws is not None else ".csl/policies"
    return f"{base}/{agent.replace('-', '_')}_mapping.py"


class Bindings:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    def all(self) -> Dict[str, Binding]:
        state = self.ws.load_state()
        out: Dict[str, Binding] = {}
        for agent, b in (state.get("bindings") or {}).items():
            if isinstance(b, dict) and b.get("policy"):
                out[agent] = Binding(agent, b["policy"], b.get("mapping"), True)
        # convention for workspaces made before explicit bindings: policies/<agent>.csl with its mapping
        for p in sorted(self.ws.policies.glob("*.csl")) if self.ws.policies.is_dir() else []:
            m = mapping_rel(p.stem, self.ws)
            if p.stem not in out and (self.ws.root / m).exists():
                out[p.stem] = Binding(p.stem, self.ws.rel(p), m, False)
        setup = (state.get("setup") or {}).get("agents") or {}
        for aid, st in setup.items():
            key = st.get("key")
            if key and key not in out and st.get("adopted") and st.get("policy"):
                m = st.get("mapping")
                out[key] = Binding(key, st["policy"], m, False)
        return out

    def get(self, agent: str) -> Optional[Binding]:
        return self.all().get(agent)

    def agents_of(self, policy: str) -> list:
        target = self.abs(policy)
        return sorted(a for a, b in self.all().items() if self.abs(b.policy) == target)

    def abs(self, path: str) -> Path:
        p = Path(path)
        return (p if p.is_absolute() else self.ws.root / p).resolve()

    def rel(self, path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.ws.root))
        except ValueError:
            return str(p)

    def bind(self, agent: str, policy, mapping: Optional[str] = None) -> Binding:
        state = self.ws.load_state()
        entry = {"policy": self.rel(policy), "mapping": mapping, "at": _now()}
        state.setdefault("bindings", {})[agent] = entry
        self.ws.save_state(state)
        self._audit(agent, "bind", entry["policy"])
        return Binding(agent, entry["policy"], mapping, True)

    def unbind(self, agent: str) -> None:
        state = self.ws.load_state()
        (state.get("bindings") or {}).pop(agent, None)
        self.ws.save_state(state)
        self._audit(agent, "unbind", "")

    def _audit(self, agent: str, action: str, detail: str) -> None:
        from .controls import Controls
        Controls(self.ws)._audit(agent, action, detail)
