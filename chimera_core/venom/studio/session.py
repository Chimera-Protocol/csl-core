"""
The studio's logic, independent of the screen: which policy is open, where it is saved, what
the last Z3 / TLA+ runs said about the current text, and going live.

Editing never touches a live policy directly: the studio works on a draft
(.csl/venom/drafts/<name>.csl). Going live verifies the text, writes the active file in
policies/ (the previous version goes to .csl/venom/history/), binds the selected agents
(each gets a fail-closed mapping) and running guards switch on their next call.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..bindings import Bindings
from ..model import Agent, Inventory
from ..policy import draft as D
from ..policy.gate import verify_text
from .engines import TLARun, Z3Run, run_tla, run_z3

BLANK = """// New CSL policy. F5 checks it with Z3, F8 with TLA+.

CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}

DOMAIN NewPolicy {
  VARIABLES {
    tool: {"read_file", "send_email"}
    amount: 0..10000
  }

  STATE_CONSTRAINT example_limit {
    WHEN tool == "send_email"
    THEN amount <= 100
  }
}
"""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class GoLive:
    ok: bool
    message: str
    active: Optional[str] = None
    bound: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)


class StudioSession:
    def __init__(self, ws, inventory: Optional[Inventory] = None) -> None:
        self.ws = ws
        self.inv = inventory
        self.name = "policy"
        self.draft: Optional[Path] = None
        self.active: Optional[Path] = None  # the live file this draft will replace
        self.external: Optional[Path] = None  # an adopted file outside the workspace (never written)
        self.agents: List[str] = []  # agents selected for binding
        self.text = ""
        self.saved_digest = ""
        self.z3: Optional[Z3Run] = None
        self.z3_digest = ""
        self.tla: Optional[TLARun] = None
        self.tla_digest = ""

    # -- opening -----------------------------------------------------------------------
    def agent_objects(self) -> List[Agent]:
        if self.inv is None:
            return []
        keys = set(self.agents)
        return [a for a in self.inv.agents if D.agent_key(a) in keys]

    def all_agents(self) -> List[str]:
        return [D.agent_key(a) for a in self.inv.agents] if self.inv is not None else []

    def policies(self) -> List[Dict[str, str]]:
        """Open-able policies: live ones (with their agents), drafts, and the policies agents already use."""
        b = Bindings(self.ws)
        out: List[Dict[str, str]] = []
        seen = set()
        for p in sorted(self.ws.policies.glob("*.csl")) if self.ws.policies.is_dir() else []:
            out.append({"path": str(p), "name": p.stem, "status": "live", "agents": ", ".join(b.agents_of(str(p)))})
            seen.add(str(p.resolve()))
        for p in sorted(self.ws.drafts.glob("*.csl")) if self.ws.drafts.is_dir() else []:
            out.append({"path": str(p), "name": p.stem, "status": "draft", "agents": ""})
        for agent, bd in b.all().items():
            ap = str(b.abs(bd.policy))
            if ap not in seen and not ap.startswith(str(self.ws.drafts)):
                out.append({"path": ap, "name": Path(ap).stem, "status": "adopted", "agents": agent})
                seen.add(ap)
        return out

    def open(self, path: Optional[str] = None, agent: Optional[str] = None) -> None:
        if agent and not path:
            bd = Bindings(self.ws).get(agent)
            if bd is not None:
                path = str(Bindings(self.ws).abs(bd.policy))
            else:
                self.new(agent)
                return
        assert path is not None
        p = Path(path).resolve()
        self.name = p.stem
        self.external = None
        if p.is_relative_to(self.ws.drafts.resolve()):
            self.draft, self.active = p, (self.ws.policies / p.name)
        elif p.is_relative_to(self.ws.root):
            self.draft, self.active = self.ws.drafts / p.name, p
        else:
            self.draft, self.active, self.external = self.ws.drafts / p.name, self.ws.policies / p.name, p
        source = self.draft if self.draft.exists() else (self.external or self.active)
        self.text = self.ws.read(source) or (self._read_external(p) if self.external else "") or ""
        self.saved_digest = digest(self.text)
        bound = Bindings(self.ws).agents_of(str(self.active)) if self.active and self.active.exists() else []
        if self.external:
            bound = sorted(set(bound) | set(Bindings(self.ws).agents_of(str(self.external))))
        self.agents = sorted(set(bound) | ({agent} if agent else set()))
        self.z3 = self.tla = None

    def _read_external(self, p: Path) -> Optional[str]:
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def new(self, agent: Optional[str] = None, name: Optional[str] = None) -> None:
        a = next((x for x in (self.inv.agents if self.inv else []) if D.agent_key(x) == agent), None) if agent else None
        self.name = name or (D.agent_key(a) if a else self._free_name())
        self.draft = self.ws.drafts / f"{self.name}.csl"
        self.active = self.ws.policies / f"{self.name}.csl"
        self.external = None
        if self.draft.exists():
            self.text = self.ws.read(self.draft) or ""
        else:
            self.text = D.draft_for(a, self.ws.load_exemptions()).text if a else BLANK
        self.saved_digest = "" if not self.draft.exists() else digest(self.text)
        self.agents = [D.agent_key(a)] if a else []
        self.z3 = self.tla = None

    def _free_name(self) -> str:
        i = 1
        while (self.ws.drafts / f"policy-{i}.csl").exists() or (self.ws.policies / f"policy-{i}.csl").exists():
            i += 1
        return f"policy-{i}"

    # -- editing ----------------------------------------------------------------------
    @property
    def modified(self) -> bool:
        return digest(self.text) != self.saved_digest

    @property
    def state(self) -> str:
        if self.external and not self.draft.exists():
            return "adopted (your file)"
        if self.active and self.active.exists():
            live = self.ws.read(self.active) or ""
            return "live" if digest(live) == digest(self.text) else "live · edited"
        return "draft"

    def save(self, text: str) -> Path:
        self.text = text
        assert self.draft is not None
        self.ws.write_text(self.draft, text)
        self.saved_digest = digest(text)
        return self.draft

    # -- verification -------------------------------------------------------------------
    def verify_z3(self, text: str) -> Z3Run:
        self.text = text
        self.z3, self.z3_digest = run_z3(text), digest(text)
        return self.z3

    def verify_tla(self, text: str, use_real_tlc: bool = True) -> TLARun:
        self.text = text
        self.tla, self.tla_digest = run_tla(text, use_real_tlc=use_real_tlc), digest(text)
        return self.tla

    def z3_current(self, text: str) -> Optional[Z3Run]:
        return self.z3 if self.z3 is not None and self.z3_digest == digest(text) else None

    def tla_current(self, text: str) -> Optional[TLARun]:
        return self.tla if self.tla is not None and self.tla_digest == digest(text) else None

    def suggestions(self, text: str):
        from . import suggest
        from .fit import fit_for

        items = []
        z = self.z3_current(text)
        if z is not None:
            items += suggest.from_z3(z, text)
        t = self.tla_current(text)
        if t is not None:
            items += suggest.from_tla(t)
        for f in fit_for(self.agent_objects(), text):
            items += suggest.from_fit(f, text)
        return suggest.ranked(items)

    def fit(self, text: str):
        from .fit import fit_for
        return fit_for(self.agent_objects(), text)

    def replay(self, text: str):
        from .fit import replay
        return replay(self.ws, self.agents, text)

    # -- going live ---------------------------------------------------------------------
    def go_live(self, text: str, agents: Optional[List[str]] = None) -> GoLive:
        from .. import controls
        from ..policy import binder

        g = verify_text(text)
        if not g.ok:
            issue = g.issues[0].message if g.issues else g.stage
            return GoLive(False, f"not live: the policy fails the gate ({issue})")
        assert self.active is not None and self.draft is not None
        targets = [a for a in (self.inv.agents if self.inv else []) if D.agent_key(a) in set(agents if agents is not None else self.agents)]
        plan = binder.bind(self.ws, self.active, targets, write=False, policy_text=text)
        refused = [f"{r.agent}: {r.message}" for r in plan.results if not r.ok]
        old = self.ws.read(self.active)
        if old is not None and old != text:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.ws.write_text(self.ws.venom / "history" / f"{self.active.stem}-{stamp}.csl", old)
        self.ws.write_text(self.active, text)
        good = [a for a in targets if D.agent_key(a) in {r.agent for r in plan.results if r.ok}]
        if good:
            binder.bind(self.ws, self.active, good, write=True, policy_text=text)
            text = self.ws.read(self.active) or text  # agent_id may have been extended
        if self.draft.exists():
            self.ws.remove(self.draft)
        self.text = text
        self.saved_digest = digest(text)
        self.external = None
        controls.Controls(self.ws)._audit(self.name, "go live", f"{self.ws.rel(self.active)} · {len(good)} agents")
        bound = [D.agent_key(a) for a in good]
        msg = f"live: {self.ws.rel(self.active)}"
        msg += f" · {len(bound)} agents switch on their next call" if bound else " · no agent bound yet (ctrl+b)"
        return GoLive(True, msg, str(self.active), bound, refused)
