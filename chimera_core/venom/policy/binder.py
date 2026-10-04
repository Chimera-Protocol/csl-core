"""
Bind a policy to agents: the one place that decides whether an agent may run under a policy.

For each agent:
  1. if the policy has an agent_id variable that does not list the agent yet, the agent is
     added to it (workspace policies only; the edit passes the gate);
  2. a mapping for this agent is generated (its own tool names onto the policy);
  3. the mapping test must find no fail-open case;
then the binding is recorded and running guards pick it up on their next call.
"""

from __future__ import annotations

import contextlib
import io
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..bindings import Bindings, mapping_rel
from ..layers.governance import read_policy
from ..model import Agent
from .draft import agent_key
from .gate import verify_text
from ..render.words import n as _n


@dataclass
class BindResult:
    agent: str
    ok: bool
    message: str
    cases: int = 0
    fail_open: int = 0
    edited_policy: bool = False


@dataclass
class BindPlan:
    policy: Path
    results: List[BindResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


def _with_agent_id(text: str, key: str) -> Optional[str]:
    """Add `key` to the agent_id enum; None when the policy has no agent_id enum."""
    m = re.search(r'(?m)^(\s*agent_id\s*:\s*)\{([^}]*)\}', text)
    if not m:
        return None
    values = re.findall(r'"([^"]*)"', m.group(2))
    if key in values:
        return text
    values.append(key)
    return text[: m.start(2)] + ", ".join(f'"{v}"' for v in values) + text[m.end(2):]


def bind(ws, policy: Path, agents: List[Agent], *, write: bool = True, policy_text: Optional[str] = None) -> BindPlan:
    from ..mapping import codegen, harness
    from ..mapping.assistant import compile_guard
    from ..mapping.spec import build_spec

    plan = BindPlan(policy)
    bindings = Bindings(ws)
    in_workspace = policy.resolve().is_relative_to(ws.root)
    text = policy_text if policy_text is not None else (ws.read(policy) or "")
    for agent in agents:
        key = agent_key(agent)
        edited = False
        if "agent_id" in text:
            new = _with_agent_id(text, key)
            if new is not None and new != text:
                if not in_workspace:
                    plan.results.append(BindResult(key, False, f"{policy.name} is your own file and does not list {key} "
                                                   "in agent_id; add it there first"))
                    continue
                g = verify_text(new)
                if not g.ok:
                    plan.results.append(BindResult(key, False, "adding the agent to agent_id fails the gate"))
                    continue
                text, edited = new, True
        rel = bindings.rel(policy)
        ref = read_policy(str(policy), text, "active")
        from .limits import load as load_limits

        spec = build_spec(agent, ref, load_limits(ws, key))
        m = mapping_rel(key, ws)
        code = codegen.generate(spec, rel, str((ws.root / m).parent))
        mod = types.ModuleType("_venom_bind_mapping")
        mod.__file__ = str(ws.root / m)  # the generated code reads its scope roots from its own place
        exec(compile(code, "<generated mapping>", "exec"), mod.__dict__)  # the generator's own output
        with contextlib.redirect_stdout(io.StringIO()):
            res = harness.run(spec, mod.map_call, compile_guard(text), mod)
        if res.fail_open:
            plan.results.append(BindResult(key, False, f"mapping would be fail-open in {_n(len(res.fail_open), 'case')}",
                                           len(res.cases), len(res.fail_open)))
            continue
        unmapped = f"; {_n(len(spec.unmapped_tools), 'tool')} not in the policy are blocked" if spec.unmapped_tools else ""
        plan.results.append(BindResult(key, True, f"{_n(len(res.cases), 'case')}, 0 fail-open{unmapped}", len(res.cases), 0, edited))
        if write and not ws.plan_only:
            if edited:
                ws.write_text(policy, text)
            ws.write_text(ws.root / m, code)
            bindings.bind(key, policy, m)
    return plan
