"""
Agent fit and replay for the studio.

fit:     for each agent the policy is (or will be) bound to: are its real tool names in the
         policy, which risky tools have no rule, does its mapping stay fail-closed.
replay:  the policy-variable values recorded in the decision logs are run through the edited
         policy and compared with what happened, so a change shows its effect before going live.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..analysis.coverage import _covers, suggest, tool_variable
from ..layers.governance import read_policy
from ..model import Agent
from ..policy import draft as D

RISKY = ("DESTRUCTIVE", "SPEND", "EXEC", "IDENTITY", "UNCLASSIFIED", "EXTERNAL", "WRITE")


@dataclass
class Fit:
    agent: str
    tools: int
    tool_var: Optional[str]
    covered: int
    renames: Dict[str, str] = field(default_factory=dict)  # policy value -> real tool name
    missing_from_enum: List[str] = field(default_factory=list)
    uncovered_risky: List[Tuple[str, str]] = field(default_factory=list)
    cases: int = 0
    fail_open: int = 0
    mapping_note: str = ""
    _agent: Optional[Agent] = field(default=None, repr=False)

    def rule_patch(self, tool: str):
        def apply(text: str):
            if self._agent is None:
                return None
            t = next((x for x in self._agent.tools if x.name == tool), None)
            if t is None:
                return None
            ref = read_policy("<studio>", text, "draft")
            new, rules = D.extend_text(text, ref.variables, self.tool_var or "tool", [t])
            return new if rules else None
        return apply


def fit_for(agents: List[Agent], text: str, ws=None) -> List[Fit]:
    from ..mapping import codegen, harness
    from ..mapping.assistant import compile_guard
    from ..mapping.spec import build_spec

    ref = read_policy("<studio>", text, "draft")
    if ref.error:
        return []
    out = []
    try:
        guard = compile_guard(text)
    except Exception:
        guard = None
    for a in agents:
        tools = [t for t in a.tools if not t.name.endswith("/*")]
        tv = tool_variable(ref, tools)
        vocab = ref.vocabulary.get(tv, []) if tv else []
        names = [t.name for t in tools]
        f = Fit(D.agent_key(a), len(tools), tv, 0, _agent=a)
        for v in vocab:
            if v not in names:
                s = suggest(v, names)
                if s and s != v:
                    f.renames[v] = s
        f.missing_from_enum = [n for n in names if tv and n not in vocab and n not in f.renames.values()]
        for t in tools:
            if _covers(ref, tv, t):
                f.covered += 1
            elif t.risk_class in RISKY and t.name in vocab:
                f.uncovered_risky.append((t.name, t.risk_class))
        if guard is not None:
            import types
            spec = build_spec(a, ref)
            mod = types.ModuleType("_venom_studio_mapping")
            exec(compile(codegen.generate(spec, "policy"), "<generated mapping>", "exec"), mod.__dict__)
            with contextlib.redirect_stdout(io.StringIO()):
                res = harness.run(spec, mod.map_call, guard, mod)
            f.cases, f.fail_open = len(res.cases), len(res.fail_open)
            if spec.unmapped_tools:
                f.mapping_note = f"{len(spec.unmapped_tools)} tools blocked (not in the policy)"
        out.append(f)
    return out


@dataclass
class Replay:
    total: int = 0
    replayed: int = 0
    skipped: int = 0
    newly_blocked: int = 0
    newly_allowed: int = 0
    examples: List[Tuple[str, str, str, List[str]]] = field(default_factory=list)  # agent, tool, change, rules
    agents: List[str] = field(default_factory=list)
    error: str = ""  # set when the text does not compile, so nothing could be replayed

    @property
    def changed(self) -> int:
        return self.newly_blocked + self.newly_allowed


def replay(ws, agents: List[str], text: str, limit: int = 2000) -> Replay:
    from ...runtime import ChimeraGuard, RuntimeConfig
    from ..observe import OUTSIDE, _compile_quiet

    out = Replay(agents=list(agents))
    try:
        guard = ChimeraGuard(_compile_quiet(text), RuntimeConfig(raise_on_block=False))
    except Exception as e:
        out.error = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        return out
    records = []
    for a in agents:
        path = ws.decision_log(a)
        raw = ws.read(path) or ""
        for line in raw.splitlines()[-limit:]:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    out.total = len(records)
    for rec in records:
        values = rec.get("values") or {}
        rules = rec.get("rules") or []
        if not values or any(v == OUTSIDE for v in values.values()) or any(str(r).startswith("__") for r in rules):
            out.skipped += 1
            continue
        out.replayed += 1
        before_blocked = rec.get("decision") in ("BLOCK", "WOULD_BLOCK")
        r = guard.verify(dict(values))
        after_blocked = not r.allowed
        if after_blocked and not before_blocked:
            out.newly_blocked += 1
            change = "now blocked"
        elif before_blocked and not after_blocked:
            out.newly_allowed += 1
            change = "now allowed"
        else:
            continue
        if len(out.examples) < 6:
            out.examples.append((str(rec.get("agent")), str(rec.get("tool")), change, list(r.violated_rule_ids) or list(rules)))
    return out
