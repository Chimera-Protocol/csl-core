"""
Coverage and vocabulary drift.

Drift is the mapping seam made visible: values the policy speaks about that no real tool
or parameter produces, and parameter types that need coercion into the policy domain.
Z3 cannot see these, because it reasons only over the declared domain.
"""

from __future__ import annotations

import difflib
import re
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from ..model import Agent, Coverage, DriftItem, PolicyRef, Tool
from ..layers.governance import policy_label

TOOL_VARIABLES = {"tool", "tool_name", "action", "function", "operation", "command_name", "capability"}
RANGE = re.compile(r"^\s*-?[\d.]+\s*\.\.\s*-?[\d.]+\s*$")


def norm(s: str) -> str:
    """snake / camel / UPPER / kebab -> one comparable form."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def suggest(value: str, candidates: List[str]) -> Optional[str]:
    by_norm = {norm(c): c for c in candidates}
    if norm(value) in by_norm:
        return by_norm[norm(value)]
    close = difflib.get_close_matches(norm(value), list(by_norm), n=1, cutoff=0.8)
    return by_norm[close[0]] if close else None


def link_policies(agent: Agent, policies: List[PolicyRef]) -> List[PolicyRef]:
    """Policies an agent's guard is wired to."""
    if agent.guard.status == "none":
        return []
    usable = [p for p in policies if p.error is None]
    linked: List[PolicyRef] = []
    for ref in agent.guard.policy_ids:
        base = PurePosixPath(ref).name
        for p in usable:
            if p in linked:
                continue
            if PurePosixPath(p.path).name == base or ref in (p.policy_id, p.domain) or p.path.endswith(ref.lstrip("./")):
                linked.append(p)
    if not linked:
        # a guard built from a non-literal path: policies next to the guard call site
        sites = {str(PurePosixPath(e.path).parent) for e in agent.guard.evidence if e.path.endswith(".py")}
        linked = [p for p in usable if p.status != "draft" and str(PurePosixPath(p.path).parent) in sites]
    if not linked:
        # the workspace policy named after the agent (what setup and `policy new` write)
        from ..policy.draft import agent_key
        key = agent_key(agent)
        linked = [p for p in usable if p.status == "active" and PurePosixPath(p.path).stem == key]
    if not linked:
        active = [p for p in usable if p.status == "active"]
        linked = active if len(active) == 1 else []
    # an active workspace policy supersedes a found copy with the same file name
    active_names = {PurePosixPath(p.path).name for p in linked if p.status == "active"}
    return [p for p in linked if p.status == "active" or PurePosixPath(p.path).name not in active_names]


def tool_variable(policy: PolicyRef, tools: List[Tool]) -> Optional[str]:
    names = [t.name for t in tools]
    best, best_hits = None, 0
    for var, vals in policy.vocabulary.items():
        hits = sum(1 for v in vals if suggest(v, names))
        if var in TOOL_VARIABLES:
            hits += 0.5  # type: ignore[assignment]
        if hits > best_hits:
            best, best_hits = var, hits
    return best if best_hits >= 1 or (best in TOOL_VARIABLES) else None


def _rule_mentions_tool(policy: PolicyRef, var: Optional[str], tool: Tool) -> bool:
    for values in policy.rule_values.values():
        for item in values:
            if "=" not in item:
                continue
            v, val = item.split("=", 1)
            if val == tool.name and (var is None or v == var):
                return True
            if val == tool.risk_class and v in ("risk_class", "tool_class", "risk", "class"):
                return True
    return False


def _covers(policy: PolicyRef, var: Optional[str], tool: Tool) -> bool:
    """A rule speaks about the tool, or a READ tool is explicitly listed in the tool enum
    (with a fail-closed mapping, tools outside the enum are blocked)."""
    if _rule_mentions_tool(policy, var, tool):
        return True
    return tool.risk_class == "READ" and var is not None and tool.name in policy.vocabulary.get(var, [])


def analyze(agents: List[Agent], policies: List[PolicyRef]) -> Tuple[Coverage, List[DriftItem], Dict[str, List[str]]]:
    cov = Coverage()
    drift: List[DriftItem] = []
    links: Dict[str, List[str]] = {}
    for a in agents:
        linked = link_policies(a, policies)
        links[a.id] = [p.path for p in linked]
        agent_exempt = a.exempt is not None and a.exempt.status == "approved"
        any_rule = False
        for t in a.tools:
            cov.tools_total += 1
            if agent_exempt or t.coverage == "exempt":
                t.coverage = "exempt"
                cov.exempt += 1
                continue
            if a.guard.status == "none":
                t.coverage = "unguarded"
                cov.unguarded += 1
                continue
            covered = any(_covers(p, tool_variable(p, a.tools), t) for p in linked)
            if covered:
                t.coverage = "guarded"
                cov.guarded += 1
                any_rule = True
            else:
                t.coverage = "wired_no_rule"
                cov.wired_no_rule += 1
        if a.guard.status != "none" and not any_rule and not agent_exempt:
            a.guard.status = "wired_no_rule"
        a.guard.policy_ids = [policy_label(p) for p in linked] or a.guard.policy_ids
        for p in linked:
            drift += _drift_for(a, p)
    return cov, _dedupe(drift), links


def _dedupe(items: List[DriftItem]) -> List[DriftItem]:
    seen = set()
    out = []
    for d in items:
        key = (d.kind, d.policy, d.variable, d.value, d.agent_id)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _drift_for(a: Agent, p: PolicyRef) -> List[DriftItem]:
    out: List[DriftItem] = []
    label = policy_label(p)
    names = [t.name for t in a.tools if not t.name.endswith("/*")]
    tvar = tool_variable(p, a.tools)
    if tvar and names:
        for val in p.vocabulary.get(tvar, []):
            if val in names:
                continue
            out.append(DriftItem("unknown_value", label, tvar, val, suggest(val, names), a.id, None,
                                 f"no tool of {a.display_name} is named \"{val}\""))
    params = {}
    for t in a.tools:
        for prm in t.params:
            params.setdefault(prm.name, (t, prm))
    for var, domain in p.variables.items():
        if var == tvar or var == "agent_id":
            continue
        hit = params.get(var)
        if hit is None:
            close = suggest(var, list(params))
            if close:
                hit = params[close]
                out.append(DriftItem("unknown_value", label, var, var, close, a.id, hit[0].name,
                                     f"policy variable \"{var}\" is supplied as parameter \"{close}\""))
                continue
            out.append(DriftItem("unsupplied_variable", label, var, None, None, a.id, None,
                                 f"no tool parameter of {a.display_name} supplies \"{var}\""))
            continue
        tool, prm = hit
        vocab = p.vocabulary.get(var)
        ptype = (prm.type or "").lower()
        if vocab is not None:
            if ptype in ("bool", "boolean"):
                out.append(DriftItem("coercion", label, var, None, None, a.id, tool.name,
                                     f"{tool.name}.{var} is bool, policy domain is {{{', '.join(vocab)}}}: map with to_flag()"))
            elif prm.enum:
                for val in vocab:
                    if val not in prm.enum:
                        out.append(DriftItem("unknown_value", label, var, val, suggest(val, prm.enum), a.id, tool.name,
                                             f"{tool.name}.{var} never produces \"{val}\""))
            elif ptype in ("int", "float", "number", "integer"):
                out.append(DriftItem("coercion", label, var, None, None, a.id, tool.name,
                                     f"{tool.name}.{var} is numeric, policy domain is an enum"))
        elif RANGE.match(domain):
            if ptype in ("str", "string"):
                out.append(DriftItem("coercion", label, var, None, None, a.id, tool.name,
                                     f"{tool.name}.{var} is a string, policy domain {domain} is numeric: map with to_range()"))
            elif ptype in ("bool", "boolean"):
                out.append(DriftItem("coercion", label, var, None, None, a.id, tool.name,
                                     f"{tool.name}.{var} is bool, policy domain {domain} is numeric"))
    return out


def blocking_drift(items: List[DriftItem]) -> List[DriftItem]:
    """Drift that counts for V10 and --check: near misses (a value with a close real counterpart)
    and coercions. A value no tool is close to (a ban on a tool the agent does not have) and
    unsupplied variables are informational."""
    return [d for d in items if d.kind == "coercion" or (d.kind == "unknown_value" and d.suggestion)]
