"""
Mapping specification: for every policy variable, where its value comes from.

    constant   agent_id
    tool       the tool name, through a value table (real name -> policy value)
    param      a tool parameter (auto-matched by name, drift suggestions applied)
    derived    computed by the integration: approval, *_allowlisted, target_in_scope, and what the call
               would do (command_class, sql_class, path_class: chimera_core.actions)
    context    supplied by the caller (for example user_role)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..analysis.coverage import RANGE, norm, suggest, tool_variable
from ..model import Agent, PolicyRef, ToolParam
from ..policy.draft import agent_key

CLASSIFY = {"command_allowlisted": "command", "destination_allowlisted": "destination", "target_in_scope": "scope"}
DERIVED = {
    "approval": "approval token from the caller (human in the loop)",
    "command_allowlisted": "command is in COMMAND_ALLOWLIST",
    "destination_allowlisted": "destination is in DESTINATION_ALLOWLIST",
    "target_in_scope": "path is under one of SCOPE_ROOTS",
    "command_class": "what the command would do (chimera_core.actions.command_class)",
    "sql_class": "what the query would do (chimera_core.actions.sql_class)",
    "path_class": "where the call writes or reads (chimera_core.actions.path_class, SCOPE_ROOTS)",
}
NEUTRAL = {"command_class": "OK", "sql_class": "READ", "path_class": "IN_SCOPE"}
COMMAND_PARAMS = re.compile(r"^(command|cmd|script|code|shell_command|query)$", re.I)
DESTINATION_PARAMS = re.compile(r"^(to|recipient|recipients|email|channel|url|webhook|phone|to_wallet|wallet|address|destination|visibility)$", re.I)
PATH_PARAMS = re.compile(r"^(path|file|file_path|notebook_path|target_file|filename|target|dest|directory|dir)$", re.I)


@dataclass
class VarSpec:
    name: str
    kind: str  # enum | range | flag | other
    values: List[str] = field(default_factory=list)
    low: Optional[int] = None
    high: Optional[int] = None
    source: str = "context"  # constant | tool | param | derived | context
    constant: Optional[str] = None
    params: Dict[str, ToolParam] = field(default_factory=dict)  # tool name -> param
    note: Optional[str] = None


@dataclass
class MappingSpec:
    agent_id: str
    agent_name: str
    agent_key: str
    policy_path: str
    tool_var: Optional[str]
    tool_table: Dict[str, str] = field(default_factory=dict)  # real tool -> policy value
    unmapped_tools: List[str] = field(default_factory=list)
    variables: List[VarSpec] = field(default_factory=list)
    tool_params: Dict[str, List[ToolParam]] = field(default_factory=dict)
    # derived values checked with bypass tricks: variable -> (scope | command | destination, parameter or None)
    classify: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)
    # from the agent's limits: the lists the mapping checks against
    scope_roots: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    destinations: List[str] = field(default_factory=list)
    # derived variable -> the policy tool values its rules speak about
    derived_tools: Dict[str, List[str]] = field(default_factory=dict)

    def var(self, name: str) -> Optional[VarSpec]:
        return next((v for v in self.variables if v.name == name), None)


def _kind(domain: str) -> Tuple[str, List[str], Optional[int], Optional[int]]:
    d = domain.strip()
    if d.startswith("{"):
        vals = re.findall(r'"([^"]*)"', d)
        if sorted(vals) == ["NO", "YES"]:
            return "flag", vals, None, None
        return "enum", vals, None, None
    if RANGE.match(d):
        lo, hi = [x.strip() for x in d.split("..")]
        return "range", [], int(float(lo)), int(float(hi))
    return "other", [], None, None


def build_spec(agent: Agent, policy: PolicyRef, limits=None) -> MappingSpec:
    from ..policy.limits import tools_of

    tools = tools_of(agent, limits)
    tv = tool_variable(policy, tools) or ("tool" if "tool" in policy.variables else None)
    spec = MappingSpec(agent.id, agent.display_name, agent_key(agent), policy.path, tv,
                       tool_params={t.name: list(t.params) for t in tools})
    if tv:
        vocab = policy.vocabulary.get(tv, [])
        for t in tools:
            server = next((re.search(r"MCP server '([^']+)'", e.detail or "") for e in t.evidence if e.detail), None)
            if t.name in vocab:
                spec.tool_table[t.name] = t.name
                if server and agent.kind == "assistant":
                    # assistants call MCP tools as mcp__<server>__<tool>
                    spec.tool_table[f"mcp__{server.group(1)}__{t.name}"] = t.name
            else:
                s = suggest(t.name, vocab)
                if s:
                    spec.tool_table[t.name] = s
                else:
                    spec.unmapped_tools.append(t.name)
    for var, domain in policy.variables.items():
        kind, vals, lo, hi = _kind(domain)
        vs = VarSpec(var, kind, vals, lo, hi)
        if var == tv:
            vs.source = "tool"
        elif var == "agent_id":
            vs.source = "constant"
            key = spec.agent_key
            vs.constant = key if key in vals else next((v for v in vals if norm(v) == norm(key)), vals[0] if vals else key)
        elif var in DERIVED:
            vs.source = "derived"
            vs.note = DERIVED[var]
            if var in CLASSIFY:
                spec.classify[var] = (CLASSIFY[var], None)
        else:
            for t in tools:
                names = [p.name for p in t.params]
                hit = var if var in names else suggest(var, names)
                if hit:
                    vs.params[t.name] = next(p for p in t.params if p.name == hit)
            vs.source = "param" if vs.params else "context"
        spec.variables.append(vs)
    spec.scope_roots = list(limits.scope) if limits is not None and limits.scope else ([agent.project] if agent.project else [])
    spec.commands = list(limits.commands) if limits is not None else []
    spec.destinations = list(limits.destinations) if limits is not None else []
    for values in policy.rule_values.values():
        named = [v.split("=", 1)[1] for v in values if v.startswith(f"{tv}=")] if tv else []
        for var in NEUTRAL:
            if var in values or any(v.startswith(var + "=") for v in values):
                for t in named:
                    spec.derived_tools.setdefault(var, [])
                    if t not in spec.derived_tools[var]:
                        spec.derived_tools[var].append(t)
    return spec


def pick_param(params: List[ToolParam], rx: re.Pattern) -> Optional[str]:
    hit = next((p.name for p in params if rx.match(p.name)), None)
    if hit:
        return hit
    strings = [p.name for p in params if (p.type or "").lower() in ("str", "string")]
    return strings[0] if strings else None


KIND_PARAMS = {"command": COMMAND_PARAMS, "destination": DESTINATION_PARAMS, "scope": PATH_PARAMS}


def param_for(params: List[ToolParam], kind: str, explicit: Optional[str] = None) -> Optional[str]:
    """The tool parameter a derived value is computed from: the one named, else the generator's choice."""
    if explicit:
        return explicit  # named by the operator: used even when discovery did not see the parameter
    return pick_param(params, KIND_PARAMS[kind])


def apply_classify(spec: MappingSpec, items: List[str]) -> List[str]:
    """`VAR=scope|command|destination[:param]` from the command line; returns the problems."""
    problems = []
    for item in items or []:
        var, _, rest = item.partition("=")
        kind, _, param = rest.partition(":")
        var, kind, param = var.strip(), kind.strip(), param.strip()
        if kind not in KIND_PARAMS:
            problems.append(f"{item}: the kind must be scope, command or destination")
            continue
        v = spec.var(var)
        if v is None:
            problems.append(f"{item}: {var} is not a variable of the policy")
            continue
        v.source = "derived"
        v.note = v.note or f"{kind} check computed by the integration"
        spec.classify[var] = (kind, param or None)
    return problems
