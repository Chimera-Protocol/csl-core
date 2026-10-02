"""
Deterministic policy drafting from the inventory.

Real tool names only, an `agent_id` variable, ranges from tool schemas, and risk-class
rule templates:

    SPEND        amount ceiling + approval above a threshold (approval only if no amount)
    EXEC         command allowlist (or block with exec_mode="block")
    DESTRUCTIVE  approval
    IDENTITY     approval
    UNCLASSIFIED approval (unknown counts as sensitive)
    EXTERNAL     destination allowlist
    WRITE        target must be in scope
    READ         no rule: the tool enum itself is the allowlist (unknown tools map to a block)

Agent exemptions become `agent_id != "..."` in every WHEN; tool exemptions drop that
tool's rules. Nothing here is ever active until it passes the gate and the operator
activates it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .. import VENOM_VERSION
from ..model import Agent, Exemption, Tool, ToolParam

AMOUNT_NAMES = re.compile(r"(amount|value|price|total|sum|cents|quantity|qty|budget|cost)", re.I)
NUMERIC = {"int", "integer", "float", "number"}
DEFAULT_CEILING = 1000


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "agent"


def ident(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return out if out and not out[0].isdigit() else f"t_{out}"


def domain_name(agent: Agent) -> str:
    return "Venom" + "".join(p.capitalize() for p in re.split(r"[^a-zA-Z0-9]+", agent.display_name) if p)


def agent_key(agent: Agent) -> str:
    """The value the integration passes as agent_id for this agent."""
    return slug(agent.display_name)


@dataclass
class Rule:
    name: str
    when: str  # "ALWAYS" or a condition
    then: str
    comment: str
    tool: Optional[str] = None


@dataclass
class Draft:
    agent_id: str
    name: str  # file stem
    text: str
    variables: Dict[str, str] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # tools without a rule, with the reason


def _q(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def _amount_param(tool: Tool) -> Optional[ToolParam]:
    nums = [p for p in tool.params if (p.type or "").lower() in NUMERIC]
    named = [p for p in nums if AMOUNT_NAMES.search(p.name)]
    return (named or nums or [None])[0]


def _range(p: ToolParam) -> Tuple[int, int]:
    lo = int(p.minimum) if isinstance(p.minimum, (int, float)) else 0
    hi = int(p.maximum) if isinstance(p.maximum, (int, float)) else 1_000_000
    return lo, max(hi, lo + 1)


def _exempt_clause(exempt_keys: Sequence[str]) -> str:
    return "".join(f' AND agent_id != {_q(k)}' for k in exempt_keys)


def rules_for(tool: Tool, variables: Dict[str, str], exempt_keys: Sequence[str], exec_mode: str = "allowlist") -> Tuple[List[Rule], Optional[str]]:
    """Template rules for one tool; adds needed variables. Returns (rules, reason if none)."""
    t = tool.name
    base = ident(t)
    cond = f"tool == {_q(t)}{_exempt_clause(exempt_keys)}"
    cls = tool.risk_class
    rules: List[Rule] = []
    if cls == "READ":
        return [], "READ: allowed by the tool enum"
    if cls == "SPEND":
        p = _amount_param(tool)
        if p is not None:
            var = p.name if p.name not in variables or variables[p.name] == f"{_range(p)[0]}..{_range(p)[1]}" else f"{base}_{p.name}"
            lo, hi = _range(p)
            variables[var] = f"{lo}..{hi}"
            ceiling = min(hi, DEFAULT_CEILING) if hi > lo else DEFAULT_CEILING
            approve_over = max(lo, ceiling // 10)
            variables["approval"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_ceiling", cond, f"{var} <= {ceiling}",
                              f"{t} moves money: hard ceiling per call (edit the limit to your needs)", t))
            rules.append(Rule(f"{base}_approval_over_{approve_over}", f"{cond} AND {var} > {approve_over}", 'approval MUST BE "YES"',
                              f"{t} above {approve_over} needs a human approval", t))
        else:
            variables["approval"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_needs_approval", cond, 'approval MUST BE "YES"', f"{t} moves money: every call needs approval", t))
    elif cls == "EXEC":
        if exec_mode == "block":
            rules.append(Rule(f"{base}_blocked", "ALWAYS" if not exempt_keys else _exempt_clause(exempt_keys)[5:],
                              f"tool MUST NOT BE {_q(t)}", f"{t} runs commands: blocked", t))
        else:
            variables["command_allowlisted"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_allowlist", cond, 'command_allowlisted MUST BE "YES"',
                              f"{t} runs commands: only allowlisted commands", t))
    elif cls in ("DESTRUCTIVE", "IDENTITY", "UNCLASSIFIED"):
        variables["approval"] = '{"YES", "NO"}'
        why = {"DESTRUCTIVE": "deletes or irreversibly changes", "IDENTITY": "changes credentials or permissions",
               "UNCLASSIFIED": "could not be classified, so it counts as sensitive"}[cls]
        rules.append(Rule(f"{base}_needs_approval", cond, 'approval MUST BE "YES"', f"{t} {why}: needs approval", t))
    elif cls == "EXTERNAL":
        variables["destination_allowlisted"] = '{"YES", "NO"}'
        rules.append(Rule(f"{base}_destination_allowlist", cond, 'destination_allowlisted MUST BE "YES"',
                          f"{t} sends data out: only allowlisted destinations", t))
    elif cls == "WRITE":
        variables["target_in_scope"] = '{"YES", "NO"}'
        rules.append(Rule(f"{base}_in_scope", cond, 'target_in_scope MUST BE "YES"',
                          f"{t} writes: only inside the agent's scope", t))
    return rules, None


def render(domain: str, policy_id: str, variables: Dict[str, str], rules: List[Rule], header: List[str]) -> str:
    lines = [f"// {h}" for h in header]
    lines += [
        "", "CONFIG {", "  ENFORCEMENT_MODE: BLOCK", "  CHECK_LOGICAL_CONSISTENCY: TRUE", "  ENABLE_FORMAL_VERIFICATION: FALSE",
        "  ENABLE_CAUSAL_INFERENCE: FALSE", '  INTEGRATION: "native"', f"  POLICY_ID: {_q(policy_id)}", '  POLICY_VERSION: "1"', "}",
        "", f"DOMAIN {domain} {{", "  VARIABLES {",
    ]
    width = max((len(k) for k in variables), default=0) + 1
    for k, v in variables.items():
        lines.append(f"    {(k + ':').ljust(width)} {v}")
    lines.append("  }")
    for r in rules:
        lines += ["", f"  // {r.comment}", f"  STATE_CONSTRAINT {r.name} {{"]
        lines.append("    ALWAYS True" if r.when == "ALWAYS" else f"    WHEN {r.when}")
        lines += [f"    THEN {r.then}", "  }"]
    lines += ["}", ""]
    return "\n".join(lines)


def draft_for(agent: Agent, exemptions: Sequence[Exemption] = (), exec_mode: str = "allowlist",
              agent_ids: Optional[Dict[str, str]] = None) -> Draft:
    """
    Draft a policy for one agent. `exemptions` are the approved exemptions; agent
    exemptions for *other* agents that share this policy are encoded as agent_id != ...
    (agent_ids maps agent id -> agent key for those).
    """
    key = agent_key(agent)
    agent_ids = agent_ids or {}
    exempt_keys = sorted({agent_ids.get(e.agent, slug(e.agent)) for e in exemptions
                          if e.scope == "agent" and e.status == "approved" and e.agent not in (agent.id, agent.display_name)})
    tool_exempt = {e.tool for e in exemptions if e.scope == "tool" and e.status == "approved" and e.agent in ("*", agent.id, agent.display_name)}
    tools = [t for t in agent.tools if not t.name.endswith("/*")]
    variables: Dict[str, str] = {
        "agent_id": "{" + ", ".join(_q(k) for k in [key] + exempt_keys) + "}",
        "tool": "{" + ", ".join(_q(t.name) for t in tools) + "}",
    }
    rules: List[Rule] = []
    skipped: List[str] = []
    for t in sorted(tools, key=lambda t: t.name.lower()):
        if t.name in tool_exempt:
            skipped.append(f"{t.name}: exempted by operator")
            continue
        rs, why = rules_for(t, variables, exempt_keys, exec_mode)
        rules += rs
        if why:
            skipped.append(f"{t.name}: {why}")
    for t in agent.tools:
        if t.name.endswith("/*"):
            skipped.append(f"{t.name}: tools of this MCP server are not enumerated yet (0.6.2 --probe)")
    header = [
        f"Draft policy for {agent.display_name} ({agent.id})",
        f"Generated by CSL-Core Venom {VENOM_VERSION} from the discovered tools. Review every limit before activating.",
        f"The integration passes agent_id = \"{key}\"." + (f" Exempted agents: {', '.join(exempt_keys)}." if exempt_keys else ""),
    ]
    text = render(domain_name(agent), f"venom.{key}", variables, rules, header)
    return Draft(agent.id, key, text, variables, rules, skipped)


def needs_policy(agent: Agent) -> bool:
    """First install: a draft for every non-exempt agent with WRITE-or-higher tools."""
    if agent.exempt is not None and agent.exempt.status == "approved":
        return False
    return any(t.risk_class != "READ" and t.coverage != "exempt" for t in agent.tools if not t.name.endswith("/*"))


# ---------------------------------------------------------------------------
# text edits on existing policies: extend and fix
# ---------------------------------------------------------------------------

def _var_line_re(var: str) -> re.Pattern:
    return re.compile(rf"(?m)^(\s*){re.escape(var)}(\s*):(\s*)(.+?)\s*$")


def extend_text(text: str, existing_vars: Dict[str, str], tool_var: Optional[str], tools: List[Tool],
                exempt_keys: Sequence[str] = ()) -> Tuple[str, List[Rule]]:
    """Add template rules (and variables) for `tools` to an existing policy's text."""
    variables = dict(existing_vars)
    tv = tool_var or "tool"
    new_rules: List[Rule] = []
    for t in tools:
        rs, _ = rules_for(t, variables, exempt_keys)
        if tv != "tool":
            for r in rs:
                r.when = r.when.replace("tool ==", f"{tv} ==")
                r.then = r.then.replace("tool MUST", f"{tv} MUST")
        new_rules += rs
    # tool enum: add names that are missing
    m = _var_line_re(tv).search(text)
    if m and m.group(4).startswith("{"):
        vals = re.findall(r'"([^"]*)"', m.group(4))
        for t in tools:
            if t.name not in vals:
                vals.append(t.name)
        text = text[: m.start(4)] + "{" + ", ".join(_q(v) for v in vals) + "}" + text[m.end(4):]
    elif not m:
        variables[tv] = "{" + ", ".join(_q(t.name) for t in tools) + "}"
    added = {k: v for k, v in variables.items() if k not in existing_vars}
    if added:
        vm = re.search(r"VARIABLES\s*\{[^\n]*\n", text)
        if vm:
            ins = "".join(f"    {k}: {v}\n" for k, v in added.items())
            text = text[: vm.end()] + ins + text[vm.end():]
    body = "".join(
        f"\n  // {r.comment} (added by cslcore policy extend)\n  STATE_CONSTRAINT {r.name} {{\n"
        + ("    ALWAYS True\n" if r.when == "ALWAYS" else f"    WHEN {r.when}\n") + f"    THEN {r.then}\n  }}\n"
        for r in new_rules
    )
    end = text.rstrip().rfind("}")
    text = text[:end].rstrip("\n") + "\n" + body + text[end:]
    return text, new_rules


def fix_text(text: str, renames: Dict[str, str], var_renames: Dict[str, str]) -> str:
    """Apply drift suggestions: enum value renames ("OLD" -> "new") and variable renames."""
    for old, new in renames.items():
        text = text.replace(f'"{old}"', f'"{new}"')
    for old, new in var_renames.items():
        parts = re.split(r'("[^"]*"|//[^\n]*)', text)
        parts = [p if p.startswith('"') or p.startswith("//") else re.sub(rf"\b{re.escape(old)}\b", new, p) for p in parts]
        text = "".join(parts)
    return text
