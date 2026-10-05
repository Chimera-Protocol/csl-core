"""
Limits: what each agent may do, in the operator's own numbers and lists, and the policy made from them.

    every tool gets a kind, and each kind its standard rule (the "standard" profile):

    spend      money moves freely up to `allow_up_to`; above it a human approval is needed; never
               above `never_above` (equal values: no approval band)
    shell      ordinary commands run; remote code, destructive and privileged commands, reading or
               sending secrets and persistence are stopped (chimera_core.actions.command_class).
               The "strict" profile allows only the commands listed in `commands`
    sql        reading runs; writing needs approval; DROP, TRUNCATE, ALTER and DELETE or UPDATE without
               WHERE are stopped
    write      only inside `scope` (the agent's own folder by default), never credentials or startup files
    read       anything except credentials and keys
    send       to anyone (recorded); "strict": only the places listed in `destinations`
    publish    public posts and bulk sends need approval
    destroy    deleting or irreversible changes need approval
    identity   credentials and permissions need approval
    other      tools that could not be classified: allowed and recorded ("strict": approval)

    each kind can be overridden per tool with `decide`: allow | approval | block

Approval comes from the caller: guard.check(tool, args, {"approval": "YES"}) after a human said yes.
Where nothing passes it (a wired decorator, a Claude Code hook), "needs approval" means stopped.

Limits live in the workspace state ("limits", per agent key); `cslcore limits` and setup edit them,
and the policy and mapping are made again from them, through the same gate as every policy.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..model import Agent, Tool, ToolParam
from .draft import Rule, _amount_param, _exempt_clause, _q, agent_key, domain_name, ident, render

KINDS = ("spend", "shell", "sql", "write", "read", "send", "publish", "destroy", "identity", "other")
DEFAULT_ALLOW_UP_TO = 100  # without the operator's own numbers: small amounts only
DEFAULT_NEVER_ABOVE = 1_000
AMOUNT_DOMAIN = 10 ** 12  # any amount maps; the rules decide
SQL_HINT = re.compile(r"(sql|query|database|db_|_db\b|select|statement)", re.I)
SQL_PARAMS = re.compile(r"^(query|sql|statement|stmt)$", re.I)
PATH_PARAMS = re.compile(r"(path|file|filename|dir|directory|target|dest|notebook)", re.I)
PUBLISH = re.compile(r"(post|publish|tweet|toot|broadcast|campaign|blast|bulk|announce|newsletter|press_release)", re.I)


@dataclass
class ToolLimit:
    kind: str
    allow_up_to: Optional[int] = None
    never_above: Optional[int] = None
    amount_param: Optional[str] = None
    decide: Optional[str] = None  # allow | approval | block: overrides the kind's standard rule
    paths: Optional[bool] = None  # a write tool: whether it takes a file path (its rule needs one)
    # any numeric parameter: param -> [free up to, never above] (a list parameter: its length)
    numbers: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class Limits:
    agent: str
    profile: str = "standard"  # standard | strict
    scope: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    destinations: List[str] = field(default_factory=list)
    tools: Dict[str, ToolLimit] = field(default_factory=dict)
    extra_tools: List[Dict[str, Any]] = field(default_factory=list)  # {"name", "risk", "amount_param"}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tools"] = {k: {kk: vv for kk, vv in asdict(v).items() if vv is not None} for k, v in self.tools.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Limits":
        tools = {k: ToolLimit(**{kk: vv for kk, vv in v.items() if kk in ToolLimit.__dataclass_fields__})
                 for k, v in (d.get("tools") or {}).items()}
        return cls(d["agent"], d.get("profile", "standard"), list(d.get("scope") or []), list(d.get("commands") or []),
                   list(d.get("destinations") or []), tools, list(d.get("extra_tools") or []))


def _has(tool: Tool, rx: re.Pattern) -> Optional[str]:
    return next((p.name for p in tool.params if rx.search(p.name)), None)


UNKNOWN_ARGS = ("mcp_server", "builtin")  # tools whose arguments the scan does not know (a catalog)


def judged_by_path(tool: Tool) -> bool:
    """Whether a write tool's rule can look at a file path: it takes one, or its arguments are not
    known (an MCP server's tool from the catalog: the real call carries them). A tool whose
    arguments are known and hold no path (deploy(service, version)) is not judged by a path."""
    return bool(_has(tool, PATH_PARAMS)) or (not tool.params and tool.source in UNKNOWN_ARGS)


def kind_of(tool: Tool) -> str:
    """The standard kind of a tool, from its risk class, name and parameters."""
    cls = tool.risk_class
    if cls == "SPEND":
        return "spend" if _amount_param(tool) is not None or not SQL_HINT.search(tool.name) else "sql"
    if cls == "EXEC":
        return "sql" if SQL_HINT.search(tool.name) or _has(tool, SQL_PARAMS) else "shell"
    if cls == "WRITE":
        return "write"  # its path arguments decide; none at all: refused (fail closed)
    if cls == "READ":
        return "sql" if SQL_HINT.search(tool.name) and (not tool.params or _has(tool, SQL_PARAMS)) else "read"
    if cls == "EXTERNAL":
        return "publish" if PUBLISH.search(tool.name) else "send"
    if cls == "DESTRUCTIVE":
        return "sql" if SQL_HINT.search(tool.name) and _has(tool, SQL_PARAMS) else "destroy"
    if cls == "IDENTITY":
        return "identity"
    return "other"


def tools_of(agent: Agent, limits: Optional[Limits] = None) -> List[Tool]:
    """The agent's discovered tools plus the ones the operator added."""
    tools = [t for t in agent.tools if not t.name.endswith("/*")]
    for extra in (limits.extra_tools if limits else []):
        if any(t.name == extra["name"] for t in tools):
            continue
        params = [ToolParam(extra["amount_param"], "int")] if extra.get("amount_param") else []
        tools.append(Tool(extra["name"], "operator", params=params, risk_class=extra.get("risk", "UNCLASSIFIED"),
                          risk_reason="added by the operator"))
    return tools


def defaults(agent: Agent, scope: List[str], existing: Optional[Limits] = None) -> Limits:
    """Limits for an agent: what the operator set before, completed with the standard defaults."""
    lim = existing or Limits(agent_key(agent), scope=list(scope))
    if not lim.scope:
        lim.scope = list(scope)
    for t in tools_of(agent, lim):
        if t.name in lim.tools:
            tl = lim.tools[t.name]
            if tl.kind == "write" and tl.paths is None:  # limits saved before 0.6.9
                tl.paths = judged_by_path(t)
            continue
        kind = kind_of(t)
        tl = ToolLimit(kind, paths=judged_by_path(t) if kind == "write" else None)
        if kind == "spend":
            p = _amount_param(t)
            tl.amount_param = p.name if p is not None else None
            tl.allow_up_to, tl.never_above = DEFAULT_ALLOW_UP_TO, DEFAULT_NEVER_ABOVE
        lim.tools[t.name] = tl
    return lim


# ---------------------------------------------------------------------------
# the policy made from limits
# ---------------------------------------------------------------------------

def rules_from(tool: Tool, tl: ToolLimit, lim: Limits, variables: Dict[str, str], exempt_keys=()) -> List[Rule]:
    t = tool.name
    base = ident(t)
    cond = f"tool == {_q(t)}{_exempt_clause(exempt_keys)}"
    if tl.decide == "allow":
        return []
    if tl.decide == "block":
        return []  # left out of the tool list: the mapping refuses every call (policy_text)
    approval = tl.decide == "approval"
    rules: List[Rule] = []
    for param, (lo_n, hi_n) in sorted((tl.numbers or {}).items()):
        if tl.kind == "spend" and param == tl.amount_param:
            continue  # the money rule below covers it
        var_name = param if variables.get(param) in (None, f"0..{AMOUNT_DOMAIN}") else f"{base}_{param}"
        variables[var_name] = f"0..{AMOUNT_DOMAIN}"
        rules.append(Rule(f"{base}_{ident(param)}_max", cond, f"{var_name} <= {int(hi_n)}",
                          f"{t}: {param} never above {int(hi_n):,}", t))
        if lo_n < hi_n:
            variables["approval"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_{ident(param)}_over_{int(lo_n)}", f"{cond} AND {var_name} > {int(lo_n)}",
                              'approval MUST BE "YES"', f"{t}: {param} above {int(lo_n):,} needs a human approval", t))

    def needs_approval(why: str, when: str = "") -> Rule:
        variables["approval"] = '{"YES", "NO"}'
        return Rule(f"{base}_needs_approval" if not when else f"{base}_{when}_needs_approval",
                    cond + (f" AND {when}" if when and " " in when else ""), 'approval MUST BE "YES"', why, t)

    kind = tl.kind
    if approval:
        # "approval" replaces what the tool's kind would check (paths, commands, queries, destinations)
        # with one rule: a person approves each call. The operator's own numbers stay: never above.
        if kind == "spend" and tl.amount_param:
            var = tl.amount_param
            var_name = var if variables.get(var) in (None, f"0..{AMOUNT_DOMAIN}") else f"{base}_{var}"
            variables[var_name] = f"0..{AMOUNT_DOMAIN}"
            hi = int(tl.never_above if tl.never_above is not None else DEFAULT_NEVER_ABOVE)
            rules.append(Rule(f"{base}_ceiling", cond, f"{var_name} <= {hi}",
                              f"{t} moves money: never above {hi:,} in one call", t))
        rules.append(needs_approval(f"{t}: every call needs a person's approval", "always"))
        return rules
    if kind == "spend":
        var = tl.amount_param
        if var:
            var_name = var if variables.get(var) in (None, f"0..{AMOUNT_DOMAIN}") else f"{base}_{var}"
            variables[var_name] = f"0..{AMOUNT_DOMAIN}"
            hi = int(tl.never_above if tl.never_above is not None else DEFAULT_NEVER_ABOVE)
            lo = int(tl.allow_up_to if tl.allow_up_to is not None else min(hi, DEFAULT_ALLOW_UP_TO))
            rules.append(Rule(f"{base}_ceiling", cond, f"{var_name} <= {hi}",
                              f"{t} moves money: never above {hi:,} in one call", t))
            if lo < hi:
                variables["approval"] = '{"YES", "NO"}'
                rules.append(Rule(f"{base}_approval_over_{lo}", f"{cond} AND {var_name} > {lo}", 'approval MUST BE "YES"',
                                  f"{t} above {lo:,} needs a human approval", t))
        else:
            rules.append(needs_approval(f"{t} moves money and its amount is not visible: every call needs approval"))
    elif kind == "shell":
        if lim.profile == "strict":
            variables["command_allowlisted"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_allowlist", cond, 'command_allowlisted MUST BE "YES"',
                              f"{t} runs commands: only the listed ones (strict)", t))
        else:
            variables["command_class"] = '{"OK", "REMOTE_EXEC", "DESTRUCTIVE", "PRIVILEGE", "SECRETS", "EXFIL", "PERSISTENCE", "UNREADABLE"}'
            rules.append(Rule(f"{base}_safe_commands", cond, 'command_class MUST BE "OK"',
                              f"{t} runs commands: no remote code, destruction, privilege, secrets, exfiltration or persistence", t))
    elif kind == "sql":
        variables["sql_class"] = '{"READ", "WRITE", "DESTRUCTIVE", "UNREADABLE"}'
        variables["approval"] = '{"YES", "NO"}'
        rules.append(Rule(f"{base}_no_destructive_sql", cond, 'sql_class MUST NOT BE "DESTRUCTIVE"',
                          f"{t}: no DROP, TRUNCATE, ALTER, or DELETE / UPDATE without WHERE", t))
        rules.append(Rule(f"{base}_readable_sql", cond, 'sql_class MUST NOT BE "UNREADABLE"',
                          f"{t}: a query that cannot be read is not run", t))
        rules.append(Rule(f"{base}_sql_write_needs_approval", f'{cond} AND sql_class == "WRITE"', 'approval MUST BE "YES"',
                          f"{t}: writing to the database needs approval", t))
    elif kind in ("write", "read"):
        # the path rule needs a path to look at: a tool without one (deploy(service, version)) is not
        # judged by where it writes; it runs and every call is recorded
        if judged_by_path(tool) or (kind == "read" and not tool.params):
            variables["path_class"] = '{"IN_SCOPE", "OUTSIDE", "SENSITIVE", "UNREADABLE"}'
            if kind == "write":
                rules.append(Rule(f"{base}_in_scope", cond, 'path_class MUST BE "IN_SCOPE"',
                                  f"{t} writes: only inside its own folder, never credentials or startup files", t))
            else:
                rules.append(Rule(f"{base}_no_secrets", cond, 'path_class MUST NOT BE "SENSITIVE"',
                                  f"{t} reads: anything except credentials and keys", t))
    elif kind == "send":
        if lim.profile == "strict" or lim.destinations:
            variables["destination_allowlisted"] = '{"YES", "NO"}'
            rules.append(Rule(f"{base}_destination_allowlist", cond, 'destination_allowlisted MUST BE "YES"',
                              f"{t} sends data out: only the listed destinations", t))
    elif kind in ("publish", "destroy", "identity") or (kind == "other" and (lim.profile == "strict" or approval)):
        why = {"publish": "publishes or sends in bulk", "destroy": "deletes or changes irreversibly",
               "identity": "changes credentials or permissions", "other": "could not be classified"}[kind]
        rules.append(needs_approval(f"{t} {why}: needs approval"))
    return rules


def policy_text(agent: Agent, lim: Limits, exempt_keys: Tuple[str, ...] = ()) -> Tuple[str, List[str]]:
    """The policy for an agent's limits, and notes for the operator (tools without a rule, and why)."""
    from .. import VENOM_VERSION

    key = agent_key(agent)
    # a tool the operator blocked is not in the tool list: its mapping refuses every call (a rule
    # "tool MUST NOT BE x" under "WHEN tool == x" could never be satisfied, and the gate rejects it)
    blocked = [t.name for t in tools_of(agent, lim) if (lim.tools.get(t.name) or ToolLimit("other")).decide == "block"]
    tools = [t for t in tools_of(agent, lim) if t.name not in blocked]
    variables: Dict[str, str] = {
        "agent_id": "{" + ", ".join(_q(k) for k in [key] + list(exempt_keys)) + "}",
        "tool": "{" + ", ".join(_q(t.name) for t in tools) + "}" if tools else '{"__none__"}',
    }
    rules: List[Rule] = []
    notes: List[str] = []
    for t in sorted(tools, key=lambda x: x.name.lower()):
        tl = lim.tools.get(t.name) or ToolLimit(kind_of(t))
        rs = rules_from(t, tl, lim, variables, exempt_keys)
        rules += rs
        if not rs:
            notes.append(f"{t.name}: {tl.kind}, allowed and recorded")
    notes += [f"{name}: blocked by the operator" for name in blocked]
    header = [
        f"Policy for {agent.display_name} ({agent.id}), made from its limits ({lim.profile} profile).",
        f"Generated by CSL-Core Venom {VENOM_VERSION}. Change the numbers with: cslcore limits --agent {key}",
        f"The integration passes agent_id = \"{key}\".",
    ]
    if blocked:
        header.append(f"Blocked by the operator, so not in the tool list (every call is refused): {', '.join(sorted(blocked))}")
    return render(domain_name(agent), f"venom.{key}", variables, rules, header), notes


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def load(ws, key: str) -> Optional[Limits]:
    data = (ws.load_state().get("limits") or {}).get(key)
    return Limits.from_dict(data) if data else None


def save(ws, lim: Limits) -> None:
    state = ws.load_state()
    state.setdefault("limits", {})[lim.agent] = lim.to_dict()
    ws.save_state(state)


def describe(lim: Limits) -> List[Tuple[str, str, str]]:
    """(tool, kind, what happens) rows for screens."""
    rows = []
    for name, tl in sorted(lim.tools.items()):
        if tl.decide == "block":
            what = "blocked"
        elif tl.decide == "allow":
            what = "runs, every call recorded"
        elif tl.kind == "spend" and tl.amount_param:
            lo, hi = tl.allow_up_to, tl.never_above
            what = (f"up to {lo:,} freely, up to {hi:,} with approval, never above {hi:,}" if lo < hi
                    else f"up to {hi:,}, never above")
        elif tl.kind == "spend":
            what = "every call needs approval (no amount visible)"
        elif tl.kind == "shell":
            what = ("only listed commands" if lim.profile == "strict"
                    else "ordinary commands; stops remote code, destruction, sudo, secrets, exfiltration, persistence")
        elif tl.kind == "sql":
            what = "reads run; writes need approval; DROP / TRUNCATE / unbounded DELETE stopped"
        elif tl.kind == "write" and not tl.paths:
            what = "runs, every call recorded (no file path to check)"
        elif tl.kind == "write":
            what = "inside its folder only; never credentials or startup files"
        elif tl.kind == "read":
            what = "anything except credentials and keys"
        elif tl.kind == "send":
            what = "only listed destinations" if (lim.profile == "strict" or lim.destinations) else "anyone, recorded"
        elif tl.kind == "other":
            what = "needs approval" if lim.profile == "strict" else "runs, every call recorded"
        else:
            what = "needs approval"
        if tl.decide == "approval":
            what = "every call needs a person's approval" + (
                f"; never above {tl.never_above:,}" if tl.kind == "spend" and tl.amount_param and tl.never_above else "")
        for param, (lo_n, hi_n) in sorted((tl.numbers or {}).items()):
            if tl.kind == "spend" and param == tl.amount_param:
                continue
            what += (f"; {param} up to {lo_n:,} freely, never above {hi_n:,}" if lo_n < hi_n
                     else f"; {param} never above {hi_n:,}")
        rows.append((name, tl.kind, what))
    return rows


# ---------------------------------------------------------------------------
# from the command line: --limit, --add-tool, --profile
# ---------------------------------------------------------------------------

RISKS = {"spend": "SPEND", "shell": "EXEC", "exec": "EXEC", "sql": "EXEC", "write": "WRITE", "read": "READ",
         "send": "EXTERNAL", "external": "EXTERNAL", "publish": "EXTERNAL", "destroy": "DESTRUCTIVE",
         "destructive": "DESTRUCTIVE", "identity": "IDENTITY", "other": "UNCLASSIFIED", "unclassified": "UNCLASSIFIED"}


class LimitError(ValueError):
    pass


def parse_range(text: str) -> Tuple[int, int]:
    """'100000..300000' (free up to, never above), '300000' or '..300000' (both the same), '100k..300k', '1m'."""
    def num(s: str) -> int:
        s = s.strip().lower().replace("_", "").replace(",", "")
        mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(s[-1:], 1)
        if mult != 1:
            s = s[:-1]
        try:
            value = float(s)
        except ValueError:
            raise LimitError(f"not a number: {s!r}")
        if value < 0:
            raise LimitError("an amount cannot be negative")
        return int(value * mult)

    parts = text.split("..")
    if len(parts) == 1:
        v = num(parts[0])
        return v, v
    if len(parts) != 2:
        raise LimitError(f"expected FREE..MAX, got {text!r}")
    lo, hi = num(parts[0]), num(parts[1])
    if lo > hi:
        raise LimitError(f"the free amount ({lo:,}) is above the maximum ({hi:,})")
    return lo, hi


def apply_flags(lim: Limits, key: str, limit_flags=(), add_flags=(), profile: Optional[str] = None,
                names: Tuple[str, ...] = ()) -> List[str]:
    """Apply --limit AGENT.TOOL=FREE..MAX (or TOOL=..., for any agent that has it), --add-tool
    AGENT:NAME:RISK[:AMOUNT_PARAM] and --profile. Returns what was applied, for the screen."""
    me = {key, *names}
    done: List[str] = []
    if profile:
        lim.profile = profile
        done.append(f"profile {profile}")
    for spec in add_flags or []:
        parts = spec.split(":")
        if len(parts) < 3:
            raise LimitError(f"--add-tool wants AGENT:NAME:RISK[:AMOUNT_PARAM], got {spec!r}")
        agent, name, risk = parts[0], parts[1], parts[2].lower()
        if agent not in me:
            continue
        if risk not in RISKS:
            raise LimitError(f"unknown risk {risk!r}; one of {', '.join(sorted(RISKS))}")
        extra = {"name": name, "risk": RISKS[risk]}
        if len(parts) > 3 and parts[3]:
            extra["amount_param"] = parts[3]
        lim.extra_tools = [e for e in lim.extra_tools if e["name"] != name] + [extra]
        lim.tools.pop(name, None)
        done.append(f"added {name} ({risk})")
    for spec in limit_flags or []:
        if "=" not in spec:
            raise LimitError(f"--limit wants [AGENT.]TOOL[.PARAM]=FREE..MAX, got {spec!r}")
        target, rng = spec.split("=", 1)
        agent, tool, param = _target(target, lim, me)
        if agent is False:
            continue
        lo, hi = parse_range(rng[2:] if rng.startswith("..") else rng)  # "..1000": up to 1000, never above
        tl = lim.tools.get(tool)
        if tl is None:
            if agent:
                raise LimitError(f"{key} has no tool {tool!r}; add it with --add-tool {key}:{tool}:spend:amount")
            continue
        if param is None or (tl.kind == "spend" and param == tl.amount_param):
            if tl.kind != "spend" and not tl.amount_param:
                raise LimitError(f"{tool} has no amount; name the parameter: {tool}.PARAM=FREE..MAX")
            tl.kind = "spend"
            tl.allow_up_to, tl.never_above = lo, hi
            done.append(f"{tool}: free up to {lo:,}, never above {hi:,}")
        else:
            tl.numbers = dict(tl.numbers or {})
            tl.numbers[param] = [lo, hi]
            done.append(f"{tool}.{param}: free up to {lo:,}, never above {hi:,}")
    return done


def _target(target: str, lim: Limits, me) -> Tuple[Any, str, Optional[str]]:
    """AGENT.TOOL.PARAM, TOOL.PARAM, AGENT.TOOL or TOOL -> (agent or None, tool, param or None);
    agent False when the flag names another agent."""
    parts = target.split(".")
    if len(parts) >= 3:
        agent, tool, param = ".".join(parts[:-2]), parts[-2], parts[-1]
        return (agent if agent in me else False), tool, param
    if len(parts) == 2:
        if parts[0] in lim.tools:
            return None, parts[0], parts[1]
        return (parts[0] if parts[0] in me else False), parts[1], None
    return None, parts[0], None
