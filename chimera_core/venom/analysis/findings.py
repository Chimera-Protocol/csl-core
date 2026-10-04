"""
Deterministic findings V01-V16.

Each rule is a small pure function over the inventory. A rule whose inputs come from a
layer that did not run returns nothing and is reported as "not evaluated".
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from ..model import Agent, Evidence, Finding, Inventory, SEVERITIES
from .coverage import blocking_drift

INBOUND = {"inbound_http": "HTTP"}  # "inbound HTTP", not "inbound inbound http"

BROAD_CREDENTIALS = {"cloud_root", "org_token", "payment", "database_url"}
STATE_CHANGING = {"WRITE", "EXTERNAL", "SPEND", "DESTRUCTIVE", "EXEC", "IDENTITY"}


def _f(fid: str, sev: str, summary: str, a: Optional[Agent] = None, tool: Optional[str] = None,
       evidence: Optional[List[Evidence]] = None, rec: Optional[str] = None) -> Finding:
    return Finding(id=fid, rule=RULES[fid][0], severity=sev, summary=summary, agent_id=a.id if a else None,
                   tool=tool, evidence=evidence or (a.evidence[:2] if a else []), recommendation=rec)


def v01(inv: Inventory, ctx) -> List[Finding]:
    return [_f("V01", "high", f"{a.display_name} runs as root", a,
               rec="run the agent as an unprivileged service user")
            for a in inv.agents if a.access.elevated]


def v02(inv, ctx):
    out = []
    for a in inv.agents:
        exec_tools = [t for t in a.tools if t.risk_class == "EXEC" and t.coverage == "unguarded"]
        if exec_tools:
            names = ", ".join(t.name for t in exec_tools[:3])
            out.append(_f("V02", "high", f"{a.display_name} can execute commands without a guard ({names})", a,
                          exec_tools[0].name, rec="guard these tools with a policy (block or allowlist)"))
    return out


def v03(inv, ctx):
    return [_f("V03", "high", f"permission prompts are disabled for {a.display_name}", a,
               rec="remove bypassPermissions / skip flags, or wire a cslcore PreToolUse hook")
            for a in inv.agents if a.kind in ("assistant", "unmanaged") and a.access.permission_mode == "bypass"]


def v04(inv, ctx):
    out = []
    for a in inv.agents:
        ext = [t for t in a.triggers if t.type in ("inbound_http", "messaging", "email")]
        risky = [t for t in a.tools if t.risk_class in ("WRITE", "EXTERNAL", "SPEND", "DESTRUCTIVE") and t.coverage == "unguarded"]
        if ext and risky:
            out.append(_f("V04", "high", f"{a.display_name}: inbound {INBOUND.get(ext[0].type, ext[0].type)} ({ext[0].schedule}) "
                          f"reaches {risky[0].risk_class} tool {risky[0].name} unguarded", a, risky[0].name,
                          rec="put an approval rule or a guard between external input and state-changing tools"))
    return out


def v05(inv, ctx):
    out = []
    for a in inv.agents:
        if not any(t.type == "time" for t in a.triggers):
            continue
        risky = [t for t in a.tools if t.risk_class in STATE_CHANGING and t.coverage in ("unguarded", "wired_no_rule")]
        if risky:
            out.append(_f("V05", "medium", f"scheduled agent {a.display_name} changes state with no approval step ({risky[0].name})",
                          a, risky[0].name, rec="add an approval rule for scheduled state changes"))
    return out


def v06(inv, ctx):
    out = []
    for a in inv.agents:
        broad = [c for c in a.access.credentials if c.kind in BROAD_CREDENTIALS]
        if broad:
            names = ", ".join(sorted({c.name for c in broad})[:3])
            out.append(_f("V06", "medium", f"{a.display_name} can reach broad credentials: {names}", a,
                          evidence=[Evidence("config", broad[0].file, None, "names only")],
                          rec="scope credentials to what the agent needs"))
    return out


def v07(inv, ctx):
    out = []
    for a in inv.agents:
        broad = [r for r in a.access.fs_roots if r.rstrip("/") in ("", ctx["home"].rstrip("/"))]
        if broad:
            out.append(_f("V07", "medium", f"{a.display_name} has filesystem access rooted at {broad[0] or '/'}", a,
                          rec="root the filesystem server at the project folder"))
    return out


def v08(inv, ctx):
    from ..layers.processes import is_loopback
    out = []
    for l in ctx.get("listeners", []):
        public = [x for x in l.addresses if not is_loopback(x)]
        if l.is_mcp and public:
            out.append(Finding("V08", RULES["V08"][0], "medium", f"MCP server listens on {public[0]} (pid {l.pid})",
                               evidence=[Evidence("runtime", f"pid {l.pid}", None, l.args[:100])],
                               recommendation="bind to 127.0.0.1 or put authentication in front"))
    return out


def v09(inv, ctx):
    out = []
    for a in inv.agents:
        for t in a.tools:
            if t.risk_class in ("DESTRUCTIVE", "SPEND") and t.coverage in ("unguarded", "wired_no_rule"):
                what = "moves money" if t.risk_class == "SPEND" else "deletes or changes irreversibly"
                out.append(_f("V09", "high", f"{a.display_name}: {t.name} {what} and no rule covers it",
                              a, t.name, rec="add a rule for this tool (block, limit or approval)"))
    return out


def v10(inv, ctx):
    out = []
    for d in blocking_drift(inv.drift):
        if d.kind == "unknown_value":
            text = f"policy value \"{d.value}\" in {d.policy}.{d.variable} matches no tool" + (f" (closest: {d.suggestion})" if d.suggestion else "")
        else:
            text = d.detail or f"coercion needed for {d.variable}"
        out.append(Finding("V10", RULES["V10"][0], "medium", text, agent_id=d.agent_id, tool=d.tool,
                           recommendation="cslcore policy fix" if d.suggestion else "map the value explicitly (cslcore map)"))
    return out


def v11(inv, ctx):
    out = []
    for agent_id, res in (ctx.get("state", {}).get("mapping_tests") or {}).items():
        if res.get("fail_open"):
            out.append(Finding("V11", RULES["V11"][0], "medium",
                               f"mapping test for {agent_id} found {res['fail_open']} fail-open case(s)", agent_id=agent_id,
                               recommendation=f"cslcore map --agent {agent_id} --test"))
    return out


def v12(inv, ctx):
    wired = {p for paths in ctx["links"].values() for p in paths}
    out = []
    for p in inv.policies:
        # only the workspace's active policies: example or vendored .csl files elsewhere are not a gap
        if p.error or p.status != "active" or p.path in wired:
            continue
        out.append(Finding("V12", RULES["V12"][0], "low", f"policy {p.policy_id or p.domain} is not wired to any agent",
                           evidence=[Evidence("governance", p.path)], recommendation="wire a guard or remove the policy"))
    return out


def v13(inv, ctx):
    return [Finding("V13", RULES["V13"][0], "low", f"exemption for {e.agent}{' / ' + e.tool if e.tool else ''} expired on {e.expires}",
                    agent_id=e.agent, recommendation="renew or remove it: cslcore exempt list")
            for e in ctx.get("expired", [])]


def v14(inv, ctx):
    return [_f("V14", "low", f"{a.display_name} is running but present in no configuration", a,
               rec="add it to the inventory or stop it") for a in inv.agents if a.kind == "unmanaged"]


def v15(inv, ctx):
    now: datetime = ctx["now"]
    out = []
    for a in inv.agents:
        if a.state == "running" or not any(t.type == "time" for t in a.triggers) or not a.runs.last_run:
            continue
        try:
            last = datetime.fromisoformat(a.runs.last_run.replace("Z", "+00:00"))
        except ValueError:
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=now.tzinfo)
        if (now - last).days > 30:
            out.append(_f("V15", "low", f"{a.display_name} has not run for {(now - last).days} days but its trigger is active", a,
                          rec="remove the trigger or check the agent"))
    return out


def v16(inv, ctx):
    now: datetime = ctx["now"]
    out = []
    for agent_id, mode in (ctx.get("state", {}).get("modes") or {}).items():
        since = mode.get("since") if isinstance(mode, dict) else None
        if isinstance(mode, dict) and mode.get("mode") == "log" and since:
            try:
                days = (now - datetime.fromisoformat(since)).days
            except ValueError:
                continue
            if days > 14:
                out.append(Finding("V16", RULES["V16"][0], "info", f"{agent_id} has been in log mode for {days} days",
                                   agent_id=agent_id, recommendation=f"review cslcore watch, then: cslcore mode --agent {agent_id} block"))
    return out


# id -> (rule name, function, layers required)
RULES: Dict[str, Tuple[str, Callable, Tuple[str, ...]]] = {
    "V01": ("agent_runs_as_root", v01, ("runtime",)),
    "V02": ("exec_without_guard", v02, ()),
    "V03": ("permission_prompts_disabled", v03, ("config",)),
    "V04": ("external_trigger_state_change", v04, ()),
    "V05": ("scheduled_state_change_no_approval", v05, ("triggers",)),
    "V06": ("broad_credentials", v06, ("config",)),
    "V07": ("filesystem_root_access", v07, ("config",)),
    "V08": ("mcp_listens_publicly", v08, ("runtime",)),
    "V09": ("destructive_or_spend_uncovered", v09, ()),
    "V10": ("vocabulary_drift", v10, ("policies",)),
    "V11": ("mapping_fail_open", v11, ()),
    "V12": ("policy_not_wired", v12, ("policies",)),
    "V13": ("exemption_expired", v13, ()),
    "V14": ("unmanaged_agent", v14, ("runtime",)),
    "V15": ("stale_trigger", v15, ("triggers", "history")),
    "V16": ("log_mode_long", v16, ()),
}


def evaluate(inv: Inventory, ctx) -> Tuple[List[Finding], Dict[str, str]]:
    findings: List[Finding] = []
    not_evaluated: Dict[str, str] = {}
    ran = set(inv.host.layers_run)
    for fid, (name, fn, needs) in RULES.items():
        missing = [n for n in needs if n not in ran]
        if missing:
            not_evaluated[fid] = f"{name}: layer {', '.join(missing)} did not run"
            continue
        findings += fn(inv, ctx)
    sev = {s: i for i, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: (sev.get(f.severity, 9), f.id, f.agent_id or "", f.tool or "", f.summary))
    return findings, not_evaluated


def split_exempt(findings: List[Finding], agents: List[Agent]) -> Tuple[List[Finding], List[Finding]]:
    by_id = {a.id: a for a in agents}
    keep, exempted = [], []
    for f in findings:
        a = by_id.get(f.agent_id or "")
        tool_exempt = a is not None and f.tool and any(t.name == f.tool and t.coverage == "exempt" for t in a.tools)
        if a is not None and ((a.exempt and a.exempt.status == "approved") or tool_exempt):
            exempted.append(f)
        else:
            keep.append(f)
    return keep, exempted


# Plain-language explanation per rule: (why it matters, what to do).
EXPLAIN: Dict[str, Tuple[str, str]] = {
    "V01": ("If the agent is steered into a harmful action, it acts with full control of the machine.",
            "Run it as an unprivileged service user."),
    "V02": ("The agent can run any command it is talked into; nothing checks the command first.",
            "Give it a policy: an allowlist of commands, or block the tool."),
    "V03": ("The assistant no longer asks before it acts, so every tool call goes straight through.",
            "Turn permission prompts back on, or put a cslcore hook in front of its tools."),
    "V04": ("Anyone who can reach the webhook can make the agent act with its permissions.",
            "Require approval or a guard between external input and state-changing tools."),
    "V05": ("It changes things on a schedule with nobody watching.",
            "Add an approval rule, or start it in log mode and review what it does."),
    "V06": ("A mistake or a manipulated prompt could use these credentials far beyond the agent's job.",
            "Give the agent narrower credentials."),
    "V07": ("The agent can read or change any file of the user or the whole system.",
            "Root its filesystem access at the project folder."),
    "V08": ("Other machines on the network can call these tools.",
            "Bind the server to 127.0.0.1 or put authentication in front of it."),
    "V09": ("A tool that moves money or deletes data has no rule at all.",
            "Add a limit, an approval rule or a block for it."),
    "V10": ("The policy talks about values the agent never sends, so those rules can silently never match.",
            "Accept the suggested fix (cslcore policy fix) or map the value explicitly."),
    "V11": ("Some malformed or unknown inputs are allowed instead of blocked.",
            "Use the fail-closed mapping helpers; re-run cslcore map --test."),
    "V12": ("The policy exists but nothing enforces it.", "Wire it into the agent, or remove it."),
    "V13": ("The exemption is past its date and no longer applies.", "Renew it or remove it."),
    "V14": ("An agent is running that no configuration describes.", "Find out what it is; add it or stop it."),
    "V15": ("Its schedule still fires but the agent has not run for a long time.", "Remove the trigger or check the agent."),
    "V16": ("Log mode records but never blocks.", "Review cslcore watch, then switch the agent to block."),
}
