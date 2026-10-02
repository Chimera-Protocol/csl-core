"""
Exemptions.

Stored in .csl/venom/exemptions.yaml. `reason` and `approved_by` are required; an expired
exemption stops applying. Only the CLI approves; assistants (MCP) can only propose.
The file format is a small YAML subset read and written here, so no YAML dependency.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from .model import Agent, Exemption

FIELDS = ["agent", "scope", "tool", "rule", "reason", "approved_by", "expires", "status"]


class ExemptionError(ValueError):
    pass


def _unquote(v: str) -> Optional[str]:
    v = v.strip()
    if not v or v in ("null", "~"):
        return None
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return re.sub(r"\\(.)", r"\1", v[1:-1])
    return v


def _quoted_prefix(v: str) -> str:
    """The leading quoted string of v (honouring backslash escapes), quotes included."""
    q = v[0]
    i = 1
    while i < len(v):
        if v[i] == "\\":
            i += 2
            continue
        if v[i] == q:
            return v[: i + 1]
        i += 1
    return v


def _quote(v: str) -> str:
    return '"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"'


def parse(text: str) -> List[Exemption]:
    items: List[Dict[str, Optional[str]]] = []
    cur: Optional[Dict[str, Optional[str]]] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "exemptions:" or stripped == "exemptions: []":
            continue
        if stripped.startswith("- "):
            cur = {}
            items.append(cur)
            stripped = stripped[2:]
        if cur is None or ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        # strip trailing comments outside quotes
        v = val.strip()
        if v and v[0] not in "\"'" and " #" in v:
            v = v.split(" #", 1)[0]
        elif v and v[0] in "\"'":
            v = _quoted_prefix(v)
        cur[key.strip()] = _unquote(v)
    out = []
    for d in items:
        if not d.get("agent"):
            continue
        out.append(Exemption(
            agent=d["agent"] or "*", scope=d.get("scope") or "agent", tool=d.get("tool"),
            reason=d.get("reason"), approved_by=d.get("approved_by"), expires=d.get("expires"),
            status=d.get("status") or "proposed", rule=d.get("rule"),
        ))
    return out


def dump(items: List[Exemption]) -> str:
    lines = ["# CSL-Core Venom exemptions. Approve with: cslcore exempt approve <n>", "exemptions:"]
    if not items:
        lines[-1] = "exemptions: []"
    for e in items:
        first = True
        for f in FIELDS:
            v = getattr(e, f)
            if v is None:
                continue
            prefix = "  - " if first else "    "
            first = False
            lines.append(f"{prefix}{f}: {_quote(str(v)) if f in ('agent', 'tool', 'rule', 'reason', 'approved_by') else v}")
    return "\n".join(lines) + "\n"


def validate(e: Exemption) -> None:
    if not e.reason or not e.reason.strip():
        raise ExemptionError("an exemption needs --reason (why this agent or tool is trusted)")
    if not e.approved_by or not e.approved_by.strip():
        raise ExemptionError("an exemption needs --approved-by (who takes responsibility)")
    if e.scope not in ("agent", "tool", "rule"):
        raise ExemptionError("scope must be 'agent', 'tool' or 'rule'")
    if e.scope == "rule" and not e.rule:
        raise ExemptionError("a rule exemption needs the rule name")
    if e.scope == "tool" and not e.tool:
        raise ExemptionError("a tool exemption needs --tool")
    if e.expires:
        try:
            date.fromisoformat(e.expires)
        except ValueError:
            raise ExemptionError("--expires must be a date like 2026-12-31")


def expired(e: Exemption, today: date) -> bool:
    if not e.expires:
        return False
    try:
        return date.fromisoformat(e.expires) < today
    except ValueError:
        return True


def matches(e: Exemption, agent: Agent) -> bool:
    return e.agent == "*" or e.agent in (agent.id, agent.display_name) or fnmatch.fnmatch(agent.id, e.agent)


def apply(agents: List[Agent], items: List[Exemption], today: date) -> Tuple[List[Exemption], List[Exemption]]:
    """Mark exempt agents and tools. Returns (applied, expired)."""
    applied, gone = [], []
    for e in items:
        if e.status != "approved":
            continue
        if expired(e, today):
            gone.append(e)
            continue
        hit = False
        for a in agents:
            if not matches(e, a):
                continue
            if e.scope == "agent":
                a.exempt = e
                hit = True
            else:
                for t in a.tools:
                    if fnmatch.fnmatch(t.name, e.tool or ""):
                        t.coverage = "exempt"
                        hit = True
        if hit:
            applied.append(e)
    return applied, gone
