"""
Targeted policy edits from the management panel: exempt an agent from one rule, or exempt a
tool from its rules. Every edit goes through the gate (parse, validate, Z3); the previous
version is kept in .csl/venom/history/ and the exemption is recorded in exemptions.yaml.

Only workspace policies are edited. An adopted policy (a file in the operator's own
repository) is never written by Venom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from ..layers.governance import read_policy
from ..model import Exemption
from .gate import verify_text


@dataclass
class EditResult:
    ok: bool
    message: str
    path: Optional[str] = None
    removed: Optional[List[str]] = None


def policy_path_for(ws, agent: str) -> Tuple[Optional[Path], str]:
    """The editable policy of an agent, or (None, reason)."""
    p = ws.policies / f"{agent}.csl"
    if p.exists():
        return p, ""
    adopted = (ws.load_state().get("adopted") or {})
    for aid, path in adopted.items():
        if aid.endswith(agent) or Path(path).stem == agent:
            return None, f"{agent} uses an adopted policy ({Path(path).name}); edit it in your repository"
    return None, f"{agent} has no workspace policy (.csl/policies/{agent}.csl)"


def _block(text: str, rule: str) -> Optional[Tuple[int, int]]:
    m = re.search(rf"(?m)^[ \t]*STATE_CONSTRAINT\s+{re.escape(rule)}\s*\{{", text)
    if not m:
        return None
    depth, i = 0, m.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return m.start(), i + 1
        i += 1
    return None


def remove_rules(text: str, rules: List[str], note: str) -> str:
    for rule in rules:
        span = _block(text, rule)
        if span is None:
            continue
        start, end = span
        body = text[start:end]
        commented = "\n".join("  // " + line.strip() for line in body.splitlines())
        text = text[:start] + f"  // {note}\n" + commented + text[end:]
    return text


def exempt_agent_from_rule(text: str, rule: str, agent: str, note: str) -> Tuple[str, str]:
    """Add `agent_id != "agent"` to the rule when other agents share the policy; otherwise the
    rule only ever applied to this agent and is removed. Returns (text, what happened)."""
    ref = read_policy("<memory>", text, "active")
    span = _block(text, rule)
    if span is None:
        return text, f"rule {rule} not found"
    ids = ref.vocabulary.get("agent_id", [])
    if ids and agent in ids and len(ids) > 1:
        start, end = span
        block = text[start:end]
        if re.search(r"(?m)^\s*ALWAYS\s+True\s*$", block):
            new = re.sub(r"(?m)^(\s*)ALWAYS\s+True\s*$", rf'\1WHEN agent_id != "{agent}"', block, count=1)
        else:
            new = re.sub(r"(?m)^(\s*WHEN\s+)(.+?)\s*$", rf'\1(\2) AND agent_id != "{agent}"', block, count=1)
        return text[:start] + f"  // {note}\n" + new + text[end:], f"{agent} excluded in {rule}"
    return remove_rules(text, [rule], note), f"{rule} removed (it only applied to {agent})"


def rules_for_tool(text: str, tool: str) -> List[str]:
    ref = read_policy("<memory>", text, "active")
    return [r for r, values in ref.rule_values.items() if any(v.endswith(f"={tool}") for v in values)]


def apply(ws, path: Path, new_text: str, exemption: Exemption) -> EditResult:
    old = ws.read(path) or ""
    if new_text == old:
        return EditResult(False, "nothing to change")
    g = verify_text(new_text)
    if not g.ok:
        issue = g.issues[0].message if g.issues else g.stage
        return EditResult(False, f"not applied: the edited policy fails the gate ({issue})")
    if ws.plan_only:
        return EditResult(True, f"--plan-only: would update {ws.rel(path)}", ws.rel(path))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ws.write_text(ws.venom / "history" / f"{path.stem}-{stamp}.csl", old)
    ws.write_text(path, new_text)
    items = ws.load_exemptions()
    items.append(exemption)
    ws.save_exemptions(items)
    return EditResult(True, f"{ws.rel(path)} updated and verified (previous version in .csl/venom/history/)", ws.rel(path))


def note_for(e: Exemption) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"exempted by {e.approved_by} on {when}: {e.reason}"
