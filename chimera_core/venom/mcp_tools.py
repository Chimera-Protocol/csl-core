"""
MCP authoring tools. The model belongs to the user's own
assistant; these tools only return redacted summaries and save verified drafts.

Guarantees:
  * no credential values, no prompt or transcript text in any output (names and counts only);
  * outputs stay short (the host's full detail stays in the reports on disk);
  * venom_save_draft verifies first and writes only to .csl/venom/drafts/, never policies/;
  * venom_propose_exemption can only create `status: proposed`; approval is a CLI action.
"""

from __future__ import annotations

from typing import List, Optional

from . import redact
from .analysis.coverage import link_policies
from .model import Exemption, Inventory
from .policy import draft as D
from .policy.gate import verify_text
from .workspace import Workspace

LIMIT = 3900


def _inv(workspace: str, root: Optional[str] = None, rescan: bool = False) -> Inventory:
    ws = Workspace(workspace)
    data = None if rescan or root else ws.latest_inventory()
    if data is not None:
        return Inventory.from_dict(data)
    from .probe import probe_for
    from .scanner import Scanner
    from . import VENOM_VERSION

    probe, roots = probe_for(root)
    return Scanner(probe, roots, ws, tool_version=VENOM_VERSION).run().inventory


def _clip(text: str) -> str:
    text = redact.text(text)
    return text if len(text) <= LIMIT else text[: LIMIT - 60].rstrip() + "\n... (truncated; full detail: cslcore venom report)"


def _find(inv: Inventory, agent_id: str):
    a = inv.agent(agent_id)
    if a is None:
        hits = [x for x in inv.agents if agent_id in (x.id, D.agent_key(x)) or agent_id in x.id]
        a = hits[0] if len(hits) == 1 else None
    return a


def inventory(workspace: str = ".", root: Optional[str] = None) -> str:
    inv = _inv(workspace, root)
    c = inv.coverage
    lines = [f"# Venom inventory: {inv.host.name} ({inv.host.os}), {len(inv.agents)} agents",
             f"Coverage: {c.guarded} of {c.tools_total - c.exempt} tools guarded, {c.wired_no_rule} wired without a rule, "
             f"{c.unguarded} unguarded, {c.exempt} exempt.", "", "| agent | kind | state | tools | riskiest | guard |", "|---|---|---|---|---|---|"]
    order = ["DESTRUCTIVE", "SPEND", "EXEC", "IDENTITY", "UNCLASSIFIED", "EXTERNAL", "WRITE", "READ"]
    for a in inv.agents[:25]:
        classes = sorted({t.risk_class for t in a.tools}, key=order.index)
        guard = "exempt" if a.exempt and a.exempt.status == "approved" else a.guard.status
        lines.append(f"| {a.display_name} | {a.kind} | {a.state} | {len(a.tools)} | {classes[0] if classes else '-'} | {guard} |")
    if inv.findings:
        lines += ["", "Findings:"] + [f"- {f.id} {f.severity}: {f.summary}" for f in inv.findings[:10]]
    lines += ["", "Next: venom_agent(agent_id) for one agent, venom_policy_context(agent_id) before drafting."]
    return _clip("\n".join(lines))


def agent(agent_id: str, workspace: str = ".", root: Optional[str] = None) -> str:
    inv = _inv(workspace, root)
    a = _find(inv, agent_id)
    if a is None:
        return f"No agent matches '{agent_id}'. Known: " + ", ".join(x.display_name for x in inv.agents[:30])
    lines = [f"# {a.display_name}", f"id: {a.id}", f"agent key (use as agent_id in policies): {D.agent_key(a)}",
             f"kind: {a.kind}, state: {a.state}, framework: {', '.join(a.framework) or 'n/a'}",
             f"guard: {a.guard.status} {a.guard.mechanism or ''} {a.guard.mode or ''}".rstrip(),
             f"runs ({a.runs.window}): {'n/a' if a.runs.count is None else a.runs.count}", "", "## Tools"]
    for t in a.tools[:40]:
        params = ", ".join(f"{p.name}: {p.type or 'any'}" + (f" in {p.enum}" if p.enum else "") +
                           (f" [{p.minimum}..{p.maximum}]" if p.minimum is not None or p.maximum is not None else "") for p in t.params)
        lines.append(f"- {t.name} ({t.risk_class}, {t.coverage or 'n/a'}): {params or 'no parameters'}")
    if a.triggers:
        lines += ["", "## Triggers"] + [f"- {t.type} {t.schedule or ''}" for t in a.triggers]
    if a.access.credentials:
        kinds = sorted({c.kind for c in a.access.credentials})
        lines += ["", f"Credentials reachable: {len(a.access.credentials)} names ({', '.join(kinds)}); values are never read."]
    if a.access.fs_roots:
        lines.append(f"Filesystem roots: {', '.join(a.access.fs_roots)}")
    if a.access.permission_mode:
        lines.append(f"Permission mode: {a.access.permission_mode}")
    return _clip("\n".join(lines))


def policy_context(agent_id: str, workspace: str = ".", root: Optional[str] = None) -> str:
    ws = Workspace(workspace)
    inv = _inv(workspace, root)
    a = _find(inv, agent_id)
    if a is None:
        return f"No agent matches '{agent_id}'."
    key = D.agent_key(a)
    lines = [f"# Policy context for {a.display_name} (agent key {key})"]
    active = ws.policies / f"{key}.csl"
    linked = link_policies(a, inv.policies)
    current = ws.read(active) if active.exists() else None
    if current is None and linked:
        current = ws.read(linked[0].path)
    if current:
        lines += ["", "## Active policy", "```csl", current.strip()[:1800], "```"]
    else:
        lines += ["", "No active policy yet (first install). A deterministic starting draft:", "```csl",
                  D.draft_for(a, ws.load_exemptions()).text.strip()[:1800], "```"]
    fs = [f for f in inv.findings if f.agent_id == a.id]
    if fs:
        lines += ["", "## Findings"] + [f"- {f.id}: {f.summary}" for f in fs[:8]]
    dr = [d for d in inv.drift if d.agent_id == a.id and d.kind != "unsupplied_variable"]
    if dr:
        lines += ["", "## Vocabulary drift"] + [f"- {d.detail or d.variable}" for d in dr[:8]]
    lines += ["", "## Rules for your draft",
              "- tool values must be the real tool names listed by venom_agent; keep an agent_id variable",
              "- each STATE_CONSTRAINT needs WHEN <condition> (or ALWAYS True) and THEN <variable> <op> <value>",
              "- two rules must never require conflicting values when both can trigger (Z3 checks this)",
              f"- save with venom_save_draft(\"{key}\", csl_content, note); the operator reviews and activates it"]
    return _clip("\n".join(lines))


def save_draft(agent_id: str, csl_content: str, note: str = "", workspace: str = ".", root: Optional[str] = None) -> str:
    ws = Workspace(workspace)
    inv = _inv(workspace, root)
    a = _find(inv, agent_id)
    key = D.agent_key(a) if a is not None else D.slug(agent_id)
    g = verify_text(csl_content)
    if not g.ok:
        issues = "\n".join(f"- {i.kind}: {i.message}" + (f" (rules: {', '.join(i.rules)})" if i.rules else "") for i in g.issues[:8])
        return _clip(f"NOT SAVED: the draft failed at {g.stage}.\n{issues}\nFix it and call venom_save_draft again.")
    path = ws.drafts / f"{key}.csl"
    header = f"// Draft written with an AI assistant via MCP. Note: {note.strip()[:200]}\n" if note.strip() else ""
    text = csl_content if csl_content.startswith("// Draft written") else header + csl_content
    if not verify_text(text).ok:
        text = csl_content
    ws.write_text(path, text)
    return (f"SAVED to {ws.rel(path)}: {g.rules} rules, {g.variables} variables, Z3: no contradictions.\n"
            f"Not active. The operator reviews it with `cslcore policy diff {key}` and activates it with "
            f"`cslcore policy activate {key}` (or continues `cslcore setup`).")


def propose_exemption(agent: str, scope: str, reason: str, tool: Optional[str] = None, workspace: str = ".") -> str:
    ws = Workspace(workspace)
    if scope not in ("agent", "tool"):
        return "scope must be 'agent' or 'tool'"
    if not reason or not reason.strip():
        return "a reason is required"
    if scope == "tool" and not tool:
        return "a tool exemption needs the tool name"
    items: List[Exemption] = ws.load_exemptions()
    items.append(Exemption(agent=agent, scope=scope, tool=tool, reason=reason.strip()[:300], approved_by=None, status="proposed"))
    ws.save_exemptions(items)
    return (f"PROPOSED exemption #{len(items)} for {agent}{' / ' + tool if tool else ''}. It has no effect until the operator "
            f"approves it: cslcore exempt approve {len(items)} --approved-by NAME")
