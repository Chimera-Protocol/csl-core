"""
Reports: report-<ts>.json (schema_version 1) and .md, plus
latest.*. The Markdown mirrors the scan screen with every list in full.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .. import redact
from ..model import Inventory
from .screen import _n


def to_json(inv: Inventory, since=None) -> Dict[str, Any]:
    """`since` is the reach diff against the previous scan (reach.since_last), when there is one."""
    from ..reach import build, diff_summary, summary

    data = inv.to_dict()
    data["coverage"]["ratio"] = inv.coverage.ratio
    data["reach"] = summary(build(inv))
    if since is not None:
        data["reach"]["since_last_scan"] = diff_summary(since)
    return redact.deep(data)


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def to_markdown(inv: Inventory, since=None) -> str:
    h = inv.host
    out: List[str] = [
        "# CSL-Core Venom report",
        "",
        f"- Host: `{h.name}` ({h.os}), scope: {h.scope} ({h.mode})",
        f"- Scanned: {h.scanned_at}, {h.duration_ms} ms, read-only",
        f"- Layers run: {', '.join(h.layers_run) or 'none'}",
        f"- Files scanned: {h.files_scanned:,}, parse errors: {h.parse_errors}, paths not readable: {h.not_readable}"
        + (", partial (time budget)" if h.partial else ""),
        f"- Venom {inv.tool_version}, schema {inv.schema_version}",
        "",
    ]
    if h.layers_unavailable:
        out += ["Sources not available: " + "; ".join(f"{k} ({v})" for k, v in sorted(h.layers_unavailable.items())), ""]

    c = inv.coverage
    pct = "n/a" if c.ratio is None else f"{c.ratio * 100:.0f}%"
    out += ["## Summary", "",
            "| Agents | Tools | Guarded | Wired, no rule | Unguarded | Exempt | Coverage |",
            "|---|---|---|---|---|---|---|",
            f"| {len(inv.agents)} | {c.tools_total} | {c.guarded} | {c.wired_no_rule} | {c.unguarded} | {c.exempt} | {pct} |", ""]

    out += ["## Agents", "",
            "| Agent | Kind | State | Runs | Tools | Guard | Id |", "|---|---|---|---|---|---|---|"]
    for a in inv.agents:
        guard = "exempt" if a.exempt and a.exempt.status == "approved" else a.guard.status + (f" ({a.guard.mechanism})" if a.guard.mechanism else "")
        out.append(f"| {_esc(a.display_name)} | {a.kind} | {a.state} | {_n(a.runs.count)} ({a.runs.window}) | {len(a.tools)} | {guard} | `{_esc(a.id)}` |")
    out.append("")

    for a in inv.agents:
        out += [f"### {_esc(a.display_name)}", ""]
        facts = [("Id", f"`{a.id}`"), ("Kind", a.kind), ("State", a.state), ("Framework", ", ".join(a.framework) or "n/a"),
                 ("Models", ", ".join(a.model_ids) or "n/a"), ("Entrypoint", a.entrypoint or "n/a"),
                 ("Runs as", (a.process_user or "n/a") + (" (elevated)" if a.access.elevated else "")),
                 ("Permission mode", a.access.permission_mode or "n/a"),
                 ("Runs", f"{_n(a.runs.count)} in {a.runs.window}" + (f" (source: {a.runs.source})" if a.runs.source else "")),
                 ("Guard", f"{a.guard.status} {a.guard.mechanism or ''} {a.guard.mode or ''} {', '.join(a.guard.policy_ids)}".strip()),
                 ("System prompt", f"present, {a.system_prompt.length} chars, sha256 {a.system_prompt.sha256}" if a.system_prompt.present else "not found")]
        out += [f"- {k}: {_esc(v)}" for k, v in facts]
        if a.tools:
            out += ["", "| Tool | Class | Coverage | Params | Why |", "|---|---|---|---|---|"]
            for t in a.tools:
                params = ", ".join(f"{p.name}: {p.type or 'any'}" for p in t.params) or "-"
                out.append(f"| {_esc(t.name)} | {t.risk_class} | {t.coverage or '-'} | {_esc(params)} | {_esc(t.risk_reason or '')} |")
        if a.access.credentials:
            out += ["", "Credentials (names only): " + ", ".join(f"`{c.name}` ({c.kind}, {c.file})" for c in a.access.credentials)]
        if a.access.fs_roots:
            out += ["", "Filesystem roots: " + ", ".join(f"`{r}`" for r in a.access.fs_roots)]
        if a.triggers:
            out += ["", "Triggers: " + "; ".join(f"{t.type} {t.schedule or ''} ({t.source})" for t in a.triggers)]
        if a.evidence:
            out += ["", "Evidence:"] + [f"- {e.layer}: `{e.path}{':' + str(e.line) if e.line else ''}` {e.detail or ''}" for e in a.evidence]
        out.append("")

    out += ["## Findings", ""]
    if not inv.findings:
        out.append("No findings.")
    for f in inv.findings:
        out.append(f"- **{f.id} {f.severity}** {_esc(f.summary)}" + (f". Recommendation: {f.recommendation}" if f.recommendation else ""))
    out.append("")
    if inv.rules_not_evaluated:
        out += ["Rules not evaluated:"] + [f"- {k}: {v}" for k, v in sorted(inv.rules_not_evaluated.items())] + [""]

    from ..reach import build, describe

    g = build(inv)
    out += ["## Reach", ""]
    if not g.chains:
        out.append("No agent can pass control on to another agent here: no reach chain.")
    else:
        top = g.top
        out.append("Strongest reach chain (" + top.confidence + f", {top.hops} steps): "
                   + " → ".join(f"**{_esc(g.nodes[n].label)}**" for n in top.nodes))
        out.append("")
        out += [f"{i}. {_esc(line)}" for i, line in enumerate(describe(g, top), 1)]
        more = len(g.chains) - 1
        if more:
            out += ["", f"{more} more reach chain{'s' if more != 1 else ''} on this host."]
    out.append("")
    if since is not None:
        out += ["### Since the last scan", "", f"Compared with the scan of {_when(since.since)}."]
        if not since.changed:
            out.append("No path opened or closed.")
        out += [f"- opened: {_esc(since.step(e))} ({_esc(e.evidence)})" for e in since.opened]
        out += [f"- closed: {_esc(since.step(e))}" for e in since.closed]
        out += [f"- new agent: {_esc(a)}" for a in since.new_agents]
        out += [f"- agent gone: {_esc(a)}" for a in since.gone_agents]
        if since.new_chains:
            n = len(since.new_chains)
            out.append(f"- {n} new reach chain{'s' if n != 1 else ''}; the strongest: "
                       + " → ".join(_esc(since.labels[x]) for x in since.new_chains[0].nodes))
        out.append("")

    out += ["## Vocabulary drift", ""]
    if not inv.drift:
        out.append("No drift." if any(p.status == "active" for p in inv.policies) else "No active policy yet (first install): drift checks start once a policy is active.")
    for d in inv.drift:
        sug = f" (suggestion: `{d.suggestion}`)" if d.suggestion else ""
        out.append(f"- {d.kind}: {d.policy}.{d.variable}" + (f" = `{d.value}`" if d.value else "") + f"{sug}. {_esc(d.detail or '')}")
    out.append("")

    out += ["## Policies", ""]
    if not inv.policies:
        out.append("No policies found.")
    else:
        out += ["| Policy | Status | Rules | Hash | Path |", "|---|---|---|---|---|"]
        for p in inv.policies:
            name = p.policy_id or p.domain or "?"
            status = p.status + (f" (error: {_esc(p.error)})" if p.error else "")
            out.append(f"| {_esc(name)} | {status} | {len(p.rules)} | `{(p.policy_hash or '')[:12]}` | `{_esc(p.path)}` |")
    out.append("")

    out += ["## Exempted by operator", ""]
    ex_agents = [a for a in inv.agents if a.exempt and a.exempt.status == "approved"]
    ex_tools = [(a, t) for a in inv.agents for t in a.tools if t.coverage == "exempt" and not (a.exempt and a.exempt.status == "approved")]
    if not ex_agents and not ex_tools and not inv.exempted:
        out.append("None.")
    for a in ex_agents:
        e = a.exempt
        out.append(f"- agent `{a.id}`: {_esc(e.reason)} (approved by {_esc(e.approved_by)}{', expires ' + e.expires if e.expires else ''})")
    for a, t in ex_tools:
        out.append(f"- tool `{t.name}` of `{a.display_name}`")
    for f in inv.exempted:
        out.append(f"- finding {f.id} suppressed: {_esc(f.summary)}")
    out.append("")
    return redact.text("\n".join(out))


def json_text(inv: Inventory, since=None) -> str:
    return json.dumps(to_json(inv, since), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _when(stamp: str) -> str:
    """'2026-10-01T14:22:00+00:00' -> '2026-10-01 14:22'."""
    return stamp[:16].replace("T", " ") if stamp else "an earlier scan"
