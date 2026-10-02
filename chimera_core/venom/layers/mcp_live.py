"""
L7 live MCP: ask configured MCP servers for their tools.

Only with `--probe` and the operator's confirmation: stdio servers are started (the one
exception to "never execute discovered code") and loopback HTTP servers are queried. The
authoritative tool lists and schemas replace the heuristic, catalog-based ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..analysis.risk import classify
from ..layers.code import _params_from_schema
from ..model import Agent, Evidence, Tool
from .config import ConfigScan, McpServer

PROTOCOL = "2025-06-18"


def _messages() -> List[dict]:
    return [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "csl-core-venom", "version": "0.6"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]


@dataclass
class LiveResult:
    servers: Dict[str, int] = field(default_factory=dict)  # server -> tools listed (-1 = failed)


def servers_to_probe(cfg: ConfigScan) -> List[McpServer]:
    seen = {}
    for rec in cfg.assistants:
        for s in rec.mcp_servers:
            key = s.url or " ".join([s.command or ""] + s.args)
            seen.setdefault(key, s)
    return list(seen.values())


def list_tools(probe, server: McpServer):
    if server.url:
        replies = probe.mcp_http(server.url, _messages())
    elif server.command:
        replies = probe.mcp_stdio([server.command] + server.args, server.env_values, _messages())
    else:
        return None
    for r in replies or []:
        if r.get("id") == 2 and isinstance(r.get("result"), dict):
            return r["result"].get("tools") or []
    return None


def enumerate_live(probe, cfg: ConfigScan, agents: List[Agent]) -> LiveResult:
    res = LiveResult()
    by_id = {a.id: a for a in agents}
    cache: Dict[str, object] = {}
    for rec in cfg.assistants:
        agent = by_id.get(rec.id)
        if agent is None:
            continue
        for srv in rec.mcp_servers:
            key = srv.url or " ".join([srv.command or ""] + srv.args)
            if key not in cache:
                cache[key] = list_tools(probe, srv)
            listed = cache[key]
            if listed is None:
                res.servers[srv.name] = -1
                continue
            res.servers[srv.name] = len(listed)  # type: ignore[arg-type]
            tag = f"MCP server '{srv.name}'"
            agent.tools = [t for t in agent.tools if not any(tag == (e.detail or "") for e in t.evidence)]
            for spec in listed:  # type: ignore[union-attr]
                if not isinstance(spec, dict) or not spec.get("name"):
                    continue
                t = Tool(name=str(spec["name"]), source="mcp_live", mcp_server=srv.package or srv.name,
                         description=(str(spec.get("description") or "")[:200] or None),
                         params=_params_from_schema(spec.get("inputSchema") or spec.get("input_schema")),
                         evidence=[Evidence("mcp", srv.source, None, tag)])
                t.risk_class, t.risk_reason = classify(t)
                if t.risk_reason.startswith("catalog"):
                    t.risk_reason += " · listed live"
                agent.tools.append(t)
            agent.tools.sort(key=lambda t: t.name.lower())
    return res
