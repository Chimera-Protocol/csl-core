"""
Reach graph: what can influence what on one host, built only from what discovery found.

Nodes
    input     where untrusted content enters: an inbound webhook or message, or a tool that
              reads from the internet (its result lands in the agent's context)
    agent     a discovered agent
    impact    what is worth protecting: root on the host, money movement, command execution,
              publishing outside, a cloud or payment credential

Edges (each with the evidence it rests on and a confidence)
    input -> agent     the agent receives that input
    agent -> agent     the first agent can change the second one's code or configuration:
                       it has a write or command tool without a rule, and the second agent's
                       files are within its reach (declared file scope, the same account, or
                       root); a changed agent behaves as the changer wants
    agent -> impact    the agent holds that capability itself

A tool with an active rule breaks the edges it would enable: the policy decides what it may do.

Chains are paths from an input to an impact through at least one agent. The open core detects
and counts them and shows the strongest one in full; nothing here runs, reads or changes anything.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Agent, Inventory

RANK = {"likely": 3, "possible": 2}
INPUT_TOOLS = {"WebFetch", "web_fetch", "http_get", "fetch", "fetch_url", "browse", "web_search", "WebSearch",
               "fetch_feed", "read_email", "read_inbox"}
IMPACTS = {
    "root": ("root on the host", 5),
    "spend": ("moves money", 5),
    "cloud": ("cloud account credential", 4),
    "payment": ("payment credential", 5),
    "exec": ("runs commands", 3),
    "publish": ("publishes outside", 2),
    "destroy": ("deletes data", 4),
}


@dataclass
class Node:
    id: str
    kind: str  # input | agent | impact
    label: str
    detail: str = ""
    weight: int = 1  # impact severity, or how much flows through an agent
    active: bool = False  # an agent that is running now


@dataclass
class Edge:
    src: str
    dst: str
    label: str
    evidence: str
    confidence: str = "likely"  # likely | possible
    guarded: bool = False  # an active rule decides this step


@dataclass
class Chain:
    nodes: List[str]
    edges: List[Edge]
    score: int

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def confidence(self) -> str:
        return min((e.confidence for e in self.edges), key=lambda c: RANK.get(c, 0), default="likely")


@dataclass
class ReachGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)

    def out(self, nid: str) -> List[Edge]:
        return [e for e in self.edges if e.src == nid]

    def neighbours(self, nid: str) -> List[str]:
        return [e.dst for e in self.edges if e.src == nid] + [e.src for e in self.edges if e.dst == nid]

    @property
    def top(self) -> Optional[Chain]:
        return self.chains[0] if self.chains else None


def _open(tool) -> bool:
    """A tool no active rule decides (no guard, or wired without a rule)."""
    return tool.coverage in (None, "unguarded", "wired_no_rule")


def _home(a: Agent) -> List[str]:
    """Where an agent's code and configuration live."""
    paths = []
    if a.project:
        paths.append(a.project)
    if a.entrypoint and a.entrypoint.startswith("/"):
        paths.append(posixpath.dirname(a.entrypoint.split()[0]))
    for e in a.evidence:
        if e.layer in ("code", "config") and e.path.startswith("/"):
            paths.append(e.path)
    return list(dict.fromkeys(paths))


def _inside(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return root == "/" or path == root or path.startswith(root + "/")


def _reach(a: Agent, b: Agent) -> Optional[Tuple[str, str, str]]:
    """Can agent a change agent b? (how, evidence, confidence) or None."""
    if a.id == b.id:
        return None
    writers = [t for t in a.tools if t.risk_class in ("WRITE", "EXEC", "DESTRUCTIVE") and _open(t)]
    if not writers:
        return None
    homes = _home(b)
    if not homes:
        return None
    tool = sorted(writers, key=lambda t: (t.risk_class != "EXEC", t.name))[0]
    target = homes[0]
    if a.access.elevated or a.process_user == "root":
        return ("can change its code", f"{a.display_name} runs as root and has {tool.name} without a rule; "
                                       f"{b.display_name} lives in {target}", "likely")
    scoped = [r for r in a.access.fs_roots if any(_inside(h, r) for h in homes)]
    if scoped:
        return ("can change its code", f"{a.display_name} may work under {scoped[0]} and has {tool.name} "
                                       f"without a rule; {b.display_name} lives in {target}", "likely")
    if a.process_user and a.process_user == b.process_user and tool.risk_class == "EXEC":
        return ("can change its configuration", f"both run as {a.process_user}; {a.display_name} has "
                                                f"{tool.name} without a rule; {b.display_name} lives in {target}",
                "likely")
    return None


def _inputs(a: Agent) -> List[Tuple[str, str, str]]:
    """(node id, label, evidence) for the untrusted inputs an agent receives."""
    out = []
    for t in a.triggers:
        if t.type in ("inbound_http", "messaging", "email"):
            src = t.schedule or t.source or t.type
            out.append((f"input:{t.type}:{src}", {"inbound_http": "inbound HTTP", "messaging": "inbound messages",
                                                   "email": "inbound email"}[t.type] + f" {src}",
                        f"{a.display_name} is triggered by {t.type} ({src})"))
    readers = [t for t in a.tools if t.name in INPUT_TOOLS or (t.risk_class in ("READ", "EXTERNAL") and
                                                                any(p.name in ("url", "uri") for p in t.params))]
    if readers:
        out.append(("input:web", "web content", f"{a.display_name} reads web content with {readers[0].name}; "
                                                f"what it reads enters its context"))
    return out


def _impacts(a: Agent) -> List[Tuple[str, str]]:
    """(impact key, evidence) for what the agent can do itself without a rule in the way."""
    out = []
    if a.access.elevated or a.process_user == "root":
        out.append(("root", f"{a.display_name} runs as root"))
    for t in a.tools:
        if not _open(t):
            continue
        if t.risk_class == "SPEND":
            out.append(("spend", f"{t.name} without a rule"))
        elif t.risk_class == "EXEC":
            out.append(("exec", f"{t.name} without a rule"))
        elif t.risk_class == "DESTRUCTIVE":
            out.append(("destroy", f"{t.name} without a rule"))
        elif t.risk_class == "EXTERNAL" and t.name not in INPUT_TOOLS:
            out.append(("publish", f"{t.name} without a rule"))
    for c in a.access.credentials:
        if c.kind == "cloud_root":
            out.append(("cloud", f"{c.name} in {c.file}"))
        elif c.kind == "payment":
            out.append(("payment", f"{c.name} in {c.file}"))
    seen, unique = set(), []
    for k, ev in out:
        if k not in seen:
            seen.add(k)
            unique.append((k, ev))
    return unique


def build(inv: Inventory) -> ReachGraph:
    g = ReachGraph()
    agents = [a for a in inv.agents if a.exempt is None or a.exempt.status != "approved"]
    for a in agents:
        g.nodes[a.id] = Node(a.id, "agent", a.display_name, a.kind, active=a.state == "running")
        for nid, label, ev in _inputs(a):
            g.nodes.setdefault(nid, Node(nid, "input", label))
            g.edges.append(Edge(nid, a.id, "reaches", ev))
        for key, ev in _impacts(a):
            label, weight = IMPACTS[key]
            nid = f"impact:{key}"
            g.nodes.setdefault(nid, Node(nid, "impact", label, weight=weight))
            g.edges.append(Edge(a.id, nid, "can", ev))
    for a in agents:
        for b in agents:
            r = _reach(a, b)
            if r:
                how, ev, conf = r
                g.edges.append(Edge(a.id, b.id, how, ev, conf))
    g.chains = _chains(g)
    for n in g.nodes.values():
        if n.kind == "agent":
            n.weight = 1 + sum(1 for c in g.chains if n.id in c.nodes)
    return g


def _chains(g: ReachGraph, max_hops: int = 5) -> List[Chain]:
    """Reach chains: routes from an input through at least two agents to an impact that none of
    the earlier agents could reach on its own (an escalation, not a detour).

    One route per (input, entry agent, impact): the shortest, then the most certain. A longer
    detour to the same place adds nothing. Strongest first: impact, then certainty, then fewer
    steps (the more direct route is the more real one)."""
    from collections import deque

    best: Dict[Tuple[str, str, str], Chain] = {}
    for e0 in g.edges:
        if g.nodes[e0.src].kind != "input" or e0.guarded:
            continue
        queue = deque([([e0.src, e0.dst], [e0])])
        while queue:
            path, edges = queue.popleft()
            for e in g.out(path[-1]):
                if e.guarded or e.dst in path:
                    continue
                node = g.nodes[e.dst]
                if node.kind == "impact":
                    agents = [p for p in path if g.nodes[p].kind == "agent"]
                    if len(agents) < 2:
                        continue
                    # an escalation only: no earlier agent on the route can do this itself
                    if any(x.dst == e.dst and not x.guarded for a in agents[:-1] for x in g.out(a)):
                        continue
                    hops = edges + [e]
                    certain = all(x.confidence == "likely" for x in hops)
                    c = Chain(path + [e.dst], hops, node.weight * 100 + (10 if certain else 0) - len(hops))
                    key = (path[0], path[1], e.dst)
                    if key not in best or (c.hops, -c.score) < (best[key].hops, -best[key].score):
                        best[key] = c
                elif node.kind == "agent" and len(edges) < max_hops:
                    queue.append((path + [e.dst], edges + [e]))
    return sorted(best.values(), key=lambda c: (-c.score, c.nodes))


def direct(g: ReachGraph) -> List[Tuple[str, str, str]]:
    """(input, agent, impact) reachable through a single agent: exposures the findings already name."""
    out = []
    for e in g.edges:
        if g.nodes[e.src].kind != "input" or e.guarded:
            continue
        for f in g.out(e.dst):
            if g.nodes[f.dst].kind == "impact" and not f.guarded:
                out.append((e.src, e.dst, f.dst))
    return out


def describe(g: ReachGraph, chain: Chain) -> List[str]:
    """One line per hop, for screens and reports."""
    lines = []
    for e in chain.edges:
        a, b = g.nodes[e.src], g.nodes[e.dst]
        lines.append(f"{a.label} -> {b.label}: {e.evidence}")
    return lines


def strongest_direct(g: ReachGraph) -> Optional[Tuple[str, str, str]]:
    """The most serious single-agent exposure (input -> agent -> impact), for hosts without chains."""
    items = direct(g)
    # the worst impact first, then an agent that is running now, then a stable order
    return min(items, key=lambda d: (-g.nodes[d[2]].weight, not g.nodes[d[1]].active, g.nodes[d[1]].label), default=None) if items else None


def summary(g: ReachGraph) -> Dict[str, object]:
    """What the open core reports: how many chains, and the strongest one in full."""
    top = g.top
    d = direct(g)
    return {
        "direct": len(d),
        "chains": len(g.chains),
        "agents_in_chains": len({n for c in g.chains for n in c.nodes if g.nodes[n].kind == "agent"}),
        "top": None if top is None else {
            "nodes": [g.nodes[n].label for n in top.nodes],
            "hops": describe(g, top),
            "confidence": top.confidence,
        },
    }


def agents_by_reach(g: ReachGraph) -> Sequence[Node]:
    return sorted((n for n in g.nodes.values() if n.kind == "agent"), key=lambda n: (-n.weight, n.label))
