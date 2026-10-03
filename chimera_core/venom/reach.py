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
        index = self.__dict__.get("_out")
        if index is None or self.__dict__.get("_out_n") != len(self.edges):
            index = {}
            for e in self.edges:
                index.setdefault(e.src, []).append(e)
            self.__dict__["_out"], self.__dict__["_out_n"] = index, len(self.edges)
        return index.get(nid, [])

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


INPUT_KIND = {"inbound_http": "inbound HTTP", "messaging": "inbound messages", "email": "inbound email"}


def _inputs(a: Agent) -> List[Tuple[str, str, str, List[str]]]:
    """(node id, kind label, evidence, routes) for the untrusted inputs an agent receives. One node per
    kind of input: an API with a hundred routes is one way in, not a hundred."""
    out = []
    routes: Dict[str, List[str]] = {}
    for t in a.triggers:
        if t.type in INPUT_KIND:
            routes.setdefault(t.type, []).append(t.schedule or t.source or t.type)
    for kind, rs in routes.items():
        rs = list(dict.fromkeys(rs))
        shown = ", ".join(rs[:3]) + (f" and {len(rs) - 3} more" if len(rs) > 3 else "")
        out.append((f"input:{kind}", INPUT_KIND[kind],
                    f"{a.display_name} is triggered by {INPUT_KIND[kind]} ({shown})", rs))
    readers = [t for t in a.tools if t.name in INPUT_TOOLS or (t.risk_class in ("READ", "EXTERNAL") and
                                                                any(p.name in ("url", "uri") for p in t.params))]
    if readers:
        out.append(("input:web", "web content", f"{a.display_name} reads web content with {readers[0].name}; "
                                                f"what it reads enters its context", []))
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
    routes: Dict[str, List[str]] = {}
    for a in agents:
        g.nodes[a.id] = Node(a.id, "agent", a.display_name, a.kind, active=a.state == "running")
        for nid, label, ev, rs in _inputs(a):
            g.nodes.setdefault(nid, Node(nid, "input", label))
            routes.setdefault(nid, []).extend(rs)
            g.edges.append(Edge(nid, a.id, "reaches", ev))
        for key, ev in _impacts(a):
            label, weight = IMPACTS[key]
            nid = f"impact:{key}"
            g.nodes.setdefault(nid, Node(nid, "impact", label, weight=weight))
            g.edges.append(Edge(a.id, nid, "can", ev))
    for nid, rs in routes.items():  # "inbound HTTP /hook", or "inbound HTTP · 113 routes"
        rs = list(dict.fromkeys(rs))
        if len(rs) == 1:
            g.nodes[nid].label += f" {rs[0]}"
        elif rs:
            g.nodes[nid].label += f" · {len(rs)} routes"
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

    One search per (input, impact): breadth first from every agent the input reaches that cannot
    do the impact itself, through agents that cannot do it either, until an agent that can. That
    gives the shortest route to each such agent, and the cost grows with the size of the graph,
    not with the number of paths through it (hundreds of agents on one account stay fast).
    One chain per (input, impact, agent that holds it). Strongest first: impact, then certainty,
    then fewer steps."""
    from collections import deque

    owners: Dict[str, set] = {}
    for e in g.edges:
        if g.nodes[e.dst].kind == "impact" and not e.guarded:
            owners.setdefault(e.dst, set()).add(e.src)
    found: List[Chain] = []
    inputs = [n.id for n in g.nodes.values() if n.kind == "input"]
    for x in inputs:
        entries = [e for e in g.out(x) if not e.guarded]
        for impact, holders in owners.items():
            parent: Dict[str, Tuple[Optional[str], Edge]] = {}
            queue = deque()
            for e in entries:
                if e.dst not in holders and e.dst not in parent:
                    parent[e.dst] = (None, e)
                    queue.append((e.dst, 1))
            reached: List[str] = []
            while queue:
                here, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                for e in g.out(here):
                    if e.guarded or g.nodes[e.dst].kind != "agent" or e.dst in parent:
                        continue
                    parent[e.dst] = (here, e)
                    if e.dst in holders:
                        reached.append(e.dst)  # an agent that can do it: the chain ends here
                    else:
                        queue.append((e.dst, depth + 1))
            direct_to = {e.dst for e in entries}
            for holder in reached:
                if holder in direct_to:
                    continue  # the input reaches it without going through anyone: a direct exposure
                path, edges = [holder], []
                node = holder
                while node is not None:
                    prev, e = parent[node]
                    edges.append(e)
                    path.append(prev if prev is not None else x)
                    node = prev
                path.reverse()
                edges.reverse()
                last = next(e for e in g.out(holder) if e.dst == impact and not e.guarded)
                hops = edges + [last]
                certain = all(h.confidence == "likely" for h in hops)
                found.append(Chain(path + [impact], hops, g.nodes[impact].weight * 100 + (10 if certain else 0) - len(hops)))
    return sorted(found, key=lambda c: (-c.score, c.nodes))


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


# ---------------------------------------------------------------------------
# since the last scan
# ---------------------------------------------------------------------------

@dataclass
class ReachDiff:
    """What changed in the reach graph between two scans of the same scope. A scan is a snapshot:
    a plugin installed or a credential added afterwards opens a path the last map does not show."""
    since: str  # when the earlier scan ran
    opened: List[Edge]  # edges of the new graph that were not there
    closed: List[Edge]  # edges of the old graph that are gone (or now decided by a rule)
    labels: Dict[str, str]  # node labels from both graphs
    new_chains: List[Chain]  # chains of the new graph whose (input, holder, impact) is new
    closed_chains: int
    new_agents: List[str]
    gone_agents: List[str]

    @property
    def changed(self) -> bool:
        return bool(self.opened or self.closed or self.new_agents or self.gone_agents)

    def step(self, e: Edge) -> str:
        return f"{self.labels.get(e.src, e.src)} -> {self.labels.get(e.dst, e.dst)}"


def _paths(g: ReachGraph) -> Dict[Tuple[str, str], Edge]:
    return {(e.src, e.dst): e for e in g.edges if not e.guarded}


def _chain_key(c: Chain) -> Tuple[str, str, str]:
    return c.nodes[0], c.nodes[-2], c.nodes[-1]


def diff(old: ReachGraph, new: ReachGraph, since: str = "") -> ReachDiff:
    """Paths that opened or closed since `old`. The most serious first: what an agent can do on
    its own, then one agent reaching another, then a new way in."""
    before, after = _paths(old), _paths(new)
    labels = {n.id: n.label for n in old.nodes.values()}
    labels.update({n.id: n.label for n in new.nodes.values()})

    def rank(g: ReachGraph, e: Edge):
        kind = g.nodes[e.dst].kind if e.dst in g.nodes else ""
        order = {"impact": 0, "agent": 1}.get(kind, 2)
        return order, -(g.nodes[e.dst].weight if kind == "impact" else 0), labels.get(e.src, ""), labels.get(e.dst, "")

    opened = sorted((e for k, e in after.items() if k not in before), key=lambda e: rank(new, e))
    closed = sorted((e for k, e in before.items() if k not in after), key=lambda e: rank(old, e))
    old_chains = {_chain_key(c) for c in old.chains}
    new_keys = {_chain_key(c) for c in new.chains}
    agents_before = {n.id for n in old.nodes.values() if n.kind == "agent"}
    agents_after = {n.id for n in new.nodes.values() if n.kind == "agent"}
    return ReachDiff(
        since=since, opened=opened, closed=closed, labels=labels,
        new_chains=[c for c in new.chains if _chain_key(c) not in old_chains],
        closed_chains=len(old_chains - new_keys),
        new_agents=sorted(labels[a] for a in agents_after - agents_before),
        gone_agents=sorted(labels[a] for a in agents_before - agents_after),
    )


def since_last(previous: Optional[Inventory], current: Inventory) -> Optional[ReachDiff]:
    """The diff against the previous scan, or None when there is none or it covered another scope."""
    if previous is None or previous.host.scope != current.host.scope or previous.host.name != current.host.name:
        return None
    return diff(build(previous), build(current), since=previous.host.scanned_at)


def diff_summary(d: ReachDiff) -> Dict[str, object]:
    """The JSON form: every opened and closed path with its evidence, and the strongest new chain
    in full (the open core shows one chain in full, as for the scan itself)."""
    top = d.new_chains[0] if d.new_chains else None
    return {
        "since": d.since,
        "opened": [{"path": d.step(e), "evidence": e.evidence, "confidence": e.confidence} for e in d.opened],
        "closed": [{"path": d.step(e), "evidence": e.evidence} for e in d.closed],
        "new_chains": len(d.new_chains),
        "closed_chains": d.closed_chains,
        "new_agents": d.new_agents,
        "gone_agents": d.gone_agents,
        "top_new_chain": None if top is None else [d.labels[n] for n in top.nodes],
    }
