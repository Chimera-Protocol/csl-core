"""Reach graph and chains: built only from discovery, guarded tools break them, escalations only."""

from __future__ import annotations

import copy

import pytest

from chimera_core.venom import reach as R
from chimera_core.venom.model import Agent, Exemption, Inventory, Tool, ToolParam, Trigger

from .conftest import HOST_OPS, scan_fixture


@pytest.fixture(scope="module")
def inv(tmp_path_factory):
    return scan_fixture(HOST_OPS, tmp_path_factory.mktemp("reach")).inventory


def _labels(g, chain):
    return [g.nodes[n].label for n in chain.nodes]


def test_strongest_chain_on_the_sample_host(inv):
    g = R.build(inv)
    assert g.top is not None
    assert _labels(g, g.top) == ["web content", "claude-code:ops", "membership-bot", "moves money"]
    hops = R.describe(g, g.top)
    assert "WebFetch" in hops[0] and "may work under /" in hops[1] and "transfer_funds" in hops[2]
    assert g.top.confidence == "likely"
    # ingest-worker reads the web itself and runs as root: a direct exposure, not a chain
    src, agent, impact = R.strongest_direct(g)
    assert (g.nodes[agent].label, g.nodes[impact].label) == ("ingest-worker", "root on the host")
    assert not any(c.nodes[-2] == agent and c.nodes[-1] == impact for c in g.chains)


def test_every_chain_escalates_through_at_least_two_agents(inv):
    g = R.build(inv)
    assert len(g.chains) == 1
    for c in g.chains:
        agents = [n for n in c.nodes if g.nodes[n].kind == "agent"]
        assert len(agents) >= 2 and g.nodes[c.nodes[0]].kind == "input" and g.nodes[c.nodes[-1]].kind == "impact"
        earlier = {e.dst for a in agents[:-1] for e in g.out(a)}
        assert c.nodes[-1] not in earlier  # no earlier agent could do it alone
        assert c.nodes[-2] not in {e.dst for e in g.out(c.nodes[0])}  # not reached directly
    keys = [(c.nodes[0], c.nodes[-2], c.nodes[-1]) for c in g.chains]
    assert len(keys) == len(set(keys))  # one route per input, holder and impact


def test_same_account_needs_a_command_tool(inv):
    g = R.build(inv)
    sandbox_to_ops = [e for e in g.edges if g.nodes[e.src].label == "claude-code:sandbox"
                      and g.nodes[e.dst].label == "claude-code:ops"]
    assert sandbox_to_ops and "both run as operator" in sandbox_to_ops[0].evidence


def test_a_rule_on_the_enabling_tools_breaks_the_chain(inv):
    guarded = copy.deepcopy(inv)
    for a in guarded.agents:
        if a.display_name == "claude-code:ops":
            for t in a.tools:
                if t.risk_class in ("WRITE", "EXEC", "DESTRUCTIVE"):
                    t.coverage = "guarded"
    g = R.build(guarded)
    assert all("claude-code:ops" not in _labels(g, c)[1:-1] or _labels(g, c).index("claude-code:ops") == len(c.nodes) - 2
               for c in g.chains)  # ops no longer passes control on to another agent
    assert not any(g.nodes[e.src].label == "claude-code:ops" and g.nodes[e.dst].kind == "agent" for e in g.edges)


def test_approved_exemptions_and_quiet_hosts():
    a = Agent("code:/a", "a", "code", tools=[Tool("run_command", "decorator", risk_class="EXEC", coverage="unguarded")],
              triggers=[Trigger("inbound_http", "/hook")], process_user="root", project="/a")
    b = Agent("code:/b", "b", "code", tools=[Tool("pay", "decorator", risk_class="SPEND", coverage="unguarded")],
              project="/b")
    g = R.build(Inventory(agents=[a, b]))
    assert [_labels(g, c) for c in g.chains] == [["inbound HTTP /hook", "a", "b", "moves money"]]
    b.exempt = Exemption(agent="code:/b", status="approved", reason="test")
    assert R.build(Inventory(agents=[a, b])).chains == []
    quiet = Agent("code:/q", "q", "code", tools=[Tool("read", "decorator", risk_class="READ", coverage="unguarded")])
    assert R.build(Inventory(agents=[quiet])).chains == []


def test_summary_is_what_the_open_core_shows(inv):
    s = R.summary(R.build(inv))
    assert s["chains"] == 1 and s["top"]["nodes"][-1] == "moves money" and len(s["top"]["hops"]) == 3


def test_without_chains_the_direct_exposure_is_named():
    """A developer machine: several assistants as one user, each reads the web and runs commands.
    Nothing escalates (no chain), but the direct exposure is real and is shown."""
    from chimera_core.venom.render.screen import reach_block

    from .conftest import render

    def assistant(name, running):
        return Agent(f"assistant:{name}", name, "assistant", state="running" if running else "configured",
                     process_user="dev" if running else None, project=f"/home/dev/{name}",
                     tools=[Tool("WebFetch", "builtin", risk_class="EXTERNAL", coverage="unguarded"),
                            Tool("Bash", "builtin", risk_class="EXEC", coverage="unguarded")])

    inv = Inventory(agents=[assistant("zeta", False), assistant("alpha", True), assistant("beta", False)])
    g = R.build(inv)
    assert g.chains == [] and len(R.direct(g)) == 3
    src, agent, impact = R.strongest_direct(g)
    assert g.nodes[agent].label == "alpha" and impact == "impact:exec"  # the running one
    text = render(reach_block(inv), width=120)
    assert "no chain across agents" in text and "alpha" in text and "3 agents take untrusted input" in text
    assert R.summary(g)["direct"] == 3


def test_a_big_real_host_stays_readable():
    """Seen on a real laptop: one API with a hundred routes, dozens of agents, no chain."""
    from chimera_core.venom.render.topo import Topo

    api = Agent("code:/api", "api/backend", "code", state="running", project="/srv/api",
                triggers=[Trigger("inbound_http", f"/route/{i}") for i in range(100)],
                tools=[Tool("refund", "decorator", risk_class="SPEND", coverage="unguarded")])
    many = [Agent(f"code:/p{i}", f"project-{i}/agent", "code", project=f"/p{i}",
                  tools=[Tool("fetch_url", "decorator", risk_class="READ", coverage="unguarded",
                              params=[ToolParam("url")]),
                         Tool("run_command", "decorator", risk_class="EXEC", coverage="unguarded")])
            for i in range(40)]
    g = R.build(Inventory(agents=[api] + many))
    inputs = [n.label for n in g.nodes.values() if n.kind == "input"]
    assert sorted(inputs) == ["inbound HTTP · 100 routes", "web content"]  # one node per kind of input
    assert g.chains == []
    t = Topo(g, w=60, h=20, max_agents=12)
    assert t.chain is not None and [g.nodes[n].label for n in t.chain.nodes] == [
        "inbound HTTP · 100 routes", "api/backend", "moves money"]  # the strongest exposure is the hero
    agents = [p for p in t.placed.values() if p.kind == "agent"]
    assert len(agents) == 12 and t.hidden_agents == 29
    ys = {round(p.y / 6) for p in agents}
    assert len(ys) >= 4  # spread over the canvas, not piled in one row or column
    assert all(0 <= p.x <= t.W and 0 <= p.y <= t.H for p in t.placed.values())
    assert set(t.reached) >= {p.id for p in t.placed.values() if p.kind == "input"}  # every way in ignites


def test_the_scanner_never_prints_warnings_from_the_operators_files(tmp_path, capsys):
    from .conftest import run_cli

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "agent.py").write_text('from langchain.tools import tool\nX = "\\ "\n\n@tool\ndef run(cmd: str):\n    """runs"""\n    return cmd\n')
    import warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")  # a warning would be recorded here, if the scanner let one through
        run_cli(["venom", "--root", str(proj), "--workspace", str(tmp_path), "--no-anim", "--no-save"], capsys)
    assert not [w for w in seen if issubclass(w.category, SyntaxWarning)]


@pytest.mark.parametrize("n", [3, 300])
def test_scales_from_a_few_agents_to_hundreds(n):
    """Every agent on one account (each can reach every other): the search stays linear in the graph,
    and the map stays readable and quick. 30 agents used to take minutes with path enumeration."""
    import time

    from chimera_core.venom.model import Access, Credential
    from chimera_core.venom.render.topo import Topo

    agents = []
    for i in range(n):
        if i % 2 == 0:
            tools = [Tool("WebFetch", "builtin", risk_class="EXTERNAL", coverage="unguarded", params=[ToolParam("url")]),
                     Tool("Bash", "builtin", risk_class="EXEC", coverage="unguarded")]
            user, creds = "dev", []
        else:
            tools = [Tool("transfer", "decorator", risk_class="SPEND", coverage="unguarded")]
            user = "root" if i % 3 == 0 else "dev"
            creds = [Credential("AWS", "/x/.env", "cloud_root")] if i % 5 == 0 else []
        agents.append(Agent(f"code:/a{i}", f"agent-{i}", "code", state="running", process_user=user,
                            project=f"/home/dev/a{i}", tools=tools, access=Access(credentials=creds)))
    t0 = time.perf_counter()
    g = R.build(Inventory(agents=agents))
    topo = Topo(g, w=80, h=30, max_agents=14)
    topo.render(10.0, complete=True, labels=True)
    assert time.perf_counter() - t0 < 3.0
    assert g.chains and g.top is not None
    assert len([p for p in topo.placed.values() if p.kind == "agent"]) <= 14
