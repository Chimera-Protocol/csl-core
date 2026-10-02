"""Reach graph and chains: built only from discovery, guarded tools break them, escalations only."""

from __future__ import annotations

import copy

import pytest

from chimera_core.venom import reach as R
from chimera_core.venom.model import Agent, Exemption, Inventory, Tool, Trigger

from .conftest import HOST_OPS, scan_fixture


@pytest.fixture(scope="module")
def inv(tmp_path_factory):
    return scan_fixture(HOST_OPS, tmp_path_factory.mktemp("reach")).inventory


def _labels(g, chain):
    return [g.nodes[n].label for n in chain.nodes]


def test_strongest_chain_on_the_sample_host(inv):
    g = R.build(inv)
    assert g.top is not None
    assert _labels(g, g.top) == ["web content", "claude-code:ops", "ingest-worker", "root on the host"]
    hops = R.describe(g, g.top)
    assert "WebFetch" in hops[0] and "may work under /" in hops[1] and "runs as root" in hops[2]
    assert g.top.confidence == "likely"


def test_every_chain_escalates_through_at_least_two_agents(inv):
    g = R.build(inv)
    assert len(g.chains) == 9
    for c in g.chains:
        agents = [n for n in c.nodes if g.nodes[n].kind == "agent"]
        assert len(agents) >= 2 and g.nodes[c.nodes[0]].kind == "input" and g.nodes[c.nodes[-1]].kind == "impact"
        earlier = {e.dst for a in agents[:-1] for e in g.out(a)}
        assert c.nodes[-1] not in earlier  # no earlier agent could do it alone
    keys = [(c.nodes[0], c.nodes[1], c.nodes[-1]) for c in g.chains]
    assert len(keys) == len(set(keys))  # one route per input, entry agent and impact


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
    assert s["chains"] == 9 and s["top"]["nodes"][-1] == "root on the host" and len(s["top"]["hops"]) == 3


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
