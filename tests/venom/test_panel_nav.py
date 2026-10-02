"""Management panel: navigation, search, exemptions and rule relaxing with live reload."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from chimera_core.venom import watch as W
from chimera_core.venom.controls import Controls
from chimera_core.venom.workspace import Workspace

from .conftest import FIXTURES, HOST_OPS, render, run_cli


@pytest.fixture
def ws(tmp_path, capsys, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(root), "--yes", "--activate"], capsys)
    monkeypatch.chdir(root)
    return Workspace(root)


def _guard(name, mode=None):
    from chimera_core.venom.observe import venom_guard
    return venom_guard(name, policy=f"policies/{name}.csl", mapping=f"policies/{name.replace('-', '_')}_mapping.py", mode=mode)


def _panel(ws):
    m = W.WatchModel()
    W.Tail(ws).poll(m)
    return W.ControlPanel(Controls(ws), m, W._inventory(ws), ws)


def _type(panel, text):
    for ch in text:
        panel.handle(ch)
    panel.handle("enter")


def _frame(panel, width=120):
    now = datetime.now(timezone.utc)
    return render(W.render(panel.model, panel.inv, {}, {}, now, now, width, 30, panel), width=width, height=30)


def test_navigation_open_back_search(ws):
    p = _panel(ws)
    assert p.crumbs() == ["Agents"]
    p.selected = p.agents().index("ingest-worker")
    p.handle("enter")
    assert p.view == "tools" and p.crumbs() == ["Agents", "ingest-worker", "Tools"]
    assert ("Esc", "back") in p.hints() and "Esc" in _frame(p)
    assert p.handle("q") is True and p.view == "tools"  # q does not quit inside a view
    p.handle("esc")
    assert p.view == "live" and p.crumbs() == ["Agents"]
    p.handle("/")
    _type(p, "claude")
    assert p.agents() == ["claude-code-ops", "claude-code-sandbox"] and "matching 'claude'" in p.crumbs()[0]
    p.handle("esc")
    assert p.query == "" and len(p.agents()) == 6  # every discovered agent, including the unmanaged one
    p.handle("tab")
    assert p.focus == "rules"
    p.handle("esc")
    assert p.focus == "agents"
    assert p.handle("q") is False


def test_exempt_agent_from_rule_reloads_live(ws):
    g = _guard("membership-bot", mode="block")
    assert not g.verify("transfer_funds", {"amount": 700, "to_wallet": "w"}).allowed  # approval rule fires
    p = _panel(ws)
    p.handle("tab")
    p.rule_index = [r for r, _ in p.rules()].index("transfer_funds_approval_over_100")
    p.handle("enter")
    assert p.view == "rule" and p.crumbs() == ["Rules", "transfer_funds_approval_over_100"]
    p.handle("x")
    _type(p, "treasury limits are enforced upstream")
    assert p.pending and "membership-bot" in p.pending[2]
    p.handle("y")
    assert "verified" in p.message[0], p.message
    text = (ws.policies / "membership-bot.csl").read_text()
    assert "STATE_CONSTRAINT transfer_funds_approval_over_100" not in text.replace("// STATE_CONSTRAINT", "")
    assert list((ws.venom / "history").glob("membership-bot-*.csl"))
    ex = [e for e in ws.load_exemptions() if e.scope == "rule"]
    assert ex and ex[0].rule == "transfer_funds_approval_over_100" and ex[0].reason
    r = g.verify("transfer_funds", {"amount": 700, "to_wallet": "w"})  # same running guard, new policy
    assert r.allowed
    assert not g.verify("transfer_funds", {"amount": 5000, "to_wallet": "w"}).allowed  # the ceiling still holds


def test_exempt_tool_removes_its_rules(ws):
    p = _panel(ws)
    p.selected = p.agents().index("ingest-worker")
    p.handle("enter")
    tools = [t for t, _ in p.tools("ingest-worker")]
    p.tool_index = tools.index("run_command")
    p.handle("e")
    _type(p, "batch host, commands are fixed")
    p.handle("y")
    assert "run_command" in p.message[0] and "verified" in p.message[0]
    g = _guard("ingest-worker", mode="block")
    assert g.verify("run_command", {"command": "anything"}).allowed
    assert not g.verify("write_file", {"path": "/etc/x", "content": "c"}).allowed


def test_exempt_whole_agent_and_back(ws):
    g = _guard("publisher", mode="block")
    assert not g.verify("post_to_page", {"text": "hi", "visibility": "public"}).allowed
    p = _panel(ws)
    p.selected = p.agents().index("publisher")
    p.handle("e")
    p.handle("enter")  # empty reason is refused
    assert "reason is required" in p.message[0]
    p.handle("e")
    _type(p, "marketing team owns this bot")
    p.handle("y")
    r = g.verify("post_to_page", {"text": "hi", "visibility": "public"})
    assert r.allowed and "__exempt__" in r.triggered_rule_ids
    assert "EXEMP" in _frame(p)
    assert any(e.agent == "publisher" and e.scope == "agent" for e in ws.load_exemptions())
    p.handle("e")
    p.handle("y")
    assert not g.verify("post_to_page", {"text": "hi", "visibility": "public"}).allowed


def test_adopted_policies_are_never_edited(tmp_path, capsys):
    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["setup", "--root", str(FIXTURES / "customer_051"), "--workspace", str(root), "--yes",
             "--strategy", "recommended"], capsys)
    from chimera_core.venom.policy.edit import policy_path_for
    path, why = policy_path_for(Workspace(root), "payments-agent")
    assert path is None and "edit it in your repository" in why


def test_edited_draft_activates_through_the_gate(ws):
    p = _panel(ws)
    active = ws.policies / "publisher.csl"
    draft = ws.drafts / "publisher.csl"
    ws.write_text(draft, active.read_text().replace("WHEN tool ==", "WHEN tool ==", 1) + "\n")
    p._apply("activate_draft", f"{draft}\x00{active}")
    assert not draft.exists() and list((ws.venom / "history").glob("publisher-*.csl"))


def test_long_agent_lists_scroll(ws):
    c = Controls(ws)
    c.set_many([f"agent-{i:03d}" for i in range(60)], "log")
    p = _panel(ws)
    p.selected = 40
    text = _frame(p)
    assert "more" in text and "▸" in text
    assert p.current() == p.agents()[40] and p.current() in text


def test_open_rule_policy_requests_the_studio(ws):
    g = _guard("membership-bot", mode="block")
    g.verify("transfer_funds", {"amount": 700, "to_wallet": "w"})
    p = _panel(ws)
    p.handle("tab")
    p.rule_index = [r for r, _ in p.rules()].index("transfer_funds_approval_over_100")
    p.handle("enter")
    p.handle("o")
    path, agent = p.studio_request
    assert agent == "membership-bot" and path.endswith("policies/membership-bot.csl")
    assert p.editor_request is None and not list(ws.drafts.glob("*.csl"))  # the studio makes its own draft


def test_live_map_toggles_and_draws_pulses(ws):
    import time

    _guard("membership-bot", mode="log").verify("transfer_funds", {"amount": 700, "to_wallet": "w"})
    p = _panel(ws)
    p.handle("g")
    assert p.map_on and p.crumbs()[-1] == "Map" and ("g", "stream") in p.hints()
    for rec in p.model.stream:
        rec["_seen"] = time.monotonic()  # as if it had just arrived
    text = _frame(p)
    assert "reach map" in text
    assert p.topo is not None and p.topo_size is not None
    p.handle("esc")
    assert not p.map_on and "reach map" not in _frame(p)
