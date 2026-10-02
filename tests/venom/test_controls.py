"""Management panel and control plane: live mode switch, kill switches, audit trail."""

from __future__ import annotations

import io
import json
import sys

import pytest

from chimera_core.venom import watch as W
from chimera_core.venom.controls import Controls
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, render, run_cli


@pytest.fixture
def wired(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate"], capsys)
    monkeypatch.chdir(ws)
    return Workspace(ws)


def _guard(name):
    from chimera_core.venom.observe import venom_guard
    return venom_guard(name, policy=f"policies/{name}.csl", mapping=f"policies/{name.replace('-', '_')}_mapping.py")


def _panel(ws):
    m = W.WatchModel()
    W.Tail(ws).poll(m)
    return W.ControlPanel(Controls(ws), m, W._inventory(ws))


def _select(panel, agent):
    names = panel.agents()
    panel.selected = names.index(agent)


def test_panel_switches_mode_live(wired):
    g = _guard("membership-bot")
    assert g.verify("transfer_funds", {"amount": 700}).allowed  # log mode: recorded, not blocked
    p = _panel(wired)
    _select(p, "membership-bot")
    p.handle("m")
    assert p.pending and "BLOCK" in p.pending[2]
    p.handle("y")
    assert Controls(wired).get("membership-bot").mode == "block"
    r = g.verify("transfer_funds", {"amount": 700})  # same running guard, no restart
    assert not r.allowed and "transfer_funds_approval_over_100" in r.violated_rule_ids
    with pytest.raises(PermissionError):
        g.check("transfer_funds", {"amount": 700})
    assert g.verify("transfer_funds", {"amount": 50}).allowed


def test_panel_cancel_changes_nothing(wired):
    p = _panel(wired)
    _select(p, "membership-bot")
    p.handle("d")
    p.handle("n")
    assert not Controls(wired).get("membership-bot").disabled
    assert p.message[0] == "cancelled"


def test_kill_switch_blocks_in_log_mode(wired):
    g = _guard("ingest-worker")
    assert g.verify("http_get", {"url": "x"}).allowed
    p = _panel(wired)
    _select(p, "ingest-worker")
    p.handle("d")
    p.handle("y")
    r = g.verify("http_get", {"url": "x"})
    assert not r.allowed and r.violated_rule_ids == ["__agent_disabled__"]
    p.handle("d")  # enabling needs no confirmation
    assert g.verify("http_get", {"url": "x"}).allowed
    log = [json.loads(l) for l in (wired.decisions / "ingest-worker.jsonl").read_text().splitlines()]
    assert [l["decision"] for l in log] == ["ALLOW", "BLOCK", "ALLOW"]


def test_tool_kill_switch_covers_mcp_alias_and_hook(wired, capsys, monkeypatch):
    p = _panel(wired)
    _select(p, "claude-code-ops")
    p.handle("t")
    tools = [t for t, _ in p.tools("claude-code-ops")]
    p.tool_index = tools.index("read_file")
    p.handle(" ")
    p.handle("y")
    assert "read_file" in Controls(wired).get("claude-code-ops").disabled_tools
    g = _guard("claude-code-ops")
    assert not g.verify("mcp__fs__read_file", {"path": "/etc/hosts"}).allowed
    assert g.verify("Read", {"file_path": "/srv/ops/a"}).allowed
    event = {"tool_name": "mcp__fs__read_file", "tool_input": {"path": "/etc/hosts"}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    rc, out, _ = run_cli(["hook", "--agent", "claude-code-ops", "--policy", "policies/claude-code-ops.csl",
                          "--mapping", "policies/claude_code_ops_mapping.py"], capsys)
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    p.handle(" ")  # enable again
    assert "read_file" not in Controls(wired).get("claude-code-ops").disabled_tools


def test_audit_trail_and_cli(wired, capsys):
    run_cli(["mode", "--agent", "publisher", "block"], capsys)
    run_cli(["mode", "--agent", "publisher", "--disable"], capsys)
    run_cli(["mode", "--agent", "publisher", "--disable-tool", "post_to_page"], capsys)
    rc, out, _ = run_cli(["mode"], capsys)
    assert "publisher" in out and "DISABLED" in out and "post_to_page" in out
    actions = [a["action"] for a in Controls(wired).audit_tail() if a["agent"] == "publisher"]
    assert actions == ["bind", "mode block", "disable agent", "disable tool"]  # setup bound it, then the operator acted
    assert Controls(wired).default_mode() == "log"  # chosen once during setup, for all agents


def test_panel_render_shows_controls(wired):
    Controls(wired).set_disabled("publisher", True)
    p = _panel(wired)
    _select(p, "publisher")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    text = render(W.render(p.model, p.inv, {}, {}, now, now, 120, 30, p), width=120, height=30)
    assert "OFF" in text and "▸" in text and "m mode" in text
    p.handle("?")
    text = render(W.render(p.model, p.inv, {}, {}, now, now, 120, 30, p), width=120, height=30)
    assert "kill switch" in text


def test_bulk_modes(wired, capsys):
    from chimera_core.venom.controls import Controls
    run_cli(["mode", "--agent", "membership-bot", "block"], capsys)
    rc, out, _ = run_cli(["mode", "--all", "log"], capsys)  # stdin is not a terminal: one agent would stop blocking
    assert "stop blocking" in out and Controls(wired).get("membership-bot").mode == "block"
    rc, out, _ = run_cli(["mode", "--all", "log", "--yes"], capsys)
    assert Controls(wired).get("membership-bot").mode == "log" and Controls(wired).get("never-seen").mode == "log"
    rc, out, _ = run_cli(["mode", "--match", "claude-code-*", "block"], capsys)
    c = Controls(wired)
    assert c.get("claude-code-ops").mode == "block" and c.get("claude-code-sandbox").mode == "block"
    assert c.get("publisher").mode == "log"


def test_panel_switch_all(wired):
    p = _panel(wired)
    p.handle("M")
    assert "ALL" in p.pending[2]
    p.handle("y")
    c = Controls(wired)
    assert c.default_mode() == "block" and c.get("publisher").mode == "block"
