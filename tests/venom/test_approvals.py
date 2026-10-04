"""The approval band is real: a Claude Code hook answers "ask" (Claude Code asks the person), a
wired Python tool waits and returns ApprovalPending, the call shows in cslcore watch, and once a
person approves it the same call runs once. Firm limits still stop; an expired, used or unreadable
approval approves nothing."""

from __future__ import annotations

import io
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone

import pytest

from chimera_core.venom import watch as W
from chimera_core.venom.controls import Controls
from chimera_core.venom.observe import ApprovalPending
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, run_cli
from .test_limits import OPS_AGENT


@pytest.fixture
def env(tmp_path, capsys, monkeypatch):
    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    (host / "fs/srv/backoffice").mkdir()
    (host / "fs/srv/backoffice/agent.py").write_text(OPS_AGENT)
    (host / "fs/srv/backoffice/requirements.txt").write_text("langchain-core\ncsl-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(host), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--limit", "backoffice.transfer_funds=100..1000"], capsys)
    assert rc == 0
    monkeypatch.chdir(ws)
    return host, ws


def _agent(host):
    import importlib.util

    path = host / "fs/srv/backoffice/agent.py"
    spec = importlib.util.spec_from_file_location(f"bo_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(fn, **kw):
    return fn.invoke(kw) if hasattr(fn, "invoke") else fn(**kw)


def _panel(ws):
    w = Workspace(ws)
    m = W.WatchModel()
    W.Tail(w).poll(m)
    return W.ControlPanel(Controls(w), m, W._inventory(w), w)


def test_a_waiting_call_is_approved_in_watch_and_runs_once(env):
    host, ws = env
    agent = _agent(host)
    pending = _call(agent.transfer_funds, amount=500, to_wallet="w")
    assert isinstance(pending, ApprovalPending) and pending.tool == "transfer_funds" and "not run" in pending
    again = _call(agent.transfer_funds, amount=500, to_wallet="w")
    assert again.request_id == pending.request_id  # one request per call, not one per try
    with pytest.raises(PermissionError):  # above the maximum: no approval can let it through
        _call(agent.transfer_funds, amount=5_000, to_wallet="w")
    stored = (ws / ".csl/venom/approvals.json").read_text()
    assert '"w"' not in stored and "to_wallet" not in stored  # the arguments are never written

    p = _panel(ws)
    assert ("a", "1 waiting for approval") in p.hints()
    p.handle("a")
    assert p.view == "approvals" and p.approvals()[0]["shown"].get("amount") == 500
    p.handle("y")
    assert p.pending and p.pending[0] == "approve" and "runs once" in p.pending[2]
    p.handle("y")
    assert "approved" in p.message[0] and not p.approvals()

    assert isinstance(_call(agent.transfer_funds, amount=600, to_wallet="w"), ApprovalPending)  # another call
    assert _call(agent.transfer_funds, amount=500, to_wallet="w") == "sent 500"  # the approved one, once
    assert isinstance(_call(agent.transfer_funds, amount=500, to_wallet="w"), ApprovalPending)  # not twice


def test_denied_expired_and_unreadable_approvals_approve_nothing(env):
    host, ws = env
    agent = _agent(host)
    first = _call(agent.transfer_funds, amount=500, to_wallet="w")
    p = _panel(ws)
    p.handle("a")
    p.handle("n")
    p.handle("y")
    assert "denied" in p.message[0]
    assert isinstance(_call(agent.transfer_funds, amount=500, to_wallet="w"), ApprovalPending)
    # approved, but longer ago than the approval lasts
    second = _call(agent.transfer_funds, amount=500, to_wallet="w")
    assert second.request_id != first.request_id
    path = ws / ".csl/venom/approvals.json"
    data = json.loads(path.read_text())
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    data[second.request_id].update(status="approved", decided_at=old)
    path.write_text(json.dumps(data))
    assert isinstance(_call(agent.transfer_funds, amount=500, to_wallet="w"), ApprovalPending)
    # a file that cannot be read: fail closed
    path.write_text("{not json")
    assert isinstance(_call(agent.transfer_funds, amount=500, to_wallet="w"), ApprovalPending)


def test_claude_code_hook_asks_for_the_approval_band_and_denies_firm_stops(env, capsys, monkeypatch):
    host, ws = env
    rc, out, _ = run_cli(["limits", "--agent", "claude-code-ops", "--decide", "WebFetch=approval", "--yes",
                          "--root", str(host), "--workspace", str(ws)], capsys)
    assert rc == 0

    def hook(tool, tool_input):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"tool_name": tool, "tool_input": tool_input,
                                                                    "session_id": "s"})))
        rc, out, _ = run_cli(["hook", "--agent", "claude-code-ops", "--workspace", str(ws)], capsys)
        assert rc == 0
        return json.loads(out)["hookSpecificOutput"] if out.strip() else {}

    asked = hook("WebFetch", {"url": "https://example.com", "prompt": "read"})
    assert asked["permissionDecision"] == "ask" and "approval" in asked["permissionDecisionReason"]
    denied = hook("Write", {"file_path": "/etc/motd", "content": "x"})
    assert denied["permissionDecision"] == "deny"
    assert hook("Read", {"file_path": str(ws / "notes.txt")}) == {}  # an ordinary call: nothing to say


def test_an_agent_with_no_place_to_approve_is_told_so(env, capsys):
    host, ws = env
    rc, out, _ = run_cli(["limits", "--agent", "publisher", "--check", "--root", str(host), "--workspace", str(ws)],
                         capsys)
    assert "no place to approve" in " ".join(out.split()) and "such calls stop" in " ".join(out.split())
