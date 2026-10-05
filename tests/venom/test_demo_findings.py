"""Found while preparing a demo: tools without a file path, money tools named credit, sending an
invoice, decisions that replace a tool's rules, and stop messages that say their real cause."""

from __future__ import annotations

import importlib.util

import pytest

from chimera_core.venom.controls import Controls
from chimera_core.venom.observe import ApprovalPending, Blocked
from chimera_core.venom.policy import limits as L
from chimera_core.venom.watch import _inventory, _shown
from chimera_core.venom.workspace import Workspace

from .conftest import run_cli

AGENT = '''from langchain_core.tools import tool


@tool
def deploy(service: str, version: str) -> str:
    """Deploy a service."""
    return f"deployed {service} {version}"


@tool
def issue_credit(customer_id: str, amount: int) -> str:
    """Give a customer a credit."""
    return f"credited {amount}"


@tool
def send_invoice(customer_id: str, amount: int) -> str:
    """Email an invoice to a customer."""
    return "sent"


@tool
def write_file(path: str, text: str) -> str:
    """Write a file."""
    return "written"
'''


@pytest.fixture
def demo(tmp_path, capsys, monkeypatch):
    pytest.importorskip("langchain_core")
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "agent.py").write_text(AGENT)
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--no-anim"], capsys)
    assert rc == 0, out
    monkeypatch.chdir(root)
    return root, ws


def _agent(root):
    spec = importlib.util.spec_from_file_location(f"demo_{id(root)}_{(root / 'agent.py').stat().st_mtime_ns}",
                                                  root / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(fn, **kw):
    return fn.invoke(kw)


def _limits(ws, capsys, *flags):
    rc, out, _ = run_cli(["limits", "--agent", "repo", *flags, "--yes", "--workspace", str(ws)], capsys)
    assert rc == 0, out
    return out


def test_kinds_and_ordinary_calls_in_the_standard_profile(demo):
    root, ws = demo
    agent = next(a for a in _inventory(Workspace(ws)).agents if a.display_name == "repo")
    kinds = {t.name: L.kind_of(t) for t in agent.tools}
    assert kinds == {"deploy": "write", "issue_credit": "spend", "send_invoice": "send", "write_file": "write"}
    mod = _agent(root)
    assert _call(mod.deploy, service="api", version="1.2") == "deployed api 1.2"  # no path: no path rule
    assert _call(mod.issue_credit, customer_id="c", amount=10) == "credited 10"
    assert _call(mod.send_invoice, customer_id="c", amount=5_000) == "sent"  # a message, not money out
    assert _call(mod.write_file, path=str(root / "notes.txt"), text="x") == "written"
    big = _call(mod.issue_credit, customer_id="c", amount=50_000)
    assert isinstance(big, Blocked) and "Ask the operator to change the limits" in big  # a limit: that advice


def test_approval_replaces_a_tools_rules_and_really_waits(demo, capsys):
    root, ws = demo
    _limits(ws, capsys, "--decide", "deploy=approval", "--decide", "write_file=approval")
    mod = _agent(root)
    waiting = _call(mod.deploy, service="api", version="1.2")
    assert isinstance(waiting, ApprovalPending)
    outside = _call(mod.write_file, path="/etc/motd", text="x")  # its path rule is replaced, not added to
    assert isinstance(outside, ApprovalPending), outside
    from chimera_core.venom.approvals import Approvals

    pending = Approvals(Workspace(ws)).pending()
    assert {r["tool"] for r in pending} == {"deploy", "write_file"}
    assert all("approval=" not in _shown(r) for r in pending)  # the panel shows the call, not the flag
    Approvals(Workspace(ws)).decide(next(r["id"] for r in pending if r["tool"] == "deploy"), True, "me")
    assert _call(mod.deploy, service="api", version="1.2") == "deployed api 1.2"
    policy = (ws / ".csl/policies/repo.csl").read_text()
    assert "write_file_in_scope" not in policy and "write_file_always_needs_approval" in policy


def test_stop_messages_say_their_cause(demo, capsys):
    root, ws = demo
    _limits(ws, capsys, "--decide", "send_invoice=block")
    mod = _agent(root)
    never = _call(mod.send_invoice, customer_id="c", amount=1)
    assert isinstance(never, Blocked) and "the operator set send_invoice to never run" in never
    assert "Ask the operator to change the limits" not in never
    c = Controls(Workspace(ws))
    c.set_tool("repo", "deploy", True)
    off = _call(mod.deploy, service="api", version="1")
    assert "the operator turned deploy off" in off and "cslcore mode --agent repo --enable-tool deploy" in off
    c.set_disabled("repo", True)
    frozen = _call(mod.issue_credit, customer_id="c", amount=1)
    assert "the operator froze this agent; unfreeze: cslcore mode --agent repo --enable" in frozen
    assert "Ask the operator to change the limits" not in frozen


def test_the_hook_line_finds_cslcore_elsewhere_and_stops_without_it(tmp_path):
    """Claude Code lets a call through when a hook fails with any code but 2: the hook line tries
    the cslcore found at wiring time, then cslcore on PATH, and else exits 2."""
    import subprocess

    from chimera_core.venom.wiring import hook_command

    project = tmp_path / "proj"
    project.mkdir()
    line = hook_command("/nowhere/old-machine/bin/cslcore", "claude-code-proj", str(project), str(project))
    assert '"$CLAUDE_PROJECT_DIR"' in line and str(project) not in line
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "cslcore").write_text('#!/bin/sh\necho "ran: $*"\n')
    (fake / "cslcore").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "CLAUDE_PROJECT_DIR": str(project)}
    res = subprocess.run(["/bin/sh", "-c", line], env=env, capture_output=True, text=True)
    assert res.returncode == 0 and res.stdout.strip() == f"ran: hook --agent claude-code-proj --workspace {project}"
    res = subprocess.run(["/bin/sh", "-c", line], env={"PATH": "/usr/bin:/bin", "CLAUDE_PROJECT_DIR": str(project)},
                         capture_output=True, text=True)
    assert res.returncode == 2 and "the call is stopped" in res.stderr  # 2: Claude Code stops the call
    assert "cslcore" in line and " hook" in line  # still recognised as a cslcore hook
