"""A scan is a snapshot. What happens when the host changes after it?

The guard: a tool that appears after the scan (a new plugin, a new MCP server) still passes
through the guard, and a tool name the mapping does not know is denied, not allowed.
The map: the next scan names every path that opened or closed since the last one.
"""

from __future__ import annotations

import io
import json
import shutil
import sys

import pytest

from chimera_core.venom import reach as R
from chimera_core.venom.model import Agent, Inventory, Tool

from .conftest import HOST_OPS, run_cli

HOOK = ["hook", "--agent", "claude-code-ops", "--policy", ".csl/policies/claude-code-ops.csl",
        "--mapping", ".csl/policies/claude_code_ops_mapping.py"]


@pytest.fixture
def wired(tmp_path, capsys, monkeypatch):
    """The sample host, scanned and wired in block mode, as an operator would leave it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate", "--mode", "block"], capsys)
    monkeypatch.chdir(ws)
    return ws


def _hook(monkeypatch, capsys, tool, tool_input, *extra):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"tool_name": tool, "tool_input": tool_input})))
    rc, out, _ = run_cli(HOOK + list(extra), capsys)
    assert rc == 0
    return json.loads(out)["hookSpecificOutput"] if out.strip() else None


def test_a_plugin_installed_after_the_scan_is_denied(wired, capsys, monkeypatch):
    for tool, args in (("mcp__newplugin__delete_bucket", {"bucket": "prod"}),
                       ("mcp__newplugin__list_items", {}),  # harmless sounding, still unknown
                       ("NewBuiltin", {"x": 1})):
        out = _hook(monkeypatch, capsys, tool, args)
        assert out["permissionDecision"] == "deny", tool
        assert "not one of the tools this agent was set up with" in out["permissionDecisionReason"], tool
    assert _hook(monkeypatch, capsys, "Read", {"file_path": "/srv/ops/a"}) is None  # known tools still work


def test_in_log_mode_the_unknown_tool_is_recorded_not_silent(wired, capsys, monkeypatch):
    run_cli(["mode", "--agent", "claude-code-ops", "log", "--workspace", str(wired)], capsys)
    assert _hook(monkeypatch, capsys, "mcp__newplugin__delete_bucket", {"bucket": "prod"}) is None
    log = [json.loads(line) for line in (wired / ".csl/venom/decisions/claude-code-ops.jsonl").read_text().splitlines()]
    assert log[-1]["decision"] == "WOULD_BLOCK" and log[-1]["rules"] == ["__mapping__"]


def test_a_tool_the_agent_never_had_is_denied(wired):
    """Another agent that changes membership-bot's code can give it new tools; calls to them still
    go through membership-bot's guard, which only knows the tools it was mapped for."""
    from chimera_core.venom.observe import venom_guard

    g = venom_guard("membership-bot", policy=".csl/policies/membership-bot.csl", mapping=".csl/policies/membership_bot_mapping.py")
    for tool in ("run_command", "refund_member"):
        r = g.verify(tool, {"amount": 5})
        assert not r.allowed and r.violated_rule_ids == ["__mapping__"]
    assert g.verify("transfer_funds", {"amount": 50, "to_wallet": "w"}).allowed


def test_the_guard_fails_closed_when_its_policy_is_gone(wired, capsys, monkeypatch):
    (wired / ".csl/policies/claude-code-ops.csl").unlink()
    out = _hook(monkeypatch, capsys, "Read", {"file_path": "/srv/ops/a"})
    assert out["permissionDecision"] == "deny" and "failing closed" in out["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# the map: what opened since the last scan
# ---------------------------------------------------------------------------

NEW_AGENT = '''from fastapi import FastAPI
import anthropic

app = FastAPI()
client = anthropic.Anthropic()
TOOLS = [{"name": "refund_payment", "description": "Refund a card payment.",
          "input_schema": {"type": "object", "properties": {"amount": {"type": "integer"}}, "required": ["amount"]}}]


@app.post("/hooks/refunds")
async def hook(payload: dict):
    return client.messages.create(model="m", max_tokens=1, tools=TOOLS, messages=[{"role": "user", "content": payload["t"]}])
'''


def _scan(host, ws, capsys, *extra):
    return run_cli(["venom", "--root", str(host), "--workspace", str(ws), "--no-anim", *extra], capsys)


def test_the_next_scan_names_what_opened(tmp_path, capsys):
    host, ws = tmp_path / "host", tmp_path / "ws"
    shutil.copytree(HOST_OPS, host)
    ws.mkdir()
    rc, out, _ = _scan(host, ws, capsys)
    assert "SINCE" not in out  # the first scan has nothing to compare with
    rc, out, _ = _scan(host, ws, capsys)
    assert "SINCE" in out and "no path opened or closed" in out

    new = host / "fs/srv/refunds"
    new.mkdir()
    (new / "agent.py").write_text(NEW_AGENT)
    (new / "requirements.txt").write_text("anthropic\n")
    rc, out, _ = _scan(host, ws, capsys, "--check", "--fail-on", "low", "--fail-on-new-reach")
    assert rc == 3 and "4 paths opened" in out and "1 new agent" in out and "1 new reach chain" in out
    assert "+ refunds  →  moves money" in out and "opened since the last scan" in out
    md = (ws / ".csl/venom/reports/latest.md").read_text()
    assert "### Since the last scan" in md and "- opened: refunds -> moves money (refund_payment without a rule)" in md
    data = json.loads((ws / ".csl/venom/reports/latest.json").read_text())["reach"]["since_last_scan"]
    assert data["new_agents"] == ["refunds"] and data["new_chains"] == 1
    assert {"path": "inbound HTTP /hooks/refunds -> refunds"}.items() <= data["opened"][2].items()
    assert data["top_new_chain"][-2:] == ["refunds", "moves money"]

    shutil.rmtree(new)  # and it closes again
    rc, out, _ = _scan(host, ws, capsys, "--json")
    since = json.loads(out)["reach"]["since_last_scan"]
    assert since["opened"] == [] and len(since["closed"]) == 4 and since["gone_agents"] == ["refunds"]
    assert since["closed_chains"] == 1


def test_another_scope_is_not_compared(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    _scan(HOST_OPS, ws, capsys)
    other = tmp_path / "proj"
    other.mkdir()
    (other / "agent.py").write_text(NEW_AGENT)
    rc, out, _ = _scan(other, ws, capsys)
    assert "SINCE" not in out


def test_a_rule_closes_the_path_it_decides():
    a = Agent("code:/a", "a", "code", tools=[Tool("pay", "decorator", risk_class="SPEND", coverage="unguarded")])
    before = R.build(Inventory(agents=[a]))
    a.tools[0].coverage = "guarded"
    d = R.diff(before, R.build(Inventory(agents=[a])), since="2026-10-01T09:00:00+00:00")
    assert [d.step(e) for e in d.closed] == ["a -> moves money"] and not d.opened and not d.new_agents
    assert R.diff_summary(d)["since"].startswith("2026-10-01")
