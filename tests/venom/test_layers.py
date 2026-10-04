"""B2-B7: discovery layers."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from chimera_core.venom.layers.code import analyze_source

from .conftest import HOST_EMPTY, HOST_OPS, render, scan_fixture

REPO = Path(__file__).resolve().parents[2]


def by_name(inv, name):
    a = inv.agent(name)
    assert a is not None, [x.display_name for x in inv.agents]
    return a


# ---------------------------------------------------------------------------
# B2 code layer
# ---------------------------------------------------------------------------

SHAPES = {
    "decorator": ('''
from langchain_core.tools import tool
@tool
def send_invoice(customer_id: int, amount: float, note: str = "") -> str:
    """Send an invoice."""
''', {"send_invoice": [("customer_id", "int"), ("amount", "float"), ("note", "str")]}),
    "decorator_named": ('''
from crewai.tools import tool
@tool("Search Docs")
def search(query: str) -> str:
    """Search."""
''', {"Search Docs": [("query", "str")]}),
    "Tool()": ('''
from langchain.tools import Tool
def lookup(q: str) -> str:
    return q
t = Tool(name="kb_lookup", func=lookup, description="Look up the knowledge base")
''', {"kb_lookup": [("q", "str")]}),
    "StructuredTool": ('''
from langchain_core.tools import StructuredTool
def refund(order_id: str, cents: int) -> str:
    return order_id
t = StructuredTool.from_function(func=refund, name="refund_order")
''', {"refund_order": [("order_id", "str"), ("cents", "int")]}),
    "openai_schema": ('''
from openai import OpenAI
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
''', {"get_weather": [("city", "string"), ("days", "integer")]}),
    "anthropic_schema": ('''
import anthropic
TOOLS = [{"name": "delete_user", "description": "Delete a user", "input_schema": {"type": "object", "properties": {"user_id": {"type": "string"}}}}]
''', {"delete_user": [("user_id", "string")]}),
    "mcp": ('''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("x")
@mcp.tool()
def run_query(sql: str, limit: int = 10) -> str:
    """Run a query."""
''', {"run_query": [("sql", "str"), ("limit", "int")]}),
    "openai_agents": ('''
from agents import function_tool
from typing import Literal
@function_tool
def set_mode(mode: Literal["fast", "safe"]) -> str:
    return mode
''', {"set_mode": [("mode", "str")]}),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_b2_tool_shapes(shape):
    src, expected = SHAPES[shape]
    cf = analyze_source(f"/x/{shape}.py", src)
    got = {t.name: [(p.name, p.type) for p in t.params] for t in cf.tools}
    assert got == expected


def test_b2_literal_enum_and_frameworks():
    src, _ = SHAPES["openai_agents"]
    cf = analyze_source("/x/a.py", src)
    assert cf.tools[0].params[0].enum == ["fast", "safe"]
    assert "openai-agents" in cf.frameworks


def test_b2_model_ids_prompt_routes_guard():
    cf = analyze_source("/x/a.py", '''
from fastapi import FastAPI
from chimera_core import load_guard
import anthropic
app = FastAPI()
SYSTEM_PROMPT = "You are a careful assistant that only reads data."
g = load_guard(".csl/policies/p.csl")
@app.post("/webhooks/slack/events")
def events(): return anthropic.Anthropic().messages.create(model="claude-sonnet-4-5")
''')
    assert cf.model_ids == ["claude-sonnet-4-5"]
    assert cf.prompt.present and cf.prompt.length == 49 and len(cf.prompt.sha256) == 16
    assert [(r.path, r.handler) for r in cf.routes] == [("/webhooks/slack/events", "events")]
    assert cf.guard_calls[0][0] == "load_guard" and cf.guard_calls[0][2] == ".csl/policies/p.csl"


def test_b2_parse_errors_counted_and_decoys_skipped(ops_inv):
    assert ops_inv.host.parse_errors == 1
    names = {t.name for a in ops_inv.agents for t in a.tools}
    assert "decoy_tool" not in names
    worker = by_name(ops_inv, "ingest-worker")
    assert {t.name for t in worker.tools} == {"run_command", "http_get", "write_file", "sideeffect_probe"}


def test_b2_performance_5000_files(tmp_path):
    from chimera_core.venom.layers.code import scan_code
    from chimera_core.venom.probe import LocalHostProbe
    from chimera_core.venom.scanner import Budget

    for i in range(50):
        d = tmp_path / f"pkg{i}"
        d.mkdir()
        for j in range(100):
            body = "from langchain_core.tools import tool\n\n@tool\ndef t%d(x: int) -> int:\n    return x\n" % j if j % 10 == 0 else "def f(x):\n    return x + 1\n"
            (d / f"m{j}.py").write_text(body)
    t0 = time.monotonic()
    cs = scan_code(LocalHostProbe("folder"), [str(tmp_path)], Budget(60))
    assert cs.files_scanned == 5000
    assert time.monotonic() - t0 < 10


# ---------------------------------------------------------------------------
# B3 config layer
# ---------------------------------------------------------------------------

def test_b3_bypass_permission_mode(ops_inv):
    assert by_name(ops_inv, "claude-code:sandbox").access.permission_mode == "bypass"


def test_b3_pretooluse_hook_wires_guard(ops_inv):
    a = by_name(ops_inv, "claude-code:ops")
    assert a.guard.mechanism == "hook" and a.guard.mode == "log" and a.guard.status in ("wired", "wired_no_rule")


def test_b3_filesystem_root(ops_inv):
    assert "/" in by_name(ops_inv, "claude-code:ops").access.fs_roots


def test_b3_credential_names_only(ops_inv):
    worker = by_name(ops_inv, "ingest-worker")
    names = {c.name: c.kind for c in worker.access.credentials}
    assert names["OPENAI_API_KEY"] == "llm_key"
    assert names["AWS_SECRET_ACCESS_KEY"] == "cloud_root"
    assert names["DATABASE_URL"] == "database_url"
    assert "LOG_LEVEL" not in names


def test_b3_sessions_counted_not_opened(tmp_path):
    result = scan_fixture(HOST_OPS, tmp_path)
    a = by_name(result.inventory, "claude-code:ops")
    assert a.sessions == 3 and a.runs.count == 3
    assert not [p for p in result.probe.opened if p.endswith(".jsonl")]


# ---------------------------------------------------------------------------
# B4 triggers
# ---------------------------------------------------------------------------

def test_b4_cron_attaches_time_trigger(ops_inv):
    pub = by_name(ops_inv, "publisher")
    assert any(t.type == "time" and t.schedule == "daily 06:00" for t in pub.triggers)
    assert pub.state == "scheduled"


def test_b4_webhook_messaging_trigger(ops_inv):
    pub = by_name(ops_inv, "publisher")
    assert any(t.type == "messaging" and t.schedule == "/webhooks/sms" for t in pub.triggers)


def test_b4_sources_unavailable_do_not_fail(tmp_path):
    inv = scan_fixture(HOST_EMPTY, tmp_path).inventory
    assert "triggers.crontab" in inv.host.layers_unavailable
    assert "triggers" in inv.host.layers_run


def test_b4_humanize_cron():
    from chimera_core.venom.layers.triggers import humanize_cron
    assert humanize_cron("0 6 * * *") == "daily 06:00"
    assert humanize_cron("*/15 * * * *") == "every 15m"
    assert humanize_cron("@reboot") == "reboot"


# ---------------------------------------------------------------------------
# B5 runtime and resolution
# ---------------------------------------------------------------------------

def test_b5_running_with_user_and_elevated(ops_inv):
    w = by_name(ops_inv, "ingest-worker")
    assert w.state == "running" and w.process_user == "root" and w.access.elevated
    assert "127.0.0.1:9100" in w.access.network_listen


def test_b5_stopped_and_scheduled(ops_inv):
    assert by_name(ops_inv, "membership-bot").state == "stopped"
    assert by_name(ops_inv, "publisher").state == "scheduled"


def test_b5_unmanaged_agent(ops_inv):
    a = by_name(ops_inv, "rogue/agent.py")
    assert a.kind == "unmanaged" and a.state == "running"


def test_b5_assistant_matched_by_cwd(ops_inv):
    ops = by_name(ops_inv, "claude-code:ops")
    assert ops.state == "running" and 4242 in ops.pids
    sandbox = by_name(ops_inv, "claude-code:sandbox")
    assert 4300 in sandbox.pids and sandbox.access.permission_mode == "bypass"


def test_b5_resolution_never_merges_distinct_entrypoints(ops_inv):
    entry = [a.entrypoint for a in ops_inv.agents if a.entrypoint and a.kind != "assistant"]
    assert len(entry) == len(set(entry))
    assert len({a.id for a in ops_inv.agents}) == len(ops_inv.agents)


# ---------------------------------------------------------------------------
# B6 history
# ---------------------------------------------------------------------------

def test_b6_no_evidence_is_na(ops_inv):
    bot = by_name(ops_inv, "membership-bot")
    assert bot.runs.count is None
    from chimera_core.venom.render.screen import agents_table
    assert "n/a" in render(agents_table(ops_inv, 100))


def test_b6_journal_counts(ops_inv):
    w = by_name(ops_inv, "ingest-worker")
    assert w.runs.count == 7
    assert w.runs.per_day == [0, 0, 0, 0, 2, 3, 2]
    assert "journalctl" in w.runs.source


def test_b6_since_window(tmp_path):
    inv = scan_fixture(HOST_OPS, tmp_path, window_days=30).inventory
    w = by_name(inv, "ingest-worker")
    assert w.runs.window == "30d" and len(w.runs.per_day) == 30 and w.runs.count == 7


# ---------------------------------------------------------------------------
# B7 governance
# ---------------------------------------------------------------------------

def test_b7_example_policies_parse():
    from chimera_core.venom.layers.governance import read_policy

    files = [f for f in sorted((REPO / "examples").glob("*.csl")) + sorted((REPO / "examples" / "community").glob("*.csl"))
             if f.is_file()]  # a local .csl/ workspace folder is not a policy
    assert len(files) == 20
    for f in files:
        ref = read_policy(str(f), f.read_text(), "found")
        assert ref.error is None, f
        assert ref.variables and ref.rules and ref.policy_hash


def test_b7_guard_linked_policy(ops_inv):
    bot = by_name(ops_inv, "membership-bot")
    assert bot.guard.mechanism == "wrapper"
    assert bot.guard.policy_ids == ["MembershipGuard"]


def test_b7_parse_error_reported():
    from chimera_core.venom.layers.governance import read_policy
    ref = read_policy("/x/bad.csl", "DOMAIN {", "found")
    assert ref.error


def test_b7_empty_workspace_first_install(tmp_path):
    inv = scan_fixture(HOST_EMPTY, tmp_path).inventory
    assert inv.policies == []
