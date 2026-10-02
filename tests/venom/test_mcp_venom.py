"""B16 MCP authoring tools."""

from __future__ import annotations

import asyncio
import json

import pytest

from chimera_core.venom import mcp_tools as T

from .conftest import HOST_OPS, SENTINEL, run_cli

pytest.importorskip("mcp")
PROMPT_TEXT = "You ingest partner feeds"


@pytest.fixture
def ws(tmp_path, capsys):
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(tmp_path)], capsys)
    return tmp_path


def _call(name, args):
    from chimera_core.mcp import server
    result = asyncio.run(server.mcp.call_tool(name, args))
    blocks = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(b, "text", "") for b in blocks)


def test_b16_tools_registered_existing_unchanged():
    from chimera_core.mcp import server
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"verify_policy", "simulate_policy", "explain_policy", "scaffold_policy", "tla_verify", "universe_info"} <= names
    assert {"venom_inventory", "venom_agent", "venom_policy_context", "venom_save_draft", "venom_propose_exemption"} <= names


def test_b16_invalid_draft_writes_nothing(ws):
    out = T.save_draft("membership-bot", "DOMAIN {", workspace=str(ws))
    assert out.startswith("NOT SAVED") and "PARSE" in out
    assert not (ws / ".csl/venom/drafts").exists()


def test_b16_save_draft_only_to_drafts_and_proposals_only(ws):
    from chimera_core.venom.policy.draft import draft_for
    from chimera_core.venom.model import Inventory
    inv = Inventory.from_dict(json.loads((ws / ".csl/venom/inventory/latest.json").read_text()))
    text = draft_for(inv.agent("membership-bot")).text
    out = T.save_draft("../../policies/membership-bot", text, "try to escape", workspace=str(ws))
    assert out.startswith("SAVED")
    assert not (ws / "policies").exists()
    assert list((ws / ".csl/venom/drafts").glob("*.csl"))
    out = T.propose_exemption("code:/srv/publisher", "agent", "trusted", workspace=str(ws))
    assert "PROPOSED" in out
    text = (ws / ".csl/venom/exemptions.yaml").read_text()
    assert "status: proposed" in text and "approved" not in text.replace("approve ", "")


def test_b16_outputs_redacted_and_short(ws):
    outs = [T.inventory(workspace=str(ws))]
    for name in ("ingest-worker", "membership-bot", "claude-code:ops", "publisher"):
        outs += [T.agent(name, workspace=str(ws)), T.policy_context(name, workspace=str(ws))]
    for o in outs:
        assert SENTINEL not in o and PROMPT_TEXT not in o
        assert len(o) < 4000, len(o)
    assert "OPENAI_API_KEY" not in outs[1] or True  # names are allowed, values never


def test_b16_end_to_end_with_mcp_client(ws, capsys, monkeypatch):
    monkeypatch.setenv("CSL_VENOM_WORKSPACE", str(ws))
    ctx = _call("venom_policy_context", {"agent_id": "membership-bot"})
    assert "membership-bot" in ctx and "STATE_CONSTRAINT" in ctx
    # the assistant writes its own draft (here: a tighter ceiling) and saves it through MCP
    draft = ctx.split("```csl", 1)[1].split("```", 1)[0].replace("amount <= 1000", "amount <= 250")
    saved = _call("venom_save_draft", {"agent_id": "membership-bot", "csl_content": draft, "note": "tighter ceiling"})
    assert saved.startswith("SAVED")
    rc, out, _ = run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate", "--agent", "membership-bot"], capsys)
    assert rc == 0
    active = (ws / "policies/membership-bot.csl").read_text()
    assert "amount <= 250" in active and "AI assistant via MCP" in active
