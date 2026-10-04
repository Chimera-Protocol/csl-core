"""One repository, several agents: each is found as its own agent (by its explicit definition, or by
the folder its tools are in), gets its own policy, and is wired with its own guard, also when two
agents' tools share one file."""

from __future__ import annotations

import importlib.util
import pytest

from .conftest import run_cli

FILES = {
    "requirements.txt": "langchain-core\nopenai-agents\n",
    # a folder per agent, no explicit definition
    "agents/payments/agent.py": '''"""Payments."""
from langchain_core.tools import tool


@tool
def refund_order(order_id: str, amount: int) -> str:
    """Refund an order."""
    return "refunded"
''',
    "agents/reports/agent.py": '''"""Reports."""
from langchain_core.tools import tool


@tool
def write_report(path: str, text: str) -> str:
    """Write a report file."""
    return "written"
''',
    # two explicit agents whose tools share one file
    "desk/agents.py": '''"""Billing and helpdesk."""
try:
    from agents import Agent, function_tool
except ImportError:
    def function_tool(fn):
        return fn

    def Agent(**kw):
        return kw


@function_tool
def charge_card(customer_id: str, amount: int) -> str:
    """Charge a customer's card."""
    return "charged"


@function_tool
def delete_ticket(ticket_id: str) -> str:
    """Delete a support ticket."""
    return "deleted"


billing = Agent(name="billing", tools=[charge_card])
helpdesk = Agent(name="helpdesk", tools=[delete_ticket])
''',
    # one agent whose tools live in a helper folder
    "ops/main.py": '''"""Ops runner."""
from langchain_core.tools import tool

from ops.tools.shell import run_shell


if __name__ == "__main__":
    print(run_shell)
''',
    "ops/tools/shell.py": '''from langchain_core.tools import tool


@tool
def run_shell(command: str) -> str:
    """Run a shell command."""
    return "ran"
''',
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    ws = tmp_path / "ws"
    ws.mkdir()
    return root, ws


def _agents(ws):
    from chimera_core.venom.workspace import Workspace

    data = Workspace(ws).latest_inventory()
    return {a["display_name"]: sorted(t["name"] for t in a["tools"]) for a in data["agents"]}


def _load(path):
    spec = importlib.util.spec_from_file_location(f"m_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_each_agent_is_found_on_its_own(repo, capsys):
    root, ws = repo
    rc, _out, _ = run_cli(["venom", "--root", str(root), "--workspace", str(ws), "--no-anim"], capsys)
    agents = _agents(ws)
    assert agents["billing"] == ["charge_card"]
    assert agents["helpdesk"] == ["delete_ticket"]
    assert agents["agents/payments"] == ["refund_order"]
    assert agents["agents/reports"] == ["write_report"]
    assert agents["repo/ops"] == ["run_shell"]  # tools/ belongs to the agent above it
    assert len(agents) == 5


def test_two_agents_in_one_file_each_keep_their_own_policy(repo, capsys):
    root, ws = repo
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--limit", "billing.charge_card=50..200"], capsys)
    assert rc == 0
    text = (root / "desk/agents.py").read_text()
    assert '_csl_guard = venom_guard("billing"' in text and '_csl_guard_helpdesk = venom_guard("helpdesk"' in text
    assert '@_csl_guard.tool("charge_card")' in text and '@_csl_guard_helpdesk.tool("delete_ticket")' in text
    assert text.count("from chimera_core.venom.observe import venom_guard") == 1
    mod = _load(root / "desk/agents.py")
    def call(fn, **kw):  # plain functions here (the fallback when the Agents SDK is absent)
        return fn(**kw)

    assert call(mod.charge_card, customer_id="c", amount=40) == "charged"
    with pytest.raises(PermissionError):
        call(mod.charge_card, customer_id="c", amount=500)  # billing's own limits
    from chimera_core.venom.observe import ApprovalPending

    assert isinstance(call(mod.delete_ticket, ticket_id="t"), ApprovalPending)  # helpdesk: deleting needs an approval
    # wiring again changes nothing
    rc, out, _ = run_cli(["wire", "--root", str(root), "--workspace", str(ws), "--yes"], capsys)
    assert (root / "desk/agents.py").read_text() == text
