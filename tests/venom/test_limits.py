"""Limits in the operator's own numbers, for money and for any other number a tool takes, enforced on
the agent's real functions: under the limit the call runs, over it the function never starts."""

from __future__ import annotations

import importlib.util
import json
import shutil

import pytest

from .conftest import HOST_OPS, run_cli

OPS_AGENT = '''"""Back office: pays members, exports reports, notifies people."""
from typing import List

try:
    from langchain_core.tools import tool
except ImportError:  # without LangChain the functions are called directly
    def tool(fn):
        return fn


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money from the treasury to a member wallet."""
    return f"sent {amount}"


@tool
def export_rows(table: str, limit: int) -> str:
    """Export rows of a reporting table."""
    return f"exported {limit}"


@tool
def notify(recipients: List[str], message: str) -> str:
    """Send a short notice to people."""
    return f"notified {len(recipients)}"
'''


@pytest.fixture
def ws(tmp_path, capsys, monkeypatch):
    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    agent = host / "fs/srv/backoffice"
    agent.mkdir()
    (agent / "agent.py").write_text(OPS_AGENT)
    (agent / "requirements.txt").write_text("langchain-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(host), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire"], capsys)
    assert rc == 0
    monkeypatch.chdir(ws)
    return ws, host


def _call(fn, **kw):
    return fn.invoke(kw) if hasattr(fn, "invoke") else fn(**kw)


def _load(path):
    spec = importlib.util.spec_from_file_location(f"backoffice_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_own_numbers_for_money_and_other_numbers(ws, capsys):
    ws, host = ws
    rc, out, _ = run_cli(["limits", "--agent", "backoffice", "--set", "transfer_funds=100k..300k",
                          "--set", "export_rows.limit=..1000", "--set", "notify.recipients=..5", "--yes",
                          "--root", str(host), "--workspace", str(ws)], capsys)
    assert rc == 0 and "✓ active" in out
    flat = " ".join(out.split())
    assert "up to 100,000 freely, up to 300,000 with approval, never above 300,000" in flat
    assert "limit never above 1,000" in flat and "recipients never above 5" in flat
    mod = _load(host / "fs/srv/backoffice/agent.py")
    assert _call(mod.transfer_funds, amount=50_000, to_wallet="w") == "sent 50000"
    for amount in (200_000, 500_000):  # above the free amount without an approval, and above the maximum
        with pytest.raises(PermissionError):
            _call(mod.transfer_funds, amount=amount, to_wallet="w")
    assert _call(mod.export_rows, table="sales", limit=500) == "exported 500"
    with pytest.raises(PermissionError):
        _call(mod.export_rows, table="sales", limit=5000)
    assert _call(mod.notify, recipients=["a", "b", "c"], message="hi") == "notified 3"
    with pytest.raises(PermissionError):
        _call(mod.notify, recipients=[f"p{i}" for i in range(20)], message="hi")
    state = json.loads((ws / ".csl/venom/state.json").read_text())["limits"]["backoffice"]["tools"]
    assert state["export_rows"]["numbers"] == {"limit": [1000, 1000]}


def test_limits_flags_in_setup_and_wrong_input(tmp_path, capsys):
    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    (host / "fs/srv/backoffice").mkdir()
    (host / "fs/srv/backoffice/agent.py").write_text(OPS_AGENT)
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(host), "--workspace", str(ws), "--yes", "--activate",
                          "--limit", "backoffice.transfer_funds=1m..2m", "--limit", "backoffice.export_rows.limit=..250"], capsys)
    assert rc == 0 and "transfer_funds: free up to 1,000,000, never above 2,000,000" in out
    policy = (ws / "policies/backoffice.csl").read_text()
    assert "amount <= 2000000" in policy and "limit <= 250" in policy
    rc, out, _ = run_cli(["limits", "--agent", "backoffice", "--set", "transfer_funds=300..100", "--workspace", str(ws),
                          "--root", str(host)], capsys)
    assert "the free amount (300) is above the maximum (100)" in out
