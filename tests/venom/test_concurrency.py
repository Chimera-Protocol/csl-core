"""Tool calls run in parallel (LangGraph's ToolNode runs a turn's calls in threads, async agents
interleave them). A call over a firm limit must stop in every interleaving, also as the first calls
of a guard that was just created: before 0.6.9 the control plane could hand a racing call the mode
it had before reading the state (log), and such a call ran."""

from __future__ import annotations

import threading

from chimera_core.venom.observe import venom_guard

from .conftest import run_cli

AGENT = '''def tool(fn):
    return fn


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money."""
    return f"sent {amount}"
'''


def test_parallel_first_calls_never_let_a_firm_stop_through(tmp_path, capsys):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "app/agent.py").write_text(AGENT)
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--limit", "repo.transfer_funds=1000", "--no-anim"], capsys)
    assert rc == 0, out
    through = 0
    for _ in range(60):
        guard = venom_guard("repo", workspace=str(ws))  # a fresh guard: its first calls race
        results = {}

        def call(i, amount):
            results[i] = (amount, guard.verify("transfer_funds", {"amount": amount, "to_wallet": "w"}).allowed)

        threads = [threading.Thread(target=call, args=(i, 5 if i % 2 else 5_000)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        through += sum(1 for amount, allowed in results.values() if amount == 5_000 and allowed)
        assert all(allowed for amount, allowed in results.values() if amount == 5)
    assert through == 0
