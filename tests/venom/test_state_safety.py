"""The control state under pressure. Tool calls read it in parallel (threads and processes) while
the operator changes modes, freezes, tool switches and limits. A frozen agent never gets a call
through, block mode never lets a call over a firm limit run, an unreadable state never reads as
"nothing set", writers never undo each other, and a tool call never waits on a writer. The lock
between writers works without fcntl too (Windows)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from chimera_core.venom.controls import Controls
from chimera_core.venom.observe import ApprovalPending, venom_guard
from chimera_core.venom.workspace import Workspace

from .conftest import run_cli

SRC = Path(__file__).resolve().parents[2]
AGENT = '''def tool(fn):
    return fn


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money."""
    return f"sent {amount}"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return "ok"
'''


@pytest.fixture
def wired(tmp_path, capsys):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "app/agent.py").write_text(AGENT)
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--limit", "repo.transfer_funds=1000", "--no-anim"], capsys)
    assert rc == 0, out
    return ws


def _allowed(guard, amount) -> bool:
    return guard.verify("transfer_funds", {"amount": amount, "to_wallet": "w"}).allowed


class Mutator(threading.Thread):
    """The operator, as fast as possible: tool switches, other agents' modes, limits (the policy
    file rewritten with another maximum, always below 5,000) and unrelated state; with `freeze`
    the agent stays frozen while its own mode flips between block and log."""

    def __init__(self, ws: Path, freeze: bool) -> None:
        super().__init__(daemon=True)
        self.ws, self.freeze, self.stop = Workspace(ws), freeze, threading.Event()
        self.writes = 0

    def run(self) -> None:
        c = Controls(self.ws)
        policy = self.ws.policies / "repo.csl"
        texts = [policy.read_text(), policy.read_text().replace("amount <= 1000", "amount <= 2000")]
        i = 0
        while not self.stop.is_set():
            i += 1
            c.set_tool("repo", "lookup_order", i % 2 == 0)
            c.set_mode(f"other-{i % 3}", "log" if i % 2 else "block")
            if self.freeze:
                c.set_mode("repo", "log" if i % 2 else "block")  # frozen whatever its mode
            else:
                c.set_mode("repo", "block")
            self.ws.write_text(policy, texts[i % 2])
            self.ws.update_state(churn=i)
            self.writes += 5


def _hammer(ws: Path, seconds: float, amounts, fresh_every: int = 5) -> dict:
    """Six threads calling for `seconds`; a new guard every few calls, so first calls race too."""
    allowed = {a: 0 for a in amounts}
    calls = [0]
    end = time.monotonic() + seconds
    lock = threading.Lock()

    def worker(seed: int) -> None:
        guard, n = None, 0
        while time.monotonic() < end:
            if guard is None or n % fresh_every == 0:
                guard = venom_guard("repo", workspace=str(ws))
            amount = amounts[(seed + n) % len(amounts)]
            ok = _allowed(guard, amount)
            with lock:
                calls[0] += 1
                allowed[amount] += ok
            n += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"calls": calls[0], "allowed": allowed}


def test_block_holds_while_everything_changes(wired):
    m = Mutator(wired, freeze=False)
    m.start()
    try:
        res = _hammer(wired, 4.0, [5, 5_000])
    finally:
        m.stop.set()
        m.join()
    assert res["calls"] > 100 and m.writes > 50
    assert res["allowed"][5_000] == 0, res
    assert res["allowed"][5] > 0  # ordinary calls kept running


def test_a_frozen_agent_gets_nothing_through_while_everything_changes(wired):
    Controls(Workspace(wired)).set_disabled("repo", True)
    m = Mutator(wired, freeze=True)
    m.start()
    try:
        res = _hammer(wired, 4.0, [5, 5_000])
    finally:
        m.stop.set()
        m.join()
    assert res["calls"] > 100 and m.writes > 50
    assert res["allowed"] == {5: 0, 5_000: 0}, res
    assert Controls(Workspace(wired)).get("repo").disabled  # no write undid the freeze


CALLER = r"""
import sys, time, json
sys.path.insert(0, sys.argv[1])
from chimera_core.venom.observe import venom_guard
ws, seconds = sys.argv[2], float(sys.argv[3])
amounts = [int(a) for a in sys.argv[4].split(",")]
end, n, allowed = time.monotonic() + seconds, 0, {a: 0 for a in amounts}
guard = None
while time.monotonic() < end:
    if n % 5 == 0:
        guard = venom_guard("repo", workspace=ws)
    a = amounts[n % len(amounts)]
    allowed[a] += guard.verify("transfer_funds", {"amount": a, "to_wallet": "w"}).allowed
    n += 1
print(json.dumps({"calls": n, "allowed": allowed}))
"""


@pytest.mark.parametrize("freeze", [False, True])
def test_processes_hold_too(wired, freeze):
    if freeze:
        Controls(Workspace(wired)).set_disabled("repo", True)
    procs = [subprocess.Popen([sys.executable, "-c", CALLER, str(SRC), str(wired), "4", "5,5000"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    m = Mutator(wired, freeze=freeze)
    m.start()
    try:
        outs = [p.communicate(timeout=120) for p in procs]
    finally:
        m.stop.set()
        m.join()
    results = [json.loads(o.strip().splitlines()[-1]) for o, _e in outs]
    assert all(r["calls"] > 20 for r in results), outs
    assert sum(r["allowed"]["5000"] for r in results) == 0, results
    if freeze:
        assert sum(r["allowed"]["5"] for r in results) == 0, results


def test_an_unreadable_state_keeps_what_was_known_else_stops(wired):
    state = wired / ".csl/venom/state.json"
    good = state.read_text()
    state.write_text("{not json")
    fresh = venom_guard("repo", workspace=str(wired))  # nothing known yet: nothing runs
    assert not _allowed(fresh, 5)
    state.write_text(good)
    assert _allowed(fresh, 5) and not _allowed(fresh, 5_000)
    state.write_text("{not json")  # known before: block mode is kept
    assert _allowed(fresh, 5) and not _allowed(fresh, 5_000)
    state.write_text(good)
    Controls(Workspace(wired)).set_disabled("repo", True)
    assert not _allowed(fresh, 5)
    state.write_text("")  # a frozen agent stays frozen when the state cannot be read
    assert not _allowed(fresh, 5)


def test_writers_never_undo_each_other(wired):
    """A writer that read the state before a freeze does not undo it when it saves."""
    w1, w2 = Workspace(wired), Workspace(wired)
    stale = w1.load_state()  # read earlier, e.g. by setup
    Controls(w2).set_disabled("repo", True)  # the operator freezes meanwhile
    stale["setup"] = {"note": "saved later"}
    w1.save_state(stale)
    assert Controls(Workspace(wired)).get("repo").disabled
    assert Workspace(wired).load_state()["setup"] == {"note": "saved later"}


def _count_in_parallel(ws: Path, threads: int = 4, each: int = 40) -> int:
    def bump() -> None:
        w = Workspace(ws)  # its own instance: only the file lock keeps writers apart
        for _ in range(each):
            with w.transaction() as state:
                state["counter"] = int(state.get("counter", 0)) + 1

    Workspace(ws).update_state(counter=0)
    ts = [threading.Thread(target=bump) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return Workspace(ws).load_state()["counter"]


def test_the_writer_lock_keeps_every_update(wired):
    assert _count_in_parallel(wired) == 160


def test_without_fcntl_the_lock_still_holds_and_the_guard_works(wired, monkeypatch):
    """As on Windows: fcntl cannot be imported (nor msvcrt, here), the lock file takes over."""
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", None)
    assert _count_in_parallel(wired) == 160
    Controls(Workspace(wired)).set_mode("repo", "block")
    guard = venom_guard("repo", workspace=str(wired))
    assert _allowed(guard, 5) and not _allowed(guard, 5_000)
    assert not (wired / ".csl/venom/state.held").exists()  # released


def test_a_tool_call_never_waits_on_a_writer(wired, capsys):
    """The panel holds the writer lock (stuck, say): calls go on; one that needs an approval is
    not approved and stops, within half a second, instead of waiting."""
    rc, out, _ = run_cli(["limits", "--agent", "repo", "--set", "transfer_funds=100..1000", "--yes",
                          "--workspace", str(wired), "--root", str(wired.parent / "repo")], capsys)
    assert rc == 0, out
    guard = venom_guard("repo", workspace=str(wired))
    ws = Workspace(wired)
    holding, release = threading.Event(), threading.Event()

    def writer() -> None:
        with ws.state_lock():
            holding.set()
            release.wait(10)

    t = threading.Thread(target=writer)
    t.start()
    holding.wait(5)
    try:
        start = time.monotonic()
        assert _allowed(guard, 5) and not _allowed(guard, 5_000)
        assert time.monotonic() - start < 1.0
        start = time.monotonic()
        result = guard.verify("transfer_funds", {"amount": 500, "to_wallet": "w"}, request_approval=True)
        assert not result.allowed and time.monotonic() - start < 2.0
        with pytest.raises(PermissionError) as e:
            guard.check("transfer_funds", {"amount": 500, "to_wallet": "w"})
        assert "approval" in str(e.value)
        assert isinstance(ApprovalPending("transfer_funds", ""), str)
    finally:
        release.set()
        t.join()
