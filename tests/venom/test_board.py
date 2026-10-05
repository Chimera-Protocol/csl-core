"""The protection board in setup: the riskiest agents first, one loop per agent (limits, policy,
wiring, check), standard protection for the rest with one key, and every stage read from disk so
it opens where it was left."""

from __future__ import annotations

import argparse
import importlib.util
import shutil

import pytest

from chimera_core.venom import board as B
from chimera_core.venom.observe import Blocked
from chimera_core.venom import setup as S
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, run_cli
from .test_check import DEV_AGENT
from .test_limits import OPS_AGENT


@pytest.fixture
def host(tmp_path, capsys):
    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    for name, src in (("devhelper", DEV_AGENT), ("backoffice", OPS_AGENT)):
        (host / f"fs/srv/{name}").mkdir()
        (host / f"fs/srv/{name}/agent.py").write_text(src)
        (host / f"fs/srv/{name}/requirements.txt").write_text("langchain-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["venom", "--root", str(host), "--workspace", str(ws), "--no-anim"], capsys)
    return host, ws


def _flow(host, ws, monkeypatch, texts, choices=None, **extra):
    flow = S.Flow(argparse.Namespace(workspace=str(ws), root=str(host), yes=False, plain=True, no_color=True, **extra))
    flow.interactive = True
    queue = list(texts)
    chosen = list(choices or [])
    monkeypatch.setattr(flow, "text_input", lambda q, d="": (queue.pop(0) if queue else "") or d)
    monkeypatch.setattr(flow, "choose", lambda q, c, d: chosen.pop(0) if chosen else d)
    monkeypatch.setattr(flow, "ask", lambda q, d: True)
    return flow


def _ordered(host, ws):
    from chimera_core.venom.policy.draft import needs_policy
    from chimera_core.venom.watch import _inventory

    w = Workspace(ws)
    inv = _inventory(w)
    agents = [a for a in inv.agents if needs_policy(a)]
    return B.rows_for(w, agents, (w.load_state().get("setup") or {}).get("agents"), argparse.Namespace(root=str(host)),
                      inv.policies)


def _rows(host, ws):
    return {r.key: r for r in _ordered(host, ws)}


def _number(host, ws, key):
    return str([r.key for r in _ordered(host, ws)].index(key) + 1)


def _load(path):
    spec = importlib.util.spec_from_file_location(f"m_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_riskiest_first_and_honest_about_what_is_not_protected(host):
    rows = list(_rows(*host).values())
    order = [r.risk for r in sorted(rows, key=lambda r: B.GROUPS.index(r.risk))]
    assert order == sorted(order, key=B.GROUPS.index)
    by = {r.key: r for r in rows}
    assert by["backoffice"].risk == "high" and "moves money" in by["backoffice"].does
    assert by["publisher"].risk == "medium"
    # a Claude Code hook that covers only Bash is not "wired"; nothing is protected before the board
    assert by["claude-code-ops"].wired == "none"
    assert not any(r.protected for r in rows)
    # an agent whose own code enforces a policy (0.5.1) is shown as such, not as unprotected
    assert by["membership-bot"].own == "membership.csl"
    assert "its code enforces" in str(by["membership-bot"].state)


def test_log_mode_is_never_shown_as_protected(host, monkeypatch, capsys):
    host_dir, ws = host
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "backoffice"), "", "c"], choices=["l", "log"])
    assert flow.policies()
    r = _rows(host_dir, ws)["backoffice"]
    assert r.wired == "wired" and r.mode == "log" and not r.protected and "recording only" in str(r.state)


def test_one_agent_then_the_rest_with_one_key(host, monkeypatch, capsys):
    host_dir, ws = host
    # the payments agent: its tool 3 (transfer_funds), 5 (limit its numbers), amount 1k..5k, go on; then Enter (a)
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "backoffice"), "3", "5", "1k", "5k", "", "a"],
                 choices=["l", "block"])
    assert flow.policies()
    rows = _rows(host_dir, ws)
    bo = rows["backoffice"]
    assert bo.protected and bo.check == "ok" and bo.mode == "block" and str(bo.state) == "protected"
    mod = _load(host_dir / "fs/srv/backoffice/agent.py")
    def call(fn, **kw):
        return fn.invoke(kw) if hasattr(fn, "invoke") else fn(**kw)

    assert call(mod.transfer_funds, amount=900, to_wallet="w") == "sent 900"
    assert isinstance(call(mod.transfer_funds, amount=9_000, to_wallet="w"), Blocked)
    for key in ("devhelper", "ingest-worker", "claude-code-ops", "claude-code-sandbox"):
        assert rows[key].protected, (key, rows[key].wired, rows[key].check, rows[key].mode)
    assert rows["publisher"].wired == "manual" and not rows["publisher"].protected
    assert "wire by hand" in str(rows["publisher"].state)
    assert not rows["membership-bot"].policy  # its own policy was left alone
    out = capsys.readouterr().out
    assert "Standard protection for" in out and "as its limits say" in out
    # the later steps leave what the board did alone
    st = flow.agents_state()
    assert all(s.get("protected") for s in st.values() if s.get("key") in ("backoffice", "devhelper"))


def test_board_opens_where_it_was_left(host, monkeypatch, capsys):
    host_dir, ws = host
    # one agent, then c: leave the board without standard protection for the rest (Enter would give it)
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "backoffice"), "", "c"], choices=["l", "block"])
    assert flow.policies()
    capsys.readouterr()
    flow = _flow(host_dir, ws, monkeypatch, ["c"])
    assert flow.policies()
    out = capsys.readouterr().out
    assert "1 of 7 protected" in " ".join(out.split())
    # a changed policy is no longer the one that was checked
    pol = ws / ".csl/policies/backoffice.csl"
    pol.write_text(pol.read_text() + "\n")
    assert _rows(host_dir, ws)["backoffice"].check == ""


def test_an_agent_with_its_own_policy_is_offered_to_keep_it(host, monkeypatch):
    host_dir, ws = host
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "membership-bot"), ""])
    monkeypatch.setattr(flow, "_menu", lambda a, cands: "1")  # keep (already in use)
    assert flow.policies()
    r = _rows(host_dir, ws)["membership-bot"]
    assert r.policy == "adopted" and r.protected and "its own code" in str(r.state)


def test_skip_leaves_an_agent_untouched(host, monkeypatch, capsys):
    """s on the board: no limits, no policy, no mode, no wiring and no change to its files; standard
    protection for the rest passes it by, and so do the later setup steps."""
    import json

    host_dir, ws = host
    agent_file = host_dir / "fs/srv/backoffice/agent.py"
    before = agent_file.read_text()
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "backoffice"), "a", ""], choices=["s", "block"])
    assert flow.policies()
    state = json.loads((ws / ".csl/venom/state.json").read_text())
    assert agent_file.read_text() == before
    assert not (ws / ".csl/policies/backoffice.csl").exists()
    assert "backoffice" not in (state.get("limits") or {})
    assert "backoffice" not in (state.get("modes") or {}) and "backoffice" not in (state.get("wiring") or {})
    assert "backoffice" not in (state.get("bindings") or {})
    rows = _rows(host_dir, ws)
    assert rows["backoffice"].skipped and "left untouched" in str(rows["backoffice"].state)
    assert rows["devhelper"].protected  # the rest got standard protection
    for step in ("verify", "map", "wire", "activate"):
        assert getattr(flow, step)()
    flow.wire_now()
    assert agent_file.read_text() == before and not (ws / ".csl/policies/backoffice.csl").exists()


def test_limits_for_data_and_any_number_from_menus(host, monkeypatch, capsys):
    """Not only money: a row count (export_rows.limit) and a list's length (notify.recipients) get
    limits from the menu, and a tool can be set to never run; no syntax to type."""
    host_dir, ws = host
    # backoffice: 1 export_rows, 2 notify, 3 transfer_funds
    answers = [_number(host_dir, ws, "backoffice"),
               "1", "5", "100", "1000",   # export_rows: limit its numbers, limit 100 free, never above 1000
               "2", "5", "5", "5",        # notify: recipients, at most 5
               "3", "3",                  # transfer_funds: never runs
               "", "c"]
    flow = _flow(host_dir, ws, monkeypatch, answers, choices=["l", "block"])
    assert flow.policies()
    out = capsys.readouterr().out
    flat = " ".join(out.split())
    for line in ("set an agent's limits yourself", "change what that tool may do, and limit its numbers",
                 "limit its numbers: limit", "never runs", "NUMBERS IT TAKES"):
        assert line in flat, line
    policy = (ws / ".csl/policies/backoffice.csl").read_text()
    assert "limit <= 1000" in policy and "recipients <= 5" in policy
    mod = _load(host_dir / "fs/srv/backoffice/agent.py")

    def call(fn, **kw):
        return fn.invoke(kw) if hasattr(fn, "invoke") else fn(**kw)

    assert call(mod.export_rows, table="t", limit=50) == "exported 50"
    assert isinstance(call(mod.export_rows, table="t", limit=5000), Blocked)
    assert isinstance(call(mod.notify, recipients=[f"p{i}" for i in range(9)], message="hi"), Blocked)
    assert isinstance(call(mod.transfer_funds, amount=1, to_wallet="w"), Blocked)


def test_the_choices_are_spelled_out(host, monkeypatch, capsys):
    host_dir, ws = host
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "devhelper"), "c"], choices=["s"])
    assert flow.policies()
    flat = " ".join(capsys.readouterr().out.split())
    assert "Enter protect them with standard limits" in flat and "c continue without protecting" in flat
    assert "l set its limits now" in flat and "o other ways" in flat and "s skip: change nothing for this agent" in flat


def test_an_older_cslcore_draft_is_not_called_yours(host, monkeypatch, capsys):
    host_dir, ws = host
    flow = _flow(host_dir, ws, monkeypatch, [_number(host_dir, ws, "backoffice"), "", "c"], choices=["l", "block"])
    assert flow.policies()
    pol = ws / ".csl/policies/backoffice.csl"
    pol.write_text(pol.read_text().replace("made from its limits", "drafted").replace("Generated by CSL-Core Venom",
                                                                                         "Generated by CSL-Core Venom 0.6.8;"))
    r = _rows(host_dir, ws)["backoffice"]
    assert r.policy == "older" and "yours" not in str(B.table([r], 120).columns[4]._cells)


def test_wiring_never_touches_files_outside_the_scanned_folder(tmp_path, capsys):
    """A workspace copied from another folder still names the original files in its scan record:
    wiring from it changes nothing there."""
    import json

    original = tmp_path / "original"
    (original / ".git").mkdir(parents=True)
    (original / "agent.py").write_text(OPS_AGENT)
    ws = tmp_path / "copy"
    ws.mkdir()
    run_cli(["setup", "--root", str(original), "--workspace", str(ws), "--yes", "--activate", "--no-anim"], capsys)
    state = json.loads((ws / ".csl/venom/state.json").read_text())
    state["scan_root"] = str(ws)  # as after copying the workspace elsewhere
    (ws / ".csl/venom/state.json").write_text(json.dumps(state))
    rc, out, _ = run_cli(["wire", "--yes", "--workspace", str(ws)], capsys)
    assert (original / "agent.py").read_text() == OPS_AGENT
    assert "outside the scanned folder" in " ".join(out.split())


@pytest.mark.parametrize("answer", [False, True])
def test_agents_outside_this_folder_are_listed_and_asked_about(host, monkeypatch, capsys, tmp_path, answer):
    """Without --root setup sees the whole machine: the board says which agents are outside the
    current folder, and Enter protects them only when asked (default no)."""
    host_dir, ws = host
    flow = _flow(host_dir, ws, monkeypatch, ["", "c"])
    flow.args.root = None  # as `cslcore setup` with no --root
    asked = []
    monkeypatch.setattr(flow, "ask", lambda q, d: asked.append((q, d)) or answer)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    import json
    state = json.loads((ws / ".csl/venom/state.json").read_text())
    state["scan_root"] = str(host_dir)  # the folder the scan covered (as setup records it)
    (ws / ".csl/venom/state.json").write_text(json.dumps(state))
    assert flow.policies()
    flat = " ".join(capsys.readouterr().out.split())
    assert "outside this folder (found elsewhere on this machine)" in flat
    assert asked and asked[0][0].startswith("Also protect the agents outside this folder?") and asked[0][1] is False
    written = (ws / ".csl/policies/backoffice.csl").exists()
    assert written is answer
