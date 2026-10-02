"""cslcore studio: session logic, suggestions, replay, going live, and the app driven by keys."""

from __future__ import annotations

import asyncio
import json

import pytest

from chimera_core.venom.model import Inventory
from chimera_core.venom.studio import suggest as S
from chimera_core.venom.studio.engines import run_tla, run_z3
from chimera_core.venom.studio.session import StudioSession
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, run_cli

CONFLICT = open(__file__.replace("test_studio.py", "") + "../contract/probes/operators.csl").read().replace(
    "THEN t3 <= 40", "THEN d <= 40")


@pytest.fixture
def ws(tmp_path, capsys, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(root), "--yes", "--activate"], capsys)
    monkeypatch.chdir(root)
    return Workspace(root)


def _session(ws):
    return StudioSession(ws, Inventory.from_dict(ws.latest_inventory()))


def test_z3_conflict_suggestion_fixes_it():
    run = run_z3(CONFLICT)
    assert not run.ok and run.conflicts == [("when_gte", "when_gt")]
    best = S.ranked(S.from_z3(run, CONFLICT))[0]
    assert best.confidence == "HIGH" and best.applicable
    fixed = best.patch(CONFLICT)
    assert run_z3(fixed).ok


def test_unreachable_rule_suggests_widening():
    text = CONFLICT.replace("WHEN a <= 10", 'WHEN mode == "FOUR"').replace("THEN d <= 40", "THEN t3 <= 40")
    run = run_z3(text)
    assert run.ok and run.unreachable == ["when_lte"]  # a warning: compiles, but never triggers
    widen = next(s for s in S.from_z3(run, text) if s.title == 'Add "FOUR" to mode')
    after = run_z3(widen.patch(text))
    assert after.ok and not after.unreachable


def test_tla_reads_as_a_guard():
    """A rule reachable states break is a rule the guard enforces, not an error."""
    strict = open(HOST_OPS.parent.parent.parent.parent / "examples/tla_demo_violation.csl").read()
    loose = strict.replace("ENABLE_FORMAL_VERIFICATION: TRUE", "ENABLE_FORMAL_VERIFICATION: FALSE")
    run = run_tla(loose, use_real_tlc=False)
    assert run.engine == "BFS" and run.ok and not run.error
    assert run.enforced == ["user_no_transfer", "large_transfer_blocked"] and not run.refuses_to_load
    assert run.checked == 45 and 0 < run.blocked < run.checked
    g = run.guard["large_transfer_blocked"]
    assert g.blocked and g.example["tool"] == "TRANSFER_FUNDS" and g.example["amount"] > 50000
    assert not S.from_tla(run)  # nothing to fix
    # the same rules with TRUE: the compiler would refuse them, the first suggestion sets FALSE
    srun = run_tla(strict, use_real_tlc=False)
    assert srun.refuses_to_load
    fix = S.from_tla(srun)[0]
    assert fix.confidence == "HIGH" and fix.patch(strict) == loose
    # a rule the domain already keeps true never fires
    never = loose.replace("amount: 0..100000", "amount: 0..100")
    nrun = run_tla(never, use_real_tlc=False)
    assert "large_transfer_blocked" in nrun.never_fires
    items = S.from_tla(nrun)
    assert items and items[0].confidence == "LOW"
    assert run_z3(items[0].patch(never)).rules == ["user_no_transfer"]  # commented out, the rest stays


def test_live_policy_is_edited_as_a_draft_and_goes_live(ws):
    s = _session(ws)
    s.open(agent="membership-bot")
    assert s.state == "live" and s.agents == ["membership-bot"]
    edited = s.text.replace("amount <= 1000", "amount <= 300")
    p = s.save(edited)
    assert p == ws.drafts / "membership-bot.csl"
    assert "amount <= 1000" in (ws.policies / "membership-bot.csl").read_text()  # live is untouched
    assert s.state == "live · edited"
    from chimera_core.venom.observe import venom_guard
    g = venom_guard("membership-bot", mode="block")
    assert g.verify("transfer_funds", {"amount": 500, "to_wallet": "w"}, {"approval": "YES"}).allowed
    assert s.verify_z3(edited).ok
    res = s.go_live(edited)
    assert res.ok and res.bound == ["membership-bot"]
    assert "amount <= 300" in (ws.policies / "membership-bot.csl").read_text()
    assert list((ws.venom / "history").glob("membership-bot-*.csl")) and not (ws.drafts / "membership-bot.csl").exists()
    assert not g.verify("transfer_funds", {"amount": 500, "to_wallet": "w"}, {"approval": "YES"}).allowed  # switched live


def test_new_policy_bound_to_several_agents(ws):
    s = _session(ws)
    s.new(name="shared-claude")
    s.agents = ["claude-code-ops", "claude-code-sandbox"]
    text = s.text.replace('tool: {"read_file", "send_email"}', 'tool: {"Bash", "Read"}\n    agent_id: {"x"}') \
                 .replace('WHEN tool == "send_email"', 'WHEN tool == "Bash"')
    assert s.verify_z3(text).ok
    res = s.go_live(text)
    assert res.ok, res.message
    assert sorted(res.bound) == ["claude-code-ops", "claude-code-sandbox"]
    from chimera_core.venom.bindings import Bindings
    assert Bindings(ws).agents_of("policies/shared-claude.csl") == ["claude-code-ops", "claude-code-sandbox"]


def test_fit_and_replay(ws, capsys):
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(HOST_OPS.parent.parent.parent.parent / "scripts/venom_demo_traffic.py"),
                          "--workspace", str(ws.root), "--count", "400", "--seed", "7"], capture_output=True, text=True)
    assert out.returncode == 0 and "calls sent" in out.stdout, out.stdout + out.stderr
    s = _session(ws)
    s.open(agent="membership-bot")
    fits = s.fit(s.text)
    assert fits and fits[0].fail_open == 0 and fits[0].covered == 2
    stricter = s.text.replace("amount <= 1000", "amount <= 150")  # still above the other rule's 100
    r = s.replay(stricter)
    assert not r.error, r.error
    assert r.total > 0 and r.replayed > 0 and r.newly_blocked > 0 and r.newly_allowed == 0
    assert s.replay(s.text.replace("amount <= 1000", "amount <= 10")).error  # contradicts amount > 100


def test_app_keys_end_to_end(ws):
    from chimera_core.venom.studio.app import StudioApp

    s = _session(ws)
    s.open(agent="publisher")

    async def drive():
        app = StudioApp(s, use_real_tlc=False)
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            await pilot.press("f5")
            for _ in range(30):
                await pilot.pause(0.05)
                if s.z3 is not None:
                    break
            assert s.z3 is not None and s.z3.ok
            await pilot.press("ctrl+l")
            await pilot.pause(0.2)
            await pilot.press("y")
            await pilot.pause(0.3)
            assert "live:" in str(app.query_one("#status").render())
            await pilot.press("ctrl+q")
    asyncio.run(asyncio.wait_for(drive(), 60))
    assert json.loads((ws.venom / "state.json").read_text())["bindings"]["publisher"]["policy"] == "policies/publisher.csl"


def test_setup_policy_step_opens_studio_and_continues(tmp_path, capsys, monkeypatch):
    """Setup menu 'w': the studio opens on the agent's template draft; going live there is picked up."""
    import argparse
    from chimera_core.venom import setup as setup_mod
    from chimera_core.venom.studio import command as studio_cmd

    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(root)], capsys)
    seen = {}

    def fake_launch(ws, inv, path=None, agent=None, use_real_tlc=True):
        seen["path"], seen["agent"] = path, agent
        s = StudioSession(ws, inv)
        s.open(path, agent=agent)
        return s.go_live(s.text).message

    monkeypatch.setattr(studio_cmd, "launch", fake_launch)
    flow = setup_mod.Flow(argparse.Namespace(workspace=str(root), root=str(HOST_OPS), yes=False, plain=True, no_color=True,
                                              agent="membership-bot", strategy="choose"))
    flow.interactive = True
    inv = flow.load_inventory()
    agent = next(a for a in inv.agents if "membership" in a.id)
    monkeypatch.setattr(flow, "_menu", lambda a, cands: "w")
    assert flow.policies()
    key = "membership-bot"
    assert seen["agent"] == key and seen["path"].endswith(f"drafts/{key}.csl")
    st = flow.setup["agents"][agent.id]
    assert st["policy"] == f"policies/{key}.csl" and "draft" not in st
    assert json.loads((root / ".csl/venom/state.json").read_text())["bindings"][key]["policy"] == f"policies/{key}.csl"


def test_setup_verify_offers_a_studio_review(tmp_path, capsys, monkeypatch):
    """After template drafts, setup offers to review one in the studio before mapping."""
    import argparse
    from chimera_core.venom import setup as setup_mod
    from chimera_core.venom.studio import command as studio_cmd

    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(root)], capsys)
    opened = []

    def fake_launch(ws, inv, path=None, agent=None, use_real_tlc=True):
        opened.append(agent)
        s = StudioSession(ws, inv)
        s.open(path, agent=agent)
        return s.go_live(s.text).message

    monkeypatch.setattr(studio_cmd, "launch", fake_launch)
    flow = setup_mod.Flow(argparse.Namespace(workspace=str(root), root=str(HOST_OPS), yes=False, plain=True, no_color=True,
                                              strategy="templates"))
    flow.interactive = True
    assert flow.policies()
    answers = iter(["1", ""])
    monkeypatch.setattr(flow, "text_input", lambda q, d="": next(answers))
    assert flow.verify()
    assert len(opened) == 1
    st = next(st for st in flow.agents_state().values() if st.get("key") == opened[0])
    assert st["policy"] == f"policies/{opened[0]}.csl" and "draft" not in st
    assert "1 rules" not in capsys.readouterr().out
