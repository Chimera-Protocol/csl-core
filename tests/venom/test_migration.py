"""0.5.1 customers: adopt existing policies in place, test their own mappers, observe() parity."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from .conftest import FIXTURES, run_cli, scan_fixture

CUSTOMER = FIXTURES / "customer_051"
POLICY = CUSTOMER / "payments-agent" / "policies" / "agent_tool_guard.csl"
AGENT_PY = CUSTOMER / "payments-agent" / "agent.py"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def migrated(tmp_path, capsys):
    before = POLICY.read_bytes()
    rc, out, _ = run_cli(["setup", "--root", str(CUSTOMER), "--workspace", str(tmp_path), "--yes",
                          "--strategy", "recommended", "--activate"], capsys)
    assert rc == 0, out[-2000:]
    assert POLICY.read_bytes() == before
    return tmp_path, out


def test_existing_setup_detected_and_adopted(migrated):
    ws, out = migrated
    assert "Your code already uses CSL-Core" in out and "adopted, not copied" in out
    state = json.loads((ws / ".csl/venom/state.json").read_text())
    assert list(state["adopted"].values()) == [str(POLICY.resolve())]
    assert not list((ws / ".csl/policies").glob("*.csl"))  # nothing copied
    assert state["modes"]["payments-agent"]["mode"] == "block"  # it already enforces: kept in block
    wiring = (ws / ".csl/venom/wiring.md").read_text()
    assert 'observe(' in wiring and 'load_guard("policies/agent_tool_guard.csl")' in wiring


def test_rescan_counts_adopted_policy(migrated):
    ws, _ = migrated
    inv = scan_fixture(CUSTOMER, ws).inventory
    agent = inv.agent("payments-agent")
    assert {t.name for t in agent.tools} == {"TRANSFER_FUNDS", "SEND_EMAIL", "QUERY_DB"}
    assert all(t.coverage == "guarded" for t in agent.tools)
    assert next(p for p in inv.policies if p.path == str(POLICY.resolve())).status == "active"


def test_own_mapper_runs_isolated_and_fail_open_found(migrated, capsys):
    ws, _ = migrated
    rc, out, _ = run_cli(["map", "--agent", "payments-agent", "--test", "--mapping", f"{AGENT_PY}:agent_context_mapper",
                          "--root", str(CUSTOMER), "--workspace", str(ws)], capsys)
    assert rc == 3 and "fail-open" in out and "function only" in out
    assert "langchain_openai" not in out  # the agent module itself never ran
    assert "amount" in out


def test_mapper_needing_module_names_is_explained(tmp_path, migrated, capsys):
    ws, _ = migrated
    helper = tmp_path / "m.py"
    # built at import time: cannot be brought along without running the file
    helper.write_text("import json\nDEFAULTS = json.loads(open('x').read())\n\ndef mapper(tool_input):\n"
                      "    return {**DEFAULTS, **tool_input}\n")
    rc, out, err = run_cli(["map", "--agent", "payments-agent", "--test", "--mapping", f"{helper}:mapper",
                            "--root", str(CUSTOMER), "--workspace", str(ws)], capsys)
    assert "uses DEFAULTS" in out + err and "--import-module" in out + err


def test_mapper_literals_helpers_and_stdlib_come_along(tmp_path, migrated, capsys):
    """Literal constants, helper functions and standard-library imports are brought along; nothing else runs."""
    ws, _ = migrated
    helper = tmp_path / "m.py"
    marker = tmp_path / "ran"
    helper.write_text(f"import re\nopen({str(marker)!r}, 'w').write('x')\nDEFAULTS = {{'region': 'EU'}}\n\n"
                      "def _clean(v):\n    return re.sub(r'\\s+', '', str(v))\n\n"
                      "def mapper(tool_input):\n    return {**DEFAULTS, **{k: _clean(v) for k, v in tool_input.items()}}\n")
    rc, out, err = run_cli(["map", "--agent", "payments-agent", "--test", "--mapping", f"{helper}:mapper",
                            "--root", str(CUSTOMER), "--workspace", str(ws)], capsys)
    assert "MAPPING TEST" in out and "--import-module" not in out + err
    assert not marker.exists()  # the file's top-level code never ran


def test_observe_is_a_drop_in_in_block_mode(tmp_path):
    from chimera_core import ChimeraError, load_guard
    from chimera_core.venom.observe import observe

    with contextlib.redirect_stdout(io.StringIO()):
        plain = load_guard(str(POLICY))
        wrapped = observe(load_guard(str(POLICY)), agent="payments-agent", workspace=str(tmp_path))
    cases = json.loads((REPO / "examples/json_files/agent_tool_guard_tests.json").read_text())
    n = 0
    for group in ("allow_cases", "block_cases"):
        for case in cases.get(group, []):
            outcomes = []
            for g in (plain, wrapped):
                try:
                    r = g.verify(dict(case["input"]))
                    outcomes.append(("ok", r.allowed, tuple(r.violated_rule_ids)))
                except ChimeraError as e:
                    outcomes.append(("raised", e.constraint_name, tuple(e.result.violated_rule_ids) if e.result else ()))
            assert outcomes[0] == outcomes[1], case["name"]
            n += 1
    assert n >= 6
    log = (tmp_path / ".csl/venom/decisions/payments-agent.jsonl").read_text().splitlines()
    assert len(log) == n and all(json.loads(l)["mode"] == "block" for l in log)
