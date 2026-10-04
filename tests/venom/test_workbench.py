"""B13 policy workbench (and B10 check 5: exemptions encoded in generated policies)."""

from __future__ import annotations


from chimera_core.venom import model as M
from chimera_core.venom.policy import draft as D
from chimera_core.venom.policy.gate import verify_text

from .conftest import HOST_OPS, run_cli, scan_fixture


def _args(tmp_path):
    return ["--root", str(HOST_OPS), "--workspace", str(tmp_path)]


def _prime(tmp_path, capsys):
    run_cli(["venom", *_args(tmp_path)], capsys)


def test_b13_first_install_drafts_verify(tmp_path, capsys):
    _prime(tmp_path, capsys)
    rc, out, _ = run_cli(["policy", "new", "--all", "--yes", *_args(tmp_path)], capsys)
    assert rc == 0
    inv = scan_fixture(HOST_OPS, tmp_path / "scratch").inventory
    expected = {D.agent_key(a) for a in inv.agents if D.needs_policy(a)}
    drafts = {p.stem: p for p in (tmp_path / ".csl/venom/drafts").glob("*.csl")}
    assert set(drafts) == expected and expected
    for p in drafts.values():
        rc, _, _ = run_cli(["verify", str(p)], capsys)
        assert rc == 0, p


def test_b13_tool_enum_equals_discovered_names(ops_inv):
    for a in ops_inv.agents:
        if not D.needs_policy(a):
            continue
        d = D.draft_for(a)
        enum = D.re.findall(r'"([^"]*)"', d.variables["tool"])
        assert enum == [t.name for t in a.tools if not t.name.endswith("/*")]


def test_b13_fix_renames_with_diff_and_verifies(tmp_path, capsys):
    _prime(tmp_path, capsys)
    original = (HOST_OPS / "fs/srv/membership-bot/policies/membership.csl").read_bytes()
    rc, out, _ = run_cli(["policy", "fix", "membership-bot", "--yes", *_args(tmp_path)], capsys)
    assert rc == 0
    assert '-    tool: {"TRANSFER_FUNDS", "CHECK_BALANCE"}' in out and '+    tool: {"transfer_funds", "check_balance"}' in out
    text = (tmp_path / ".csl/venom/drafts/membership.csl").read_text()
    assert '"transfer_funds"' in text and "TRANSFER_FUNDS" not in text
    assert verify_text(text).ok
    # the hand-written policy itself is untouched (B13 check 6)
    assert (HOST_OPS / "fs/srv/membership-bot/policies/membership.csl").read_bytes() == original


def test_b13_failing_draft_cannot_be_activated(tmp_path, capsys):
    _prime(tmp_path, capsys)
    drafts = tmp_path / ".csl/venom/drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / "broken.csl").write_text('''CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}
DOMAIN Broken {
  VARIABLES {
    tool: {"a", "b"}
    amount: 0..100
  }
  STATE_CONSTRAINT low {
    WHEN tool == "a"
    THEN amount <= 10
  }
  STATE_CONSTRAINT high {
    WHEN tool == "a"
    THEN amount >= 50
  }
}
''')
    rc, out, _ = run_cli(["policy", "activate", "broken", "--yes", *_args(tmp_path)], capsys)
    assert rc == 4
    assert "CONTRADICTION" in out and "cannot be activated" in out
    assert not (tmp_path / ".csl/policies" / "broken.csl").exists()


def test_b13_declining_and_plan_only_write_nothing(tmp_path, capsys):
    _prime(tmp_path, capsys)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    run_cli(["policy", "new", "--agent", "membership-bot", *_args(tmp_path)], capsys)  # stdin is not a tty: declined
    run_cli(["policy", "new", "--agent", "membership-bot", "--yes", "--plan-only", *_args(tmp_path)], capsys)
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.name != "state.json")
    assert [p for p in after if "drafts" in str(p)] == []
    assert set(after) <= set(before)


def test_b13_activate_then_guarded(tmp_path, capsys):
    _prime(tmp_path, capsys)
    run_cli(["policy", "fix", "membership-bot", "--yes", *_args(tmp_path)], capsys)
    rc, out, _ = run_cli(["policy", "activate", "membership", "--yes", *_args(tmp_path)], capsys)
    assert rc == 0 and (tmp_path / ".csl/policies/membership.csl").exists()
    assert not (tmp_path / ".csl/venom/drafts/membership.csl").exists()
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    bot = inv.agent("membership-bot")
    assert {t.name: t.coverage for t in bot.tools} == {"check_balance": "guarded", "transfer_funds": "guarded"}
    assert not [d for d in inv.drift if d.kind == "unknown_value" and d.agent_id == bot.id]


def test_b13_extend_adds_rules_for_uncovered_tools(tmp_path, capsys):
    _prime(tmp_path, capsys)
    ws_pol = tmp_path / ".csl/policies"
    ws_pol.mkdir()
    (ws_pol / "ingest-worker.csl").write_text('''CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}
DOMAIN Ingest {
  VARIABLES {
    tool: {"http_get"}
  }
  STATE_CONSTRAINT no_sideeffect {
    ALWAYS True
    THEN tool MUST NOT BE "sideeffect_probe"
  }
}
''')
    rc, out, _ = run_cli(["policy", "extend", "ingest-worker", "--agent", "ingest-worker", "--yes", *_args(tmp_path)], capsys)
    assert rc == 0
    text = (tmp_path / ".csl/venom/drafts/ingest-worker.csl").read_text()
    assert "run_command_allowlist" in text and "write_file_in_scope" in text and '"run_command"' in text
    assert verify_text(text).ok
    assert "run_command" not in (ws_pol / "ingest-worker.csl").read_text()


def test_b10_exemption_encoded_in_generated_policy(ops_inv):
    bot = ops_inv.agent("membership-bot")
    plain = D.draft_for(bot)
    ex = [M.Exemption("assistant:claude-code:/srv/ops", "agent", None, "operator session", "aytug", None, "approved")]
    exempted = D.draft_for(bot, ex, agent_ids={"assistant:claude-code:/srv/ops": "claude-code-ops"})
    assert 'agent_id != "claude-code-ops"' in exempted.text
    g1, g2 = verify_text(plain.text), verify_text(exempted.text)
    assert g1.ok and g2.ok and g1.policy_hash != g2.policy_hash


def test_b13_policy_list_and_show(tmp_path, capsys):
    _prime(tmp_path, capsys)
    rc, out, _ = run_cli(["policy", "list", *_args(tmp_path)], capsys)
    assert rc == 0 and "MembershipGuard" in out
    rc, out, _ = run_cli(["policy", "show", "membership-bot", *_args(tmp_path)], capsys)
    assert rc == 0 and "STATE_CONSTRAINT" in out and "transfer_limit" in out
