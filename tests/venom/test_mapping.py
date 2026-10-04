"""B14: fail-closed helpers, mapping assistant, mapping test."""

from __future__ import annotations


import pytest

from chimera_core.mapping import MappingError, guarded_verify, to_enum, to_flag, to_range

from .conftest import HOST_OPS, SENTINEL, run_cli, scan_fixture


# ---------------------------------------------------------------------------
# helpers (check 1)
# ---------------------------------------------------------------------------

def test_to_enum():
    assert to_enum("transfer_funds", {"transfer_funds": "TRANSFER_FUNDS"}) == "TRANSFER_FUNDS"
    assert to_enum("A", ["A", "B"]) == "A"
    for bad in ("a", "C", None, 3):
        with pytest.raises(MappingError):
            to_enum(bad, ["A", "B"])
    assert to_enum(None, ["A"], required=False, default="A") == "A"


def test_to_flag():
    assert to_flag(True) == "YES" and to_flag(False) == "NO"
    assert to_flag("yes") == "YES" and to_flag("0") == "NO" and to_flag("NO") == "NO"
    for bad in ("maybe", 2, None, 1.5, []):
        with pytest.raises(MappingError):
            to_flag(bad)
    assert to_flag(None, required=False, default="NO") == "NO"


def test_to_range():
    assert to_range(5, 0, 10) == 5 and to_range("5", 0, 10) == 5 and to_range(5.0, 0, 10) == 5
    for bad in (-1, 11, "5x", True, None, float("nan"), 5.5, "1e3"):
        with pytest.raises(MappingError):
            to_range(bad, 0, 10)
    assert to_range(2.5, 0, 10, integer=False) == 2.5


def test_guarded_verify_blocks_on_mapping_error():
    class G:
        def verify(self, ctx):
            raise AssertionError("must not be called")

    def mapping(tool, args, ctx):
        return {"tool": to_enum(tool, ["a"], name="tool")}

    r = guarded_verify(G(), mapping, "b", {})
    assert r.allowed is False and r.violated_rule_ids == ["__mapping__"]

    def broken(tool, args, ctx):
        raise KeyError("x")

    assert guarded_verify(G(), broken, "a", {}).allowed is False


def test_mapping_module_is_public_and_isolated():
    import subprocess
    import sys
    code = "import sys, chimera_core.mapping; assert 'chimera_core.venom' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------------------------------------------------------------------
# assistant and harness
# ---------------------------------------------------------------------------

def _setup_policy(tmp_path, capsys):
    args = ["--root", str(HOST_OPS), "--workspace", str(tmp_path)]
    run_cli(["venom", *args], capsys)
    run_cli(["policy", "fix", "membership-bot", "--yes", *args], capsys)
    run_cli(["policy", "activate", "membership", "--yes", *args], capsys)
    return args


def test_b14_generated_mapping_has_no_fail_open(tmp_path, capsys):
    args = _setup_policy(tmp_path, capsys)
    rc, out, _ = run_cli(["map", "--agent", "membership-bot", "--yes", *args], capsys)
    assert rc == 0 and "0 fail-open" in out
    path = tmp_path / ".csl/policies" / "membership_bot_mapping.py"
    assert path.exists()
    rc, out, _ = run_cli(["map", "--agent", "membership-bot", "--test", *args], capsys)
    assert rc == 0 and "0 fail-open" in out and "membership_bot_mapping.py" in out


def test_b14_generated_mappings_for_every_draft(tmp_path, capsys):
    args = ["--root", str(HOST_OPS), "--workspace", str(tmp_path)]
    run_cli(["venom", *args], capsys)
    run_cli(["policy", "new", "--all", "--yes", *args], capsys)
    for draft in sorted((tmp_path / ".csl/venom/drafts").glob("*.csl")):
        run_cli(["policy", "activate", draft.stem, "--yes", *args], capsys)
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    from chimera_core.venom.policy.draft import needs_policy
    for a in inv.agents:
        if not needs_policy(a):
            continue
        rc, out, _ = run_cli(["map", "--agent", a.display_name, "--yes", *args], capsys)
        assert rc == 0 and "0 fail-open" in out, (a.display_name, out[-600:])


def test_b14_raw_hand_written_mapping_is_flagged(tmp_path, capsys):
    args = _setup_policy(tmp_path, capsys)
    raw = tmp_path / "raw_mapping.py"
    raw.write_text('''
def map_call(tool_name, args, context=None):
    ctx = {"tool": tool_name}
    ctx.update(args)
    return ctx
''')
    rc, out, _ = run_cli(["map", "--agent", "membership-bot", "--test", "--mapping", str(raw), *args], capsys)
    assert rc == 3 and "fail-open" in out and "✗" in out
    import json
    state = json.loads((tmp_path / ".csl/venom/state.json").read_text())
    assert state["mapping_tests"]["code:/srv/membership-bot"]["fail_open"] > 0
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    assert any(f.id == "V11" for f in inv.findings)


def test_b14_sentinel_in_mapping_input_never_printed(tmp_path, capsys):
    args = _setup_policy(tmp_path, capsys)
    leaky = tmp_path / "leaky_mapping.py"
    leaky.write_text(f'''
from chimera_core.mapping import to_enum
def map_call(tool_name, args, context=None):
    # a careless mapping that echoes a raw input value (here a credential) into the policy context
    return {{"tool": to_enum(tool_name, ["transfer_funds", "check_balance"], name="tool"), "amount": 0,
             "requires_dual_approval": "{SENTINEL}"}}
''')
    rc, out, err = run_cli(["map", "--agent", "membership-bot", "--test", "--mapping", str(leaky), *args], capsys)
    assert SENTINEL not in out + err
    assert "(outside domain)" in out
    state = (tmp_path / ".csl/venom/state.json").read_text()
    assert SENTINEL not in state
