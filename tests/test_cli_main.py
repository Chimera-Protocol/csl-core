from __future__ import annotations

import io
import json

from chimera_core.cli import main


def test_verify_success(agent_tool_guard_policy_path):
    rc = main(["verify", str(agent_tool_guard_policy_path)])
    assert rc == 0


def test_verify_missing_file():
    rc = main(["verify", "/nonexistent/policy.csl"])
    assert rc != 0


def test_verify_parse_error(tmp_path):
    bad = tmp_path / "bad.csl"
    bad.write_text("this is not valid CSL {{{", encoding="utf-8")
    rc = main(["verify", str(bad)])
    assert rc != 0


def test_simulate_allow(agent_tool_guard_policy_path, tmp_path):
    ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    in_file = tmp_path / "allow.json"
    in_file.write_text(json.dumps(ctx), encoding="utf-8")
    rc = main([
        "simulate", str(agent_tool_guard_policy_path),
        "--input-file", str(in_file),
        "--no-raise",
    ])
    assert rc == 0


def test_simulate_block(agent_tool_guard_policy_path, tmp_path):
    ctx = {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    in_file = tmp_path / "block.json"
    in_file.write_text(json.dumps(ctx), encoding="utf-8")
    rc = main([
        "simulate", str(agent_tool_guard_policy_path),
        "--input-file", str(in_file),
        "--no-raise",
    ])
    assert rc == 10


def test_simulate_dry_run_never_blocks(agent_tool_guard_policy_path, tmp_path):
    ctx = {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    in_file = tmp_path / "block.json"
    in_file.write_text(json.dumps(ctx), encoding="utf-8")
    rc = main([
        "simulate", str(agent_tool_guard_policy_path),
        "--input-file", str(in_file),
        "--dry-run", "--no-raise",
    ])
    assert rc == 0


def test_simulate_inline_json_input(agent_tool_guard_policy_path):
    ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    rc = main([
        "simulate", str(agent_tool_guard_policy_path),
        "--input", json.dumps(ctx),
        "--no-raise", "--quiet",
    ])
    assert rc == 0


def test_simulate_json_output(agent_tool_guard_policy_path, capsys):
    ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    rc = main([
        "simulate", str(agent_tool_guard_policy_path),
        "--input", json.dumps(ctx),
        "--no-raise", "--json", "--quiet",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Output also contains the banner + compiler progress lines before the JSON,
    # and rich soft-wraps long lines at terminal width; isolate and rejoin the JSON.
    json_start = out.index("{")
    payload = json.loads("".join(out[json_start:].splitlines()))
    assert payload["allowed"] is True


def test_simulate_default_input_when_none_given(agent_tool_guard_policy_path):
    rc = main(["simulate", str(agent_tool_guard_policy_path), "--no-raise", "--quiet"])
    assert rc in (0, 10)


def test_simulate_compile_error(tmp_path):
    bad = tmp_path / "bad.csl"
    bad.write_text("not valid csl", encoding="utf-8")
    rc = main(["simulate", str(bad), "--no-raise"])
    assert rc == 2


def test_formal_mock_engine(examples_dir):
    tla_policy = examples_dir / "tla_demo.csl"
    rc = main(["formal", str(tla_policy), "--mock"])
    assert rc == 0


def test_repl_allow_then_exit(agent_tool_guard_policy_path, monkeypatch):
    ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(ctx) + "\n"))
    rc = main(["repl", str(agent_tool_guard_policy_path)])
    assert rc == 0


def test_repl_invalid_json_then_exit(agent_tool_guard_policy_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json\n"))
    rc = main(["repl", str(agent_tool_guard_policy_path)])
    assert rc == 0


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "csl-core" in out
