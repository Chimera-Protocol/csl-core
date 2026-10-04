"""Bypass tricks for derived values, hardened classifiers, regression cases."""

from __future__ import annotations

import json

import pytest

from chimera_core.mapping import command_allowed, destination_allowed, in_scope
from chimera_core.venom.mapping import tricks as T

from .conftest import FIXTURES, HOST_OPS, run_cli

BYPASS = FIXTURES / "mapping_bypass"
CHECKS = ["--classify", "path_ok=scope:file_path", "--classify", "cmd_ok=command:command",
          "--classify", "dest_ok=destination:url", "--allowed-root", "/srv/app",
          "--allowed-command", "git status", "--allowed-destination", "https://api.example.com/v1"]


@pytest.fixture
def ws(tmp_path, capsys):
    root = tmp_path / "ws"
    root.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(root), "--yes", "--activate", "--profile", "strict"], capsys)
    return root


def _map(ws, capsys, mapper, *extra):
    return run_cli(["map", "--agent", "claude-code:ops", "--policy", str(BYPASS / "ops.csl"), "--test",
                    "--mapping", f"{BYPASS / mapper}:classify", "--workspace", str(ws), *extra], capsys)


@pytest.mark.parametrize("kind,base,check", [
    ("scope", "/srv/app", lambda v: in_scope(v, ["/srv/app"])),
    ("command", "git status", lambda v: command_allowed(v, ["git status"])),
    ("destination", "https://api.example.com/v1", lambda v: destination_allowed(v, ["api.example.com"])),
    ("destination", "ops@example.com", lambda v: destination_allowed(v, ["ops@example.com"])),
    ("destination", "C024BE91L", lambda v: destination_allowed(v, ["C024BE91L"])),
])
def test_hardened_classifiers_hold_every_trick(kind, base, check):
    assert all(check(v) == "YES" for v in T.valid(kind, base))
    leaks = [(t.family, t.value) for t in T.tricks(kind, base) if check(t.value) == "YES"]
    assert leaks == []


def test_classifier_details():
    assert in_scope("/srv/app/a/../b.txt", ["/srv/app"]) == "YES"
    assert in_scope("/srv/app-evil/x", ["/srv/app"]) == "NO"
    assert in_scope("/anything", ["/"]) == "YES"
    assert command_allowed("git log -n 5", ["git log *"]) == "YES"
    assert command_allowed("sh -c id", ["sh *"]) == "NO"  # a wildcard never opens a shell
    assert command_allowed(["git", "status"], ["git status"]) == "YES"
    assert destination_allowed("https://a.hooks.example.com/x", ["*.hooks.example.com"]) == "YES"
    assert destination_allowed("https://hooks.example.com/x", ["*.hooks.example.com"]) == "NO"
    assert destination_allowed("https://api.example.com:8443/", ["api.example.com:8443"]) == "YES"
    assert destination_allowed("http://api.example.com/", ["api.example.com"]) == "NO"
    assert destination_allowed("http://api.example.com/", ["api.example.com"], schemes=("http", "https")) == "YES"
    assert destination_allowed(["a@example.com", "b@example.com"], ["a@example.com", "b@example.com"]) == "YES"


def test_generated_mappings_pass_the_tricks(ws, capsys):
    rc, out, _ = run_cli(["map", "--agent", "claude-code:ops", "--test", "--workspace", str(ws)], capsys)
    assert rc == 0 and "0 fail-open" in out and "BYPASS TRICKS" in out
    for family in ("chaining", "credentials"):  # strict profile: listed commands and destinations only
        assert family in out
    code = (ws / ".csl/policies" / "claude_code_ops_mapping.py").read_text()
    # writes are judged by the path classifier (tested in tests/test_actions.py), commands by the allowlist
    assert "args_path_class(" in code and "command_allowed(" in code and "posixpath" not in code


def test_hand_written_mapper_bypasses_are_found(ws, capsys):
    rc, out, _ = _map(ws, capsys, "mapper_naive.py", *CHECKS)
    assert rc == 3
    for leak in ('"/srv/app/../etc/passwd"', '"/srv/app-evil/file.txt"', '"git status; rm -rf /"',
                 '"https://api.example.com@evil.example/v1"'):
        assert leak in out, leak
    assert "chimera_core.mapping.in_scope" in out and "chimera_core.mapping.command_allowed" in out


def test_the_same_mapper_with_hardened_classifiers_passes(ws, capsys):
    rc, out, _ = _map(ws, capsys, "mapper_fixed.py", *CHECKS)
    assert rc == 0 and "0 fail-open" in out
    assert "✓ path_ok" in out and "✓ cmd_ok" in out and "✓ dest_ok" in out


def test_without_an_accepted_value_coverage_is_reported_not_claimed(ws, capsys):
    rc, out, _ = _map(ws, capsys, "mapper_fixed.py", "--classify", "path_ok=scope:file_path")
    assert "not covered" in out and "--allowed-root" in out
    assert "✓ path_ok" not in out


def test_regression_cases_and_keeping_them(ws, capsys):
    rc, out, _ = _map(ws, capsys, "mapper_naive.py", "--cases", str(BYPASS / "redteam.jsonl"))
    assert rc == 3 and "2 of 5 cases keep their decision" in out and "expected BLOCK, got ALLOW" in out
    rc, out, _ = _map(ws, capsys, "mapper_fixed.py", "--cases", str(BYPASS / "redteam.jsonl"), "--keep-cases")
    assert rc == 0 and "5 of 5" in out and "kept 5" in out
    assert "computes it itself" in out and "--classify path_ok=" in out  # not mistaken for caller context
    kept = ws / ".csl" / "venom" / "cases" / "claude-code-ops.jsonl"
    assert len(kept.read_text().splitlines()) == 5
    rc, out, _ = _map(ws, capsys, "mapper_naive.py")  # kept cases run on every later test
    assert rc == 3 and "REGRESSION" in out
    _map(ws, capsys, "mapper_fixed.py", "--cases", str(BYPASS / "redteam.jsonl"), "--keep-cases")
    assert len(kept.read_text().splitlines()) == 5  # no duplicates


def test_classify_problems_are_usage_errors(ws, capsys):
    rc, out, _ = _map(ws, capsys, "mapper_fixed.py", "--classify", "nope=scope", "--classify", "path_ok=folder")
    assert rc == 2 and "not a variable of the policy" in out and "scope, command or destination" in out


def test_case_file_errors_are_clear(ws, capsys, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"tool": "Bash", "expect": "BLOCK"}\nnot json\n')
    rc, out, err = _map(ws, capsys, "mapper_fixed.py", "--cases", str(bad))
    assert rc != 0 and "bad.jsonl:2" in out + err
    json.loads((BYPASS / "redteam.jsonl").read_text().splitlines()[0])


def test_setup_asks_for_the_checks_of_an_own_mapper(ws, capsys, monkeypatch):
    import argparse

    from chimera_core.venom import setup as setup_mod
    from chimera_core.venom.layers.governance import read_policy
    from chimera_core.venom.mapping.spec import build_spec

    flow = setup_mod.Flow(argparse.Namespace(workspace=str(ws), root=str(HOST_OPS), yes=False, plain=True, no_color=True))
    flow.interactive = True
    agent = next(a for a in flow.load_inventory().agents if a.display_name == "claude-code:ops")
    text = (BYPASS / "ops.csl").read_text()
    spec = build_spec(agent, read_policy(str(BYPASS / "ops.csl"), text, "active"))
    answers = iter(["path_ok=scope:file_path, cmd_ok=command:command", "/srv/app", "git status"])
    monkeypatch.setattr(flow, "text_input", lambda q, d="": next(answers))
    st = {}
    allowed = flow._mapper_checks(agent, spec, st)
    assert allowed == {"scope": ["/srv/app"], "command": ["git status"]}
    assert spec.classify["path_ok"] == ("scope", "file_path") and spec.classify["cmd_ok"] == ("command", "command")
    assert st["mapper_checks"]["classify"] == ["path_ok=scope:file_path", "cmd_ok=command:command"]
    # a later run reuses the answers without asking
    monkeypatch.setattr(flow, "text_input", lambda q, d="": pytest.fail(f"asked again: {q}"))
    spec2 = build_spec(agent, read_policy(str(BYPASS / "ops.csl"), text, "active"))
    assert flow._mapper_checks(agent, spec2, st) == allowed and "path_ok" in spec2.classify


def test_unused_imports_of_the_mapper_file_are_not_run(tmp_path):
    """A 0.5.1 agent file imports its framework at the top; the mapper itself does not need it."""
    from chimera_core.venom.mapping.assistant import isolated_function

    src = ("from chimera_core.plugins.not_installed_framework import guard_tools\nimport json\nimport re\n\n"
           "def mapper(tool_input):\n    return {'amount': int(re.sub(r'\\D', '', str(tool_input.get('amount', 0))) or 0)}\n")
    fn = isolated_function(src, "mapper", "agent.py")
    assert fn({"amount": "1,200"}) == {"amount": 1200}
