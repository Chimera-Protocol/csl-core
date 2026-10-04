"""The policy check: sample calls made from an agent's limits, decided by its real policy and mapping in
block mode. What the limits allow runs, what they do not allow stops, and a policy or a mapping that
does not do what the limits say shows up as a failed row."""

from __future__ import annotations

import re
import shutil

import pytest

from chimera_core.venom import check
from chimera_core.venom.watch import _inventory
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, run_cli

DEV_AGENT = '''"""Dev helper: runs commands and queries, writes files, mails people, removes accounts."""
from typing import List

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):
        return fn


@tool
def run_command(command: str) -> str:
    """Run a shell command."""
    return "ran"


@tool
def run_sql_query(query: str) -> str:
    """Run a SQL query on the database."""
    return "rows"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file."""
    return "written"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return "sent"


@tool
def delete_user(user_id: str) -> str:
    """Delete a user account."""
    return "deleted"


@tool
def notify(recipients: List[str], message: str) -> str:
    """Send a short notice to people."""
    return "notified"
'''


@pytest.fixture
def env(tmp_path, capsys):
    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    agent = host / "fs/srv/devhelper"
    agent.mkdir()
    (agent / "agent.py").write_text(DEV_AGENT)
    (agent / "requirements.txt").write_text("langchain-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["setup", "--root", str(host), "--workspace", str(ws), "--yes", "--activate", "--mode", "block"],
                         capsys)
    assert rc == 0
    return host, ws, out


def _agent(ws, name="devhelper"):
    w = Workspace(str(ws))
    return w, next(a for a in _inventory(w).agents if a.display_name == name)


def _rows(report):
    return {(c.tool, c.what, c.approval): c for c in report.cases}


def test_every_kind_is_decided_as_its_limits_say(env):
    _host, ws, out = env
    w, agent = _agent(ws)
    report = check.run(w, agent)
    assert report.ok and not report.note, [(c.tool, c.what, c.got) for c in report.failed]
    rows = _rows(report)
    assert rows[("run_command", "an ordinary command", False)].got == check.RUNS
    for words in check.COMMAND_WORDS.values():  # every harmful category stops, approval or not
        assert rows[("run_command", words, True)].got == check.STOPPED
    assert rows[("run_sql_query", "a writing query", False)].got == check.STOPPED
    assert rows[("run_sql_query", "a writing query", True)].got == check.RUNS
    assert rows[("run_sql_query", "a destructive query", True)].got == check.STOPPED
    assert rows[("write_file", "a file inside its folder", False)].got == check.RUNS
    assert rows[("write_file", "a file outside its folder", True)].got == check.STOPPED
    assert rows[("delete_user", "a call", False)].got == check.STOPPED
    assert rows[("delete_user", "a call", True)].got == check.RUNS
    # categories are given as values: no sample call carries command or query text
    for c in report.cases:
        for v in c.args.values():
            assert not (isinstance(v, str) and re.search(r"\b(rm|curl|sudo|drop|delete|select)\b", v, re.I))
    # and setup shows the table at its end
    assert "Check: sample calls decided by each active policy" in out
    assert "devhelper" in out and "as its limits say" in out


def test_limits_change_is_checked_right_away(env, capsys):
    host, ws, _ = env
    rc, out, _ = run_cli(["limits", "--agent", "devhelper", "--set", "notify.recipients=2..5", "--decide",
                          "delete_user=block", "--profile", "strict", "--yes", "--root", str(host), "--workspace", str(ws)],
                         capsys)
    assert rc == 0 and "✓ active" in out
    flat = " ".join(out.split())
    assert "as its limits say" in flat and "recipients 6" in flat and "not on the list" in flat
    w, agent = _agent(ws)
    rows = _rows(check.run(w, agent))
    assert rows[("notify", "recipients 2", False)].got == check.RUNS
    assert rows[("notify", "recipients 3", False)].got == check.STOPPED
    assert rows[("notify", "recipients 5", True)].got == check.RUNS
    assert rows[("notify", "recipients 6", True)].got == check.STOPPED
    assert rows[("delete_user", "any call", True)].got == check.STOPPED
    assert rows[("run_command", "a command not on the list", True)].got == check.STOPPED
    rc, out, _ = run_cli(["limits", "--check", "--root", str(host), "--workspace", str(ws)], capsys)
    assert rc == 0 and "devhelper" in out


def test_a_mapping_that_does_not_classify_is_caught(env, capsys):
    host, ws, _ = env
    mapping = ws / "policies/devhelper_mapping.py"
    text = mapping.read_text()
    assert "args_command_class(args, SCOPE_ROOTS)" in text
    mapping.write_text(text.replace("args_command_class(args, SCOPE_ROOTS)", '"OK"'))  # never looks at the command
    w, agent = _agent(ws)
    report = check.run(w, agent)
    assert not report.ok
    assert {c.tool for c in report.failed} == {"run_command"}
    assert all(c.expected == check.STOPPED and c.got == check.RUNS for c in report.failed)
    rc, out, _ = run_cli(["limits", "--agent", "devhelper", "--check", "--root", str(host), "--workspace", str(ws)], capsys)
    assert rc == 3 and "not as its limits say" in out and "✗" in out


def test_a_policy_missing_a_rule_is_caught(env):
    _host, ws, _ = env
    policy = ws / "policies/devhelper.csl"
    text = policy.read_text()
    text2 = re.sub(r'(?s)\n\s*STATE_CONSTRAINT write_file_in_scope \{.*?\n\s*\}\n', "\n", text)
    assert text2 != text
    policy.write_text(text2)
    w, agent = _agent(ws)
    report = check.run(w, agent)
    assert {c.tool for c in report.failed} == {"write_file"}


def test_a_hand_written_policy_is_left_to_the_studio(env):
    _host, ws, _ = env
    policy = ws / "policies/devhelper.csl"
    policy.write_text(policy.read_text().replace("made from its limits", "written by me"))
    w, agent = _agent(ws)
    report = check.run(w, agent)
    assert not report.cases and "by hand" in report.note
