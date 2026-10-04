"""What the screens say: counted words agree with their numbers, triggers are named once, and an
agent-like process is named after its script or command, never after the shell that started it."""

from __future__ import annotations

import pytest

from chimera_core.venom.render.words import n
from chimera_core.venom.resolve import _process_name


def test_counted_words():
    assert n(1, "agent") == "1 agent" and n(2, "agent") == "2 agents" and n(0, "tool") == "0 tools"
    assert n(1, "policy") == "1 policy" and n(3, "policy") == "3 policies"


@pytest.mark.parametrize("args,name", [
    ("/bin/zsh -c claude -p fix", "claude"),
    ("/usr/bin/python3 /srv/app/agent.py --x", "app/agent.py"),
    ('zsh -lc "uv run langgraph dev"', "langgraph"),
    ("/opt/homebrew/bin/aider --model x", "aider"),
])
def test_process_names_are_not_shells(args, name):
    assert _process_name(args) == name


def test_inbound_http_is_said_once():
    from chimera_core.venom.analysis.findings import INBOUND

    assert f"inbound {INBOUND.get('inbound_http', 'inbound_http')}" == "inbound HTTP"


def _help(capsys, *argv):
    from chimera_core.cli import main

    with pytest.raises(SystemExit):
        main(list(argv))
    return capsys.readouterr().out


def test_help_starts_with_the_golden_path(capsys):
    out = _help(capsys, "--help")
    lines = out.splitlines()
    assert lines[0] == "Start here: cslcore setup"
    advanced = out.index("advanced:")
    for name in ("setup ", "watch ", "limits ", "venom map "):
        assert out.index(f"  {name}") < advanced
    assert "hook" not in out  # not listed
    assert "usage: cslcore hook" in _help(capsys, "hook", "--help")  # but still a command
