"""Friction: a developer with a few agents finishes setup with a handful of Enters, and every agent
that can be wired automatically ends up protected (block mode, guard in its call path, check
passed). Each question asked is counted, so a new question in the main flow shows up here."""

from __future__ import annotations

import argparse
import io
import shutil
import sys

from rich.console import Console
from rich.prompt import Confirm, Prompt

from chimera_core.venom import board as B
from chimera_core.venom import setup as S
from chimera_core.venom.policy.draft import needs_policy
from chimera_core.venom.render.theme import THEME
from chimera_core.venom.watch import _inventory
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS

MAX_ENTERS = 5
NOTES_BOT = '''from langchain_core.tools import tool


@tool
def save_note(path: str, text: str) -> str:
    """Save a note."""
    return "saved"
'''


def test_a_solo_developer_finishes_with_a_few_enters(tmp_path, monkeypatch):
    host = tmp_path / "host"  # two Claude Code projects and one small LangChain helper
    shutil.copytree(HOST_OPS, host)
    for d in ("ingest-worker", "membership-bot", "publisher"):
        shutil.rmtree(host / "fs/srv" / d)
    (host / "fs/srv/notes-bot").mkdir()
    (host / "fs/srv/notes-bot/agent.py").write_text(NOTES_BOT)
    (host / "fs/srv/notes-bot/requirements.txt").write_text("langchain-core\ncsl-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    asked = []
    monkeypatch.setattr(Prompt, "ask", classmethod(lambda cls, q="", **k: asked.append(q) or k.get("default", "")))
    monkeypatch.setattr(Confirm, "ask", classmethod(lambda cls, q="", **k: asked.append(q) or k.get("default", False)))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    flow = S.Flow(argparse.Namespace(workspace=str(ws), root=str(host), yes=False, plain=True, no_color=True,
                                     no_anim=True))
    flow.interactive = True
    flow.console = Console(theme=THEME, file=io.StringIO(), width=120)
    assert flow.run() == 0
    assert len(asked) <= MAX_ENTERS, asked
    w = Workspace(ws)
    inv = _inventory(w)
    rows = B.rows_for(w, [a for a in inv.agents if needs_policy(a)], flow.agents_state(),
                      argparse.Namespace(root=str(host)), inv.policies)
    assert rows and all(r.protected for r in rows), [(r.key, str(r.state)) for r in rows]
