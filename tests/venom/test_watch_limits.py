"""From the live panel: l changes an agent's limits (its policy, mapping and check follow) and the
running agent obeys them on its next call, without a restart; w unwires it and wires it again."""

from __future__ import annotations

import importlib.util
import io
import shutil
import sys

import pytest

from chimera_core.venom.observe import Blocked

from .conftest import HOST_OPS, run_cli
from .test_limits import OPS_AGENT


@pytest.fixture
def room(tmp_path, capsys, monkeypatch):
    from chimera_core.cli import build_parser
    from chimera_core.venom.render.theme import THEME
    from chimera_core.venom.watch import WatchRoom
    from rich.console import Console

    host = tmp_path / "host"
    shutil.copytree(HOST_OPS, host)
    (host / "fs/srv/backoffice").mkdir()
    (host / "fs/srv/backoffice/agent.py").write_text(OPS_AGENT)
    (host / "fs/srv/backoffice/requirements.txt").write_text("langchain-core\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, _out, _ = run_cli(["setup", "--root", str(host), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                           "--wire"], capsys)
    assert rc == 0
    args = build_parser().parse_args(["watch", "--workspace", str(ws)])
    args.root = str(host)
    r = WatchRoom(args, Console(theme=THEME, file=io.StringIO(), width=130))
    r.panel.selected = r.panel.agents().index("backoffice")
    return r, host, ws


def _answers(monkeypatch, prompts, confirms=()):
    from rich.prompt import Confirm, Prompt

    p, c = list(prompts), list(confirms)
    # Enter gives the default, as at a real prompt
    monkeypatch.setattr(Prompt, "ask", classmethod(lambda cls, *a, **k: (p.pop(0) if p else "") or k.get("default", "")))
    monkeypatch.setattr(Confirm, "ask", classmethod(lambda cls, *a, **k: c.pop(0) if c else True))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


def _load(path):
    spec = importlib.util.spec_from_file_location(f"bo_{id(path)}_{path.stat().st_mtime_ns}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(fn, **kw):
    return fn.invoke(kw) if hasattr(fn, "invoke") else fn(**kw)


def test_l_changes_the_limits_of_a_running_agent(room, monkeypatch):
    r, host, _ws = room
    agent = _load(host / "fs/srv/backoffice/agent.py")  # running before the change
    assert isinstance(_call(agent.transfer_funds, amount=1_500, to_wallet="w"), Blocked)  # standard limits: above 100 needs an approval
    r.handle("l")
    assert r.external is not None
    # money limits (Enter keeps them), one change, done; no extra tool; block mode; back to the panel
    _answers(monkeypatch, ["", "", "transfer_funds=2k..4k", "", "", "block", ""])
    r.external()
    assert "new limits active" in r.panel.message[0]
    assert _call(agent.transfer_funds, amount=1_500, to_wallet="w") == "sent 1500"  # same process, no restart
    assert isinstance(_call(agent.transfer_funds, amount=4_500, to_wallet="w"), Blocked)


def test_w_unwires_and_wires_again(room, monkeypatch):
    r, host, ws = room
    path = host / "fs/srv/backoffice/agent.py"
    assert "_csl_guard.tool" in path.read_text()
    r.handle("w")
    assert r.panel.pending and r.panel.pending[0] == "unwire"
    r.handle("y")
    assert "_csl_guard" not in path.read_text() and path.read_text() == OPS_AGENT
    assert r.external is not None
    r.external()  # scans again: every view shows it unwired
    assert not r.panel.in_path("backoffice")
    r.handle("w")
    _answers(monkeypatch, [""], [True])
    r.external()
    assert "_csl_guard.tool" in path.read_text() and "wired" in r.panel.message[0]
    assert r.panel.in_path("backoffice")
