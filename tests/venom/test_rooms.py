"""One flow: the end of a scan, the reach map and the live panel lead into each other."""

from __future__ import annotations

import io
import time

import pytest
from rich.console import Console

from chimera_core.venom import rooms
from chimera_core.venom.render import mapview as M
from chimera_core.venom.render.theme import THEME

from .conftest import HOST_OPS, render, run_cli, scan_fixture


@pytest.fixture
def ws(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate"], capsys)
    return ws


def _console(width=130, height=40):
    return Console(theme=THEME, file=io.StringIO(), width=width, height=height, force_terminal=True)


def _args(ws):
    from chimera_core.cli import build_parser
    return build_parser().parse_args(["watch", "--workspace", str(ws)])


def _play(room, seconds, width=130, height=40):
    """Frames over `seconds` of animation, by moving the room's clocks back."""
    for name in ("leave", "arrive"):
        v = getattr(room, name, None)
        if v is not None:
            setattr(room, name, (v[0] - seconds, v[1]))
    for name in ("grow0", "settle0"):
        v = getattr(room, name, None)
        if v is not None:
            setattr(room, name, v - seconds)
    return render(room.frame(time.monotonic(), width, height), width=width)


def test_the_map_shrinks_into_the_panel_and_grows_back(ws):
    inv = scan_fixture(HOST_OPS, ws.parent / "scan").inventory
    m = M.MapRoom(inv, _console())
    m.enter("scan")
    assert m.arrive is not None and m.spread0 < time.monotonic() - 90  # complete, growing out of one point
    assert "REACH CHAIN" in _play(m, 0.2)
    _play(m, 1.0)
    assert m.arrive is None
    m.handle("w")
    assert "w watch" in render(m.frame(time.monotonic(), 130), width=130)
    m.handle("down")  # keys wait while it leaves
    _play(m, 1.0)
    assert m.exit_to == "watch"

    from chimera_core.venom.watch import WatchRoom

    w = WatchRoom(_args(ws), _console())
    w.enter("map")
    assert w.panel.map_on and w.settle0 is not None
    assert "reach map" in _play(w, 0.1)
    _play(w, 1.0)
    assert w.settle0 is None and w.panel.pane_zoom == 1.0
    w.handle("g")  # the map again: the full map
    assert w.grow0 is not None
    _play(w, 1.0)
    assert w.exit_to == "map" and w.panel.pane_zoom == 1.0
    m.enter("watch")
    assert m.arrive[1] == "watch"


def test_the_map_opened_on_its_own_still_spreads(ws):
    inv = scan_fixture(HOST_OPS, ws.parent / "scan").inventory
    m = M.MapRoom(inv, _console())
    m.enter("command")
    assert m.arrive is None and m.spread0 > time.monotonic() - 1


def test_freeze_is_x_everywhere_and_d_still_works(ws, monkeypatch):
    from chimera_core.venom import watch as W
    from chimera_core.venom.controls import Controls
    from chimera_core.venom.workspace import Workspace

    monkeypatch.chdir(ws)
    p = W.ControlPanel(Controls(Workspace(ws)), W.WatchModel(), W._inventory(Workspace(ws)))
    p.selected = p.agents().index("ingest-worker")
    assert ("x", "freeze") in p.hints()
    p.handle("x")
    assert p.pending and p.pending[2].startswith("Freeze ingest-worker")
    p.handle("y")
    assert Controls(Workspace(ws)).get("ingest-worker").disabled and "FROZEN" in p.message[0]
    p.selected = p.agents().index("ingest-worker")  # a frozen agent moves in the list
    p.handle("d")  # the key before 0.6.6 unfreezes too
    assert not Controls(Workspace(ws)).get("ingest-worker").disabled


def test_after_a_scan_a_person_is_asked_where_to_go(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim"], capsys)
    assert rc == 0 and "reach map" not in out.split("REPORT")[-1]  # a pipe is never asked

    from chimera_core.venom import commands

    asked = []
    monkeypatch.setattr(rooms, "interactive", lambda console: True)
    monkeypatch.setattr(rooms, "ask_next", lambda console, choices: asked.append(choices) or "m")
    opened = []
    monkeypatch.setattr(rooms, "run", lambda console, args, start, inv=None, came_from="command", made=None:
                        opened.append((start, came_from, len(inv.agents))) or 0)
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim"], capsys)
    assert rc == 0 and list(asked[0]) == ["m", "s", "q"] and opened == [("map", "scan", 6)]
    for flags in (["--json"], ["--check"]):  # scripts and CI are never asked
        asked.clear()
        run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim", *flags], capsys)
        assert not asked
    args = commands.setup_args(type("A", (), {"root": str(HOST_OPS), "workspace": str(ws), "no_anim": True})())
    assert args.root == str(HOST_OPS) and args.workspace == str(ws) and args.no_anim and not args.yes


def test_ask_next_takes_one_key():
    class Keys:
        def __init__(self, seq):
            self.seq = list(seq)

        def __call__(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, timeout):
            return self.seq.pop(0) if self.seq else "q"

    c = _console()
    assert rooms.ask_next(c, {"m": "reach map", "q": "quit"}, Keys([None, "m"])) == "m"
    assert rooms.ask_next(c, {"m": "reach map", "q": "quit"}, Keys(["enter"])) is None
    assert rooms.ask_next(c, {"m": "reach map", "q": "quit"}, Keys(["q"])) is None
