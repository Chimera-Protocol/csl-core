"""One flow: the end of a scan, the reach map and the live panel lead into each other."""

from __future__ import annotations

import io
import time

import pytest
from rich.console import Console

from chimera_core.venom import rooms
from chimera_core.venom.render import mapview as M
from chimera_core.venom.render.theme import THEME

from .conftest import HOST_OPS, render, run_cli, scan_fixture, wired_setup


@pytest.fixture
def ws(tmp_path, capsys):
    return wired_setup(tmp_path, capsys)


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
    assert not w.panel.map_on and w.settle0 is None  # the panel opens on the live decisions, always
    assert "reach map · live" not in _play(w, 0.1)
    w.handle("f")  # the full map, straight from the stream
    assert w.exit_to == "map"
    w.exit_to = None
    w.handle("g")  # the map beside the decisions
    assert w.panel.map_on and "reach map · live" in _play(w, 0.1)
    w.handle("g")  # and back to the stream
    assert not w.panel.map_on
    w.handle("g")
    w.handle("f")  # from the panel's map, it grows into the full map
    assert w.grow0 is not None and w.exit_to is None
    _play(w, 1.0)
    assert w.exit_to == "map" and w.panel.pane_zoom == 1.0
    w.exit_to = None
    w.enter("map")  # back from the full map: as it was left, its map settling into place
    assert w.panel.map_on and w.settle0 is not None
    _play(w, 1.0)
    assert w.settle0 is None and w.panel.pane_zoom == 1.0
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


def _map_on(ws, monkeypatch):
    from chimera_core.venom.watch import _inventory
    from chimera_core.venom.workspace import Workspace

    monkeypatch.chdir(ws)
    w = Workspace(ws)
    v = M.MapView(_inventory(w), 130, 40, ws=w)
    v.spread0 -= 99
    return v, w


def _select(v, name):
    v.sel = v.order.index(next(n for n in v.order if v.g.nodes[n].label == name))


def test_freeze_on_the_map_stops_the_real_agent(ws, monkeypatch):
    from chimera_core.venom.controls import Controls
    from chimera_core.venom.observe import venom_guard

    v, w = _map_on(ws, monkeypatch)
    _select(v, "membership-bot")
    v.handle("x")
    assert v.pending and v.pending[2].startswith("Freeze membership-bot") and "no policy" not in v.pending[2]
    text = render(v.frame(time.monotonic(), 130), width=130)
    assert "[y/n]" in text
    assert v.handle("q") and v.pending is None  # q answers the question (cancel); it does not quit
    v.handle("x")
    v.handle("y")
    assert Controls(w).get("membership-bot").disabled
    g = venom_guard("membership-bot", policy="policies/membership-bot.csl", mapping="policies/membership_bot_mapping.py")
    r = g.verify("transfer_funds", {"amount": 5, "to_wallet": "w"})
    assert not r.allowed and r.violated_rule_ids == ["__agent_disabled__"]  # the map's x is enforcement
    node = next(n for n in v.order if v.g.nodes[n].label == "membership-bot")
    assert v.topo.marks[node] == "frozen"
    text = render(v.frame(time.monotonic(), 130), width=130)
    assert "FROZEN" in text and "frozen: every call is blocked" in text.split("SELECTED")[1]
    assert v.g.top is None  # setup wired the agents: the chain through claude-code:ops is closed
    v.handle("x")  # unfreezing needs no question
    assert not Controls(w).get("membership-bot").disabled and node not in v.topo.marks
    assert g.verify("transfer_funds", {"amount": 5, "to_wallet": "w"}).allowed


def test_mode_on_the_map(ws, monkeypatch):
    from chimera_core.venom.controls import Controls

    v, w = _map_on(ws, monkeypatch)
    _select(v, "ingest-worker")
    v.handle("m")
    assert "BLOCK mode" in v.pending[2]
    v.handle("n")
    assert Controls(w).get("ingest-worker").mode == "log" and v.message[0] == "cancelled"
    v.handle("m")
    v.handle("y")
    node = next(n for n in v.order if v.g.nodes[n].label == "ingest-worker")
    assert Controls(w).get("ingest-worker").mode == "block" and v.topo.marks[node] == "block"
    assert "block mode" in render(v.frame(time.monotonic(), 130), width=130).split("SELECTED")[1]


def test_freezing_an_unwired_agent_offers_to_guard_it_first(tmp_path, capsys, monkeypatch):
    """Nothing is in publisher's call path, so freezing it would stop nothing: the map says so
    and offers to put it under a guard instead of pretending."""
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim"], capsys)
    v, w = _map_on(ws, monkeypatch)
    _select(v, "publisher")
    text = render(v.frame(time.monotonic(), 130), width=130)
    assert "not wired: nothing stops it yet" in text.split("SELECTED")[1]
    v.handle("x")
    assert v.pending is None and "is not wired: cslcore wire --agent publisher" in v.message[0]  # no args: a hint
    v.args = _args(ws)
    v.handle("x")
    assert v.pending[0] == "guard" and "freezing it would stop nothing" in v.pending[2]
    v.handle("y")
    assert v.external is not None  # runs outside the screen: policy, check, wiring, each asked
    node = next(n for n in v.order if v.g.nodes[n].label == "publisher")
    assert node not in v.topo.marks  # no ice on an agent nothing can stop
    v.sel = v.order.index(next(n for n in v.order if v.g.nodes[n].kind == "input"))
    v.external = None
    v.handle("x")
    assert v.pending is None and "select an agent" in v.message[0]


def test_the_panels_map_shows_the_same_marks(ws, monkeypatch):
    from chimera_core.venom import watch as W
    from chimera_core.venom.controls import Controls
    from chimera_core.venom.workspace import Workspace

    monkeypatch.chdir(ws)
    w = Workspace(ws)
    Controls(w).set_disabled("publisher", True)
    p = W.ControlPanel(Controls(w), W.WatchModel(), W._inventory(w), w)
    W.map_pane(p, p.model, p.inv, 60, 16)
    node = next(a.id for a in p.inv.agents if a.display_name == "publisher")
    assert p.topo.marks.get(node) == "frozen"
