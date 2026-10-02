"""Discovery animation and the home screen."""

from __future__ import annotations

import argparse
import io
import time

from rich.console import Console

from chimera_core.venom.render import reveal as R
from chimera_core.venom.render.theme import THEME

from .conftest import HOST_OPS, render, run_cli, scan_fixture


def _console(terminal=True, color=True):
    return Console(theme=THEME, file=io.StringIO(), width=100, force_terminal=terminal,
                   color_system="truecolor" if color else None, no_color=not color)


def test_animation_only_where_it_belongs(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("CSL_NO_ANIM", raising=False)
    plain = argparse.Namespace()
    assert R.enabled(_console(), plain)
    assert not R.enabled(_console(terminal=False), plain)
    assert not R.enabled(_console(color=False), plain)
    for flag in ("json", "check", "compact", "no_anim"):
        assert not R.enabled(_console(), argparse.Namespace(**{flag: True}))
    monkeypatch.setenv("CI", "true")
    assert not R.enabled(_console(), plain)


def test_reveal_paces_layers_and_finishes(tmp_path):
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    r = R.Reveal(_console(), "0.6.0")
    r.t0 = 0.0
    for i, layer in enumerate(["triggers", "runtime", "code", "config", "policies", "history"]):
        r.events[layer] = ("done", f"{i + 1} things", 0.01 * i)
    r.done, r.inv = True, inv
    r.frame(0.05)
    assert len(r.shown_at) == 0  # nothing jumps on screen at once
    t = 0.0
    while t < 1.0:
        r.frame(t)
        t += 0.04
    assert 2 <= len(r.shown_at) < 6  # layers appear one after another
    assert not r.finished(t)
    while not r.finished(t):
        r.frame(t)
        t += 0.04
        assert t < 8
    text = render(r.frame(t), width=100)
    assert "6 agents" in text and "membership-bot" in text and "discovered" in text


def test_any_key_skips(tmp_path):
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    r = R.Reveal(_console(), "0.6.0")
    r.events["code"] = ("done", "5 files", 0.0)
    r.done, r.inv, r.skipped = True, inv, True
    r.frame(0.01)
    assert r.finished(0.01)


def test_run_returns_the_real_scan_result(tmp_path):
    from chimera_core.venom.probe import probe_for
    from chimera_core.venom.scanner import Scanner
    from chimera_core.venom.workspace import Workspace

    probe, roots = probe_for(str(HOST_OPS))
    t0 = time.monotonic()
    res = R.run(_console(), "0.6.0", lambda ev: Scanner(probe, roots, Workspace(tmp_path), on_event=ev).run())
    assert len(res.inventory.agents) == 6
    assert time.monotonic() - t0 < 10


def test_home_screen(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate"], capsys)
    from chimera_core.venom.setup import Flow
    f = Flow(argparse.Namespace(workspace=str(ws), root=str(HOST_OPS), yes=False, no_color=True, plan_only=False))
    text = render(f.home_panel(), width=80)
    for part in ("6 agents discovered", "5 with a policy", "LOG by default", "none yet", "5 high findings",
                 "scan again", "live management panel"):
        assert part in text, part


def test_discovery_web_is_quick_steady_and_only_on_wide_terminals(tmp_path):
    from chimera_core.venom.render.web import Web

    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    a = Web(["code", "config"], seed="x").render(1.0, {"code": 0.5, "config": 1.0}, {"config": 0.2}, [])
    b = Web(["code", "config"], seed="x").render(1.0, {"code": 0.5, "config": 1.0}, {"config": 0.2}, [])
    assert a.plain == b.plain and len(a.plain.splitlines()) == 13  # the same host grows the same web
    r = R.Reveal(_console(), "0.6.0")
    r.t0 = 0.0
    for layer, _ in R.LAYERS:
        r.events[layer] = ("done", "1 thing", 0.0)
    r.done, r.inv = True, inv
    t0 = time.perf_counter()
    for i in range(60):
        r.frame(i * 0.05)
    assert (time.perf_counter() - t0) / 60 < 0.02  # well inside one frame at 24 fps
    narrow = R.Reveal(Console(theme=THEME, file=io.StringIO(), width=80, force_terminal=True), "0.6.0")
    narrow.events["code"] = ("done", "1 thing", 0.0)
    assert "◉" not in render(narrow.frame(0.5), width=80)
    assert "◉" in render(r.frame(0.5), width=100)
