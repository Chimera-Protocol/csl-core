"""`cslcore venom map`: the interactive reach map (keys, dive in, sphere) and its one-frame mode."""

from __future__ import annotations

import io
import time

import pytest
from rich.console import Console

from chimera_core.venom.render import mapview as M
from chimera_core.venom.render.theme import THEME
from chimera_core.venom.render.topo import Sphere, Zoom

from .conftest import HOST_OPS, render, run_cli, scan_fixture


@pytest.fixture(scope="module")
def inv(tmp_path_factory):
    return scan_fixture(HOST_OPS, tmp_path_factory.mktemp("map")).inventory


def test_keys_select_dive_and_back(inv):
    v = M.MapView(inv, 130, 40)
    first = v.selected()
    v.handle("down")
    assert v.selected() != first
    v.sel = v.order.index(next(n for n in v.order if v.g.nodes[n].label == "claude-code:ops"))
    assert v.handle("enter") and v.zoom[1] == "in"
    v.zoom = (time.monotonic() - 5, "in")
    v.frame(time.monotonic(), 130)
    assert v.mode == "dive" and v.dive is not None
    assert {s.label for s in v.dive.tools} >= {"Bash", "Edit"} and len(v.dive.tools) <= 10
    assert any(sp.label == "ingest-worker" for sp in v.dive.targets.values())
    v.handle("esc")
    v.zoom = (time.monotonic() - 5, "out")
    v.frame(time.monotonic(), 130)
    assert v.mode == "map"
    v.handle("3")
    assert v.sphere
    assert v.handle("q") is False


def test_frames_render_in_every_mode(inv):
    v = M.MapView(inv, 130, 40)
    v.spread0 -= 30
    text = render(v.frame(time.monotonic(), 130), width=130)
    assert "REACH CHAIN" in text and "claude-code:ops" in text and "SELECTED" in text
    v.sphere = True
    assert "3D" in render(v.frame(time.monotonic(), 130), width=130)


def test_cameras():
    z = Zoom(10, 10, 2, 100, 60)
    assert z(10, 10) == (50, 30, 1.0) and z(20, 10)[0] == 70
    flat = Sphere(0.0, 100, 60, tilt=0.0)
    x, y, d = flat(50, 30)  # untilted, the centre of the map faces the viewer
    assert abs(x - 50) < 1e-6 and abs(y - 30) < 1e-6 and d > 0.99
    s = Sphere(0.0, 100, 60)
    assert s(50, 30)[1] > 30  # the default tilt shows the globe slightly from above
    assert abs(s.R - 60 * 0.46) < 1e-6


def test_once_prints_one_frame(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--no-anim"], capsys)
    rc, out, _ = run_cli(["venom", "map", "--once", "--workspace", str(ws)], capsys)
    assert rc == 0 and "reach map" in out and "REACH CHAIN" in out
