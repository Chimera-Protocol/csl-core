from __future__ import annotations

import io
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
HOST_OPS = FIXTURES / "host_ops"
HOST_EMPTY = FIXTURES / "host_empty"
SENTINEL = "CSL_VENOM_SENTINEL_7f3a9c"


def scan_fixture(fixture: Path, workspace: Path, window_days: int = 7):
    from chimera_core.venom.probe import probe_for
    from chimera_core.venom.scanner import Scanner
    from chimera_core.venom.workspace import Workspace

    probe, roots = probe_for(str(fixture))
    result = Scanner(probe, roots, Workspace(workspace), window_days=window_days, tool_version="0.6.0").run()
    return result


@pytest.fixture
def ops(tmp_path):
    return scan_fixture(HOST_OPS, tmp_path)


@pytest.fixture
def ops_inv(ops):
    return ops.inventory


def render(renderable, width: int = 100, color: bool = False, height: int | None = None) -> str:
    from chimera_core.venom.render.theme import make_console

    from rich.console import Console
    from chimera_core.venom.render.theme import THEME

    buf = io.StringIO()
    if color:
        console = Console(theme=THEME, width=width, file=buf, force_terminal=True, color_system="truecolor")
    else:
        console = make_console(no_color=True, width=width, file=buf)
    if height:
        console.print(renderable, height=height)
    else:
        console.print(renderable)
    return buf.getvalue()


def run_cli(argv, capsys):
    from chimera_core.cli import main

    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err
