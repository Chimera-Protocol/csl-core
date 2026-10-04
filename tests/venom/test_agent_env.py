"""Before a Python agent is wired, the interpreter it runs with (its project's virtual environment,
when it has one) must import chimera_core; otherwise the wired agent would stop at its first import.
Setup and wire say how to install it and ask; a scripted run does not wire that agent."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path


from .conftest import run_cli

SRC = Path(__file__).resolve().parents[2]
AGENT = '''"""Payouts."""


def tool(fn):
    return fn


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money to a wallet."""
    return f"sent {amount}"
'''


def _repo(tmp_path, name, with_csl):
    root = tmp_path / name
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "app/agent.py").write_text(AGENT)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")], check=True)
    if with_csl:  # csl-core visible in that environment, as `pip install csl-core` would make it
        purelib = sysconfig.get_path("purelib", vars={"base": str(root / ".venv"), "platbase": str(root / ".venv")})
        Path(purelib).mkdir(parents=True, exist_ok=True)
        deps = sysconfig.get_path("purelib")  # its dependencies (z3, rich), as pip would install them
        (Path(purelib) / "csl_core_src.pth").write_text(f"{SRC}\n{deps}\n")
    ws = tmp_path / f"{name}-ws"
    ws.mkdir()
    return root, ws


def test_a_scripted_setup_does_not_wire_an_agent_whose_environment_lacks_csl_core(tmp_path, capsys):
    root, ws = _repo(tmp_path, "bare", with_csl=False)
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--no-anim"], capsys)
    flat = " ".join(out.split())
    assert rc == 0 and "cannot import chimera_core" in flat
    assert ".venv/bin/python -m pip install csl-core" in out.replace("\n", "") and "not wired" in flat
    assert (root / "app/agent.py").read_text() == AGENT


def test_an_environment_with_csl_core_is_wired_without_a_question(tmp_path, capsys):
    root, ws = _repo(tmp_path, "ready", with_csl=True)
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
                          "--wire", "--no-anim"], capsys)
    assert rc == 0 and "cannot import chimera_core" not in out
    assert "_csl_guard.tool" in (root / "app/agent.py").read_text()


def test_wire_asks_to_install_then_to_wire_anyway(tmp_path, capsys, monkeypatch):
    from rich.prompt import Confirm

    from chimera_core.venom import probe

    root, ws = _repo(tmp_path, "asks", with_csl=False)
    run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--mode", "block",
             "--no-anim"], capsys)
    asked = []
    answers = [False, True, True]  # install now? no · wire anyway? yes · wire it? yes
    monkeypatch.setattr(Confirm, "ask", classmethod(lambda cls, q, **k: asked.append(q) or answers.pop(0)))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    installed = []
    monkeypatch.setattr(probe, "run_install", lambda cmd, timeout=600.0: installed.append(cmd) or (True, ""))
    rc, out, _ = run_cli(["wire", "--root", str(root), "--workspace", str(ws)], capsys)
    assert rc == 0 and not installed
    assert "Install csl-core there now" in asked[0] and "anyway" in asked[1]
    assert "_csl_guard.tool" in (root / "app/agent.py").read_text()
