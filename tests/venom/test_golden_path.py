"""The golden path, as a person sees it: a fresh folder with a LangChain agent and a Claude Code
project, `cslcore setup` in a real terminal (a pty), Enter for every question. It must finish
without a crash, ask at most five questions, show no internal terms, end with every agent protected
and a check table; then the wired tool runs a small amount and answers a large one with a readable
result instead of running it."""

from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a pty")

CSLCORE = str(Path(sys.executable).parent / "cslcore")
JARGON = ("fail-open", "Z3", "mapping", "V0", "derived", "case variants", "bypass")
PROMPT = re.compile(r"(\]: ?|\): ?|\? ?|: ?|quit ?)$")
AGENT = '''from langchain_core.tools import tool


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money to a wallet."""
    return f"sent {amount}"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return "ok"
'''


def _clean(raw: bytes) -> str:
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return re.sub(r"\x1b\][^\x07]*\x07", "", text).replace("\r", "")


def run_with_enter(argv, cwd, env, idle=1.0, timeout=180):
    """Run argv in a pty; whenever it waits at a question, press Enter. (screen, questions, exit code, seconds)."""
    import pty

    pid, fd = pty.fork()
    if pid == 0:  # the child: the command, in its own terminal
        os.chdir(cwd)
        os.execvpe(argv[0], argv, env)
    out, answered, start, last = b"", 0, time.time(), time.time()
    while time.time() - start < timeout:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break  # the terminal closed: the command ended
            if not chunk:
                break
            out, last = out + chunk, time.time()
            continue
        if time.time() - last > idle:
            tail = _clean(out[-500:]).rstrip()
            if tail and PROMPT.search(tail.splitlines()[-1]):
                os.write(fd, b"\r")
                answered, last = answered + 1, time.time()
    else:
        os.kill(pid, 9)
    _, status = os.waitpid(pid, 0)
    return _clean(out), answered, os.waitstatus_to_exitcode(status), time.time() - start


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "project"
    (root / ".claude").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "agent.py").write_text(AGENT)
    (root / "requirements.txt").write_text("langchain-core\n")
    (root / ".claude/settings.json").write_text('{"permissions": {"allow": ["Bash(git status)"]}}')
    (root / "CLAUDE.md").write_text("# project\n")
    home = tmp_path / "home"
    home.mkdir()
    return root, home


def test_golden_path_with_enter_only(folder):
    pytest.importorskip("langchain_core")
    root, home = folder
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "TERM": "xterm-256color",
           "COLUMNS": "110", "LINES": "40", "CSL_NO_ANIM": "1"}
    screen, questions, code, seconds = run_with_enter([CSLCORE, "setup", "--root", "."], root, env)
    assert code == 0, screen[-3000:]
    assert "Traceback" not in screen
    assert questions <= 5, (questions, screen[-3000:])
    assert seconds < 180
    for word in JARGON:
        assert word not in screen, word
    assert "setup complete · 2 of 2 protected" in screen
    assert "CHECK" in screen and "as its limits say" in screen
    assert "Undo all of it: cslcore wire --undo" in screen

    # the wired tool, as the agent calls it
    probe = (
        "import sys; sys.path.insert(0, '.'); import agent\n"
        "print(agent.transfer_funds.invoke({'amount': 50, 'to_wallet': 'w'}))\n"
        "print(agent.transfer_funds.invoke({'amount': 5000, 'to_wallet': 'w'}))\n"
    )
    res = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True, text=True,
                         env={**env, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}, timeout=60)
    small, large = res.stdout.strip().splitlines()[-2:]
    assert small == "sent 50", res.stderr
    assert large.startswith("CSL-Core: transfer_funds was not run:") and "never above" in large
