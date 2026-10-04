"""The guard line cslcore wire writes holds in a clone of the repository: on another machine, in CI,
from any working folder. Its workspace path is relative to the agent's file, CSL_WORKSPACE
overrides it, and the nearest .csl above the file is found when neither holds. Lines from 0.6.8
with an absolute path keep working; with no workspace anywhere the guard refuses (fail closed)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import run_cli

SRC = Path(__file__).resolve().parents[2]  # the csl-core source, as an installed package would be

AGENT = '''"""Payouts."""


def tool(fn):
    return fn


@tool
def transfer_funds(amount: int, to_wallet: str) -> str:
    """Send money to a wallet."""
    return f"sent {amount}"


@tool
def write_file(path: str, text: str) -> str:
    """Write a file."""
    return "written"
'''

RUN = r"""
import sys
sys.path.insert(0, sys.argv[1])
import agent
out = [agent.transfer_funds(amount=50, to_wallet="w")]
for call in (lambda: agent.transfer_funds(amount=900, to_wallet="w"),
             lambda: agent.write_file(path=sys.argv[2], text="x")):
    try:
        r = call()
        out.append("BLOCKED" if "was not run" in str(r) else r)  # a framework tool returns Blocked
    except PermissionError:
        out.append("BLOCKED")
print("|".join(out))
"""


@pytest.fixture
def repo(tmp_path, capsys):
    root = tmp_path / "team" / "repo"
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "app/agent.py").write_text(AGENT)
    (root / "requirements.txt").write_text("csl-core\n")
    rc, out, _ = run_cli(["setup", "--root", str(root), "--workspace", str(root), "--yes", "--activate", "--mode",
                          "block", "--wire", "--limit", "repo.transfer_funds=100..500", "--no-anim"], capsys)
    assert rc == 0, out
    return root


def _clean_run(agent_dir: Path, inside: Path, cwd: Path, home: Path, **env) -> str:
    """The agent in a fresh process: only PATH, its own HOME, another working folder, csl-core
    importable as an installed package would be, nothing else from this session."""
    base = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONPATH": str(SRC), "NO_COLOR": "1"}
    res = subprocess.run([sys.executable, "-c", RUN, str(agent_dir), str(inside)], cwd=cwd,
                         env={**base, **env}, capture_output=True, text=True, timeout=120)
    return res.stdout.strip() or res.stderr.strip().splitlines()[-1]


def test_the_line_has_no_absolute_path(repo):
    text = (repo / "app/agent.py").read_text()
    line = next(ln for ln in text.splitlines() if "venom_guard(" in ln and "import" not in ln)
    assert 'workspace=".."' in line and "near=__file__" in line and str(repo) not in line
    mapping = (repo / "policies/repo_mapping.py").read_text()
    assert str(repo) not in mapping.split("SCOPE_ROOTS", 1)[1].splitlines()[0]


def test_a_clone_elsewhere_works_from_any_folder(repo, tmp_path):
    clone = tmp_path / "elsewhere" / "checkout"
    shutil.copytree(repo, clone)
    shutil.rmtree(tmp_path / "team")  # the original machine's paths are gone
    cwd, home = tmp_path / "somewhere", tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    out = _clean_run(clone / "app", clone / "app/notes.txt", cwd, home)
    assert out == "sent 50|BLOCKED|written", out
    out = _clean_run(clone / "app", "/etc/passwd-copy", cwd, home)  # outside its own folder
    assert out.endswith("|BLOCKED"), out


def test_csl_workspace_overrides_and_old_absolute_lines_still_work(repo, tmp_path):
    clone = tmp_path / "ci" / "repo"
    shutil.copytree(repo, clone)
    cwd, home = tmp_path / "run", tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    agent = clone / "app/agent.py"
    # a 0.6.8 line: an absolute path that holds on this machine
    agent.write_text(re.sub(r'workspace="\.\.", near=__file__', f'workspace="{clone}"', agent.read_text()))
    assert _clean_run(clone / "app", clone / "app/n.txt", cwd, home) == "sent 50|BLOCKED|written"
    # an absolute path from another machine: the workspace above the file is found
    agent.write_text(agent.read_text().replace(str(clone), "/nowhere/old-machine/repo"))
    assert _clean_run(clone / "app", clone / "app/n.txt", cwd, home) == "sent 50|BLOCKED|written"
    # CSL_WORKSPACE points somewhere else: that workspace is used
    moved = tmp_path / "ws-elsewhere"
    shutil.copytree(clone / ".csl", moved / ".csl")
    shutil.copytree(clone / "policies", moved / "policies")
    shutil.rmtree(clone / ".csl")
    assert _clean_run(clone / "app", clone / "app/n.txt", cwd, home, CSL_WORKSPACE=str(moved)).startswith("sent 50|BLOCKED")


def test_no_workspace_anywhere_fails_closed(repo, tmp_path):
    lone = tmp_path / "lone" / "app"
    lone.mkdir(parents=True)
    shutil.copy(repo / "app/agent.py", lone / "agent.py")
    cwd, home = tmp_path / "run", tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    out = _clean_run(lone, lone / "n.txt", cwd, home)
    assert "sent 50" not in out and ("LookupError" in out or "BLOCKED" in out), out
