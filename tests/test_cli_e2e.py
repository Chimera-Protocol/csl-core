from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd: Path):
    env = os.environ.copy()
    # Ensure repo root is importable when running `python -m chimera_core.cli`
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )


def test_cli_verify_e2e(repo_root: Path, agent_tool_guard_policy_path: Path):
    cmd = [sys.executable, "-m", "chimera_core.cli", "verify", str(agent_tool_guard_policy_path)]
    p = _run(cmd, cwd=repo_root)

    assert p.returncode == 0, f"verify should succeed. rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "Verification passed" in p.stdout


def test_cli_simulate_e2e_allow(repo_root: Path, agent_tool_guard_policy_path: Path, tmp_path: Path):
    # ALLOW input should return 0
    allow_ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    in_file = tmp_path / "allow.json"
    in_file.write_text(json.dumps(allow_ctx), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "chimera_core.cli", "simulate",
        str(agent_tool_guard_policy_path),
        "--input-file", str(in_file),
        "--no-raise",
    ]
    p = _run(cmd, cwd=repo_root)

    assert p.returncode == 0, f"simulate allow should be rc=0. rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "Simulation Summary" in p.stdout


def test_cli_simulate_e2e_block(repo_root: Path, agent_tool_guard_policy_path: Path, tmp_path: Path):
    # BLOCK input should return 10 (your CLI rule: blocked -> 10 unless dry-run)
    block_ctx = {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    in_file = tmp_path / "block.json"
    in_file.write_text(json.dumps(block_ctx), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "chimera_core.cli", "simulate",
        str(agent_tool_guard_policy_path),
        "--input-file", str(in_file),
        "--no-raise",
    ]
    p = _run(cmd, cwd=repo_root)

    assert p.returncode == 10, f"simulate block should be rc=10. rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "Simulation Summary" in p.stdout
