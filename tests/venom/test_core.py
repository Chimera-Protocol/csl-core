"""B1 skeleton, model and CLI wiring; global checks G2-G8 on the discovery side."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import HOST_EMPTY, HOST_OPS, SENTINEL, run_cli, scan_fixture

VENOM = Path(__file__).resolve().parents[2] / "chimera_core" / "venom"


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [["setup", "--help"], ["venom", "--help"]])
def test_help_lists_subcommands(cmd, capsys):
    from chimera_core.cli import main

    with pytest.raises(SystemExit) as e:
        main(cmd)
    assert e.value.code == 0
    out = capsys.readouterr().out
    if cmd[0] == "venom":
        assert "report" in out and "--json" in out and "--check" in out
    assert "--root" in out and "--plan-only" in out


def test_top_level_commands(capsys):
    from chimera_core.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("setup", "venom", "policy", "map", "exempt", "mode", "watch", "verify", "simulate"):
        assert cmd in out


def test_scan_empty_host_json(tmp_path, capsys):
    rc, out, _ = run_cli(["venom", "--root", str(HOST_EMPTY), "--json", "--no-save", "--workspace", str(tmp_path)], capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == 1
    assert data["agents"] == [] and data["findings"] == []


def test_model_round_trip(ops_inv):
    from chimera_core.venom import model as M

    assert M.Inventory.from_dict(ops_inv.to_dict()) == ops_inv
    samples = {
        M.Evidence: M.Evidence("code", "/a.py", 3, "x"), M.ToolParam: M.ToolParam("a", "int", False, ["X"], 0, 9),
        M.Credential: M.Credential("K", "/f", "llm_key"), M.Trigger: M.Trigger("time", "daily", "cron"),
        M.RunStats: M.RunStats(3, "7d", "2026-10-01", [1, 2], "journal"), M.Exemption: M.Exemption("a", "tool", "t", "r", "me", "2026-12-31", "approved"),
        M.PromptInfo: M.PromptInfo(True, 10, "abc"), M.Coverage: M.Coverage(3, 1, 1, 1, 0),
        M.DriftItem: M.DriftItem("coercion", "P", "v"), M.Finding: M.Finding("V01", "r", "high", "s"),
    }
    for cls in M.ALL_MODELS:
        inst = samples.get(cls)
        if inst is None:
            continue
        assert cls.from_dict(inst.to_dict()) == inst, cls.__name__
    for a in ops_inv.agents:
        assert M.Agent.from_dict(a.to_dict()) == a
    for p in ops_inv.policies:
        assert M.PolicyRef.from_dict(p.to_dict()) == p


def test_os_access_only_in_probe_and_workspace():
    """No module outside probe.py / workspace.py calls open(), os.walk, subprocess or os.environ."""
    import ast

    offenders = []
    for path in VENOM.rglob("*.py"):
        if path.name in ("probe.py", "workspace.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            bad = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                bad = "open()"
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in ("walk", "environ", "system", "popen"):
                bad = f"os.{node.attr}"
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                if any(n.split(".")[0] == "subprocess" for n in names):
                    bad = "import subprocess"
            if bad:
                offenders.append(f"{path.relative_to(VENOM)}:{node.lineno}: {bad}")
    assert not offenders, "\n".join(offenders)


def test_import_isolation():
    code = "import sys, chimera_core, chimera_core.cli; assert not [m for m in sys.modules if m.startswith('chimera_core.venom')]"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------------------------------------------------------------------
# global checks
# ---------------------------------------------------------------------------

def test_g2_no_credential_value_anywhere(tmp_path, capsys):
    rc, out, err = run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(tmp_path)], capsys)
    assert SENTINEL not in out + err
    rc, out, _ = run_cli(["venom", "--root", str(HOST_OPS), "--json", "--workspace", str(tmp_path)], capsys)
    assert SENTINEL not in out
    for f in (tmp_path / ".csl").rglob("*"):
        if f.is_file():
            assert SENTINEL not in f.read_text(encoding="utf-8"), f
    for agent in json.loads(out)["agents"]:
        rc, text, _ = run_cli(["venom", "report", "--agent", agent["id"], "--workspace", str(tmp_path)], capsys)
        assert SENTINEL not in text


def test_g3_discovered_code_never_imported(tmp_path):
    marker = HOST_OPS / "fs" / "srv" / "ingest-worker" / "IMPORTED.marker"
    scan_fixture(HOST_OPS, tmp_path)
    assert not marker.exists()
    assert "sideeffect" not in sys.modules


def test_g4_no_network(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access during discovery")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    inv = scan_fixture(HOST_OPS, tmp_path).inventory
    assert inv.agents


def test_g5_deterministic(tmp_path):
    from chimera_core.venom.render.report import to_json

    a = to_json(scan_fixture(HOST_OPS, tmp_path).inventory)
    b = to_json(scan_fixture(HOST_OPS, tmp_path).inventory)
    for d in (a, b):
        d["host"].pop("scanned_at")
        d["host"].pop("duration_ms")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_g7_no_em_dash_in_venom():
    for path in list(VENOM.rglob("*.py")) + [VENOM.parent / "cli_venom.py"]:
        assert "—" not in path.read_text(encoding="utf-8"), path
    mapping = VENOM.parent / "mapping.py"
    if mapping.exists():
        assert "—" not in mapping.read_text(encoding="utf-8")


def _tree(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_g8_discovery_writes_nothing(tmp_path, capsys):
    before = _tree(HOST_OPS)
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(tmp_path)], capsys)
    assert _tree(HOST_OPS) == before
    # --plan-only writes nothing, not even the report
    ws = tmp_path / "plan"
    ws.mkdir()
    run_cli(["venom", "--root", str(HOST_OPS), "--workspace", str(ws), "--plan-only"], capsys)
    assert list(ws.iterdir()) == []


def test_probe_refuses_unlisted_commands():
    from chimera_core.venom.probe import LocalHostProbe

    with pytest.raises(PermissionError):
        LocalHostProbe().run(["rm", "-rf", "/tmp/x"])
