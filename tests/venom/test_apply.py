"""csl-limits.ini: the limits a repository keeps for its agents, reviewed in pull requests, made into
active policies by cslcore apply, and checked in CI by cslcore apply --check."""

from __future__ import annotations

import importlib.util

import pytest

from .conftest import run_cli
from .test_multi_agent_repo import FILES


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / ".git").mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(root / "desk")  # the file is found from a subfolder, up to the repository root
    return root, ws


def _apply(ws, capsys, *extra):
    return run_cli(["apply", "--workspace", str(ws), "--no-anim", *extra], capsys)


def _load(path):
    spec = importlib.util.spec_from_file_location(f"m_{id(path)}_{path.stat().st_mtime_ns}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_init_apply_and_check_in_ci(repo, capsys):
    root, ws = repo
    rc, out, _ = _apply(ws, capsys)
    assert rc == 2 and "--init" in out
    rc, out, _ = _apply(ws, capsys, "--init")
    assert rc == 0
    ini = (root / "csl-limits.ini").read_text()
    for section in ("[billing]", "[helpdesk]", "[agents-payments]", "[agents-reports]", "[repo-ops]"):
        assert section in ini
    assert "; charge_card = 100..1k   ; standard" in ini and "mode = block" in ini
    ini = ini.replace("; charge_card = 100..1k   ; standard", "charge_card = 50..200")
    ini = ini.replace("[helpdesk]", "[helpdesk]\ndelete_ticket = approval")
    (root / "csl-limits.ini").write_text(ini)
    rc, out, _ = _apply(ws, capsys, "--check")  # CI: nothing active there, the file itself is checked
    flat = " ".join(out.split())
    assert rc == 0 and "not active in this workspace" in flat and "as its limits say" in flat
    assert not (ws / "policies/billing.csl").exists()
    rc, out, _ = _apply(ws, capsys, "--yes", "--wire")
    assert rc == 0, out
    assert "amount <= 200" in (ws / "policies/billing.csl").read_text()
    flat = " ".join(out.split())
    assert "as its limits say" in flat and "not as its limits say" not in flat
    assert "billing mode block: stops what its limits do not allow" in flat
    mod = _load(root / "desk/agents.py")
    assert mod.charge_card(customer_id="c", amount=50) == "charged"
    with pytest.raises(PermissionError):
        mod.charge_card(customer_id="c", amount=150)  # over 50: needs an approval
    rc, out, _ = _apply(ws, capsys, "--check")
    assert rc == 0 and "not what the file says" not in out
    # a pull request changes the file: CI says so, and nothing changes until it is applied
    (root / "csl-limits.ini").write_text(ini.replace("charge_card = 50..200", "charge_card = 50..300"))
    before = (ws / "policies/billing.csl").read_text()
    rc, out, _ = _apply(ws, capsys, "--check")
    assert rc == 1 and "billing" in out and (ws / "policies/billing.csl").read_text() == before


def test_wrong_names_are_refused(repo, capsys):
    root, ws = repo
    (root / "csl-limits.ini").write_text("[nobody]\nx = 1..2\n")
    rc, out, _ = _apply(ws, capsys)
    assert rc == 2 and "no agent by that name" in out and "billing" in out
    (root / "csl-limits.ini").write_text("[billing]\nrefund = 1..2\n")
    rc, out, _ = _apply(ws, capsys)
    assert rc == 2 and "no tool 'refund'" in out and "charge_card" in out
    (root / "csl-limits.ini").write_text("[billing]\ncharge_card = 300..100\n")
    rc, out, _ = _apply(ws, capsys)
    assert rc == 2 and "above the maximum" in out


def test_limits_on_an_agent_the_file_names_say_where_they_belong(repo, capsys):
    root, ws = repo
    (root / "csl-limits.ini").write_text("[billing]\ncharge_card = 50..200\n")
    assert _apply(ws, capsys, "--yes")[0] == 0
    rc, out, _ = run_cli(["limits", "--agent", "billing", "--set", "charge_card=10..20", "--yes", "--workspace", str(ws),
                          "--root", str(root)], capsys)
    assert rc == 0 and "lasts until the next cslcore apply" in " ".join(out.split())
