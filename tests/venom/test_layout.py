"""Everything CSL-Core writes lives under .csl/ and stays out of the repository (its own .gitignore);
the one file a team commits is csl-limits.ini. A workspace made before 0.6.9, with policies/ in its
root, is read as it is."""

from __future__ import annotations

import shutil

from chimera_core.venom.bindings import Bindings
from chimera_core.venom.workspace import Workspace

from .conftest import HOST_OPS, run_cli


def test_a_new_setup_writes_only_under_csl(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    rc, _out, _ = run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate", "--no-anim"],
                          capsys)
    assert rc == 0
    assert sorted(p.name for p in ws.iterdir()) == [".csl"]
    assert (ws / ".csl/.gitignore").read_text().strip().endswith("*")
    assert list((ws / ".csl/policies").glob("*.csl")) and list((ws / ".csl/policies").glob("*_mapping.py"))


def test_a_workspace_from_0_6_8_keeps_policies_in_its_root(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(HOST_OPS), "--workspace", str(ws), "--yes", "--activate", "--no-anim"], capsys)
    old = tmp_path / "old"  # the same workspace laid out as 0.6.8 wrote it
    shutil.copytree(ws, old)
    shutil.move(str(old / ".csl/policies"), str(old / "policies"))
    state = (old / ".csl/venom/state.json").read_text().replace(".csl/policies/", "policies/")
    (old / ".csl/venom/state.json").write_text(state)
    w = Workspace(old)
    assert w.policies == (old / "policies").resolve() or w.policies == old / "policies"
    b = Bindings(w).get("publisher")
    assert b is not None and b.policy == "policies/publisher.csl" and (old / b.mapping).exists()
    rc, out, _ = run_cli(["limits", "--agent", "publisher", "--check", "--workspace", str(old), "--root", str(HOST_OPS)],
                         capsys)
    assert rc == 0 and "as its limits say" in out
