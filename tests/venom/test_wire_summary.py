"""Before a wiring diff: one line that says how many lines in how many files, whether anything that
exists changes, and how to undo it. A long diff is cut; cslcore wire --diff shows all of it and
changes nothing."""

from __future__ import annotations

import re

from .conftest import run_cli

TOOLS = "\n\n".join(f'''@tool
def step_{i}(path: str) -> str:
    """Write step {i}."""
    return "ok"''' for i in range(20))
AGENT = f'''"""Many steps."""


def tool(fn):
    return fn


{TOOLS}
'''


def _repo(tmp_path, capsys):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "app/agent.py").write_text(AGENT)
    ws = tmp_path / "ws"
    ws.mkdir()
    run_cli(["setup", "--root", str(root), "--workspace", str(ws), "--yes", "--activate", "--no-anim"], capsys)
    return root, ws


def test_summary_then_a_cut_diff_and_the_whole_one_on_request(tmp_path, capsys):
    root, ws = _repo(tmp_path, capsys)
    rc, out, _ = run_cli(["wire", "--root", str(root), "--workspace", str(ws), "--no-color"], capsys)
    flat = " ".join(out.split())
    assert re.search(r"\b2[0-9] lines added in 1 file; function bodies and existing lines are not touched", flat)
    assert "undo: cslcore wire --undo --agent repo" in flat
    assert flat.index("lines added in 1 file") < flat.index("+++")  # the summary comes before the diff
    assert "more diff lines · all of it: cslcore wire --agent repo --diff" in flat
    assert (ws / ".csl/venom/wire/repo.diff").read_text().count("@_csl_guard.tool") == 20
    assert (root / "app/agent.py").read_text() == AGENT  # not a terminal, no --yes: nothing changed
    rc, out, _ = run_cli(["wire", "--agent", "repo", "--diff", "--root", str(root), "--workspace", str(ws), "--yes",
                          "--no-color"], capsys)
    assert rc == 0 and out.count("@_csl_guard.tool") == 20 and "more diff lines" not in out
    assert "nothing was changed (--diff)" in out and (root / "app/agent.py").read_text() == AGENT


def test_a_rewritten_line_is_never_called_untouched():
    from chimera_core.venom import wiring
    from chimera_core.venom.wire_cmd import plan_summary

    plan = wiring.Plan("k", "k", "code", changes=[wiring.Change("f.py", "f.py", "a = 1\n", "a = 2\n", "")])
    text = plan_summary(plan)
    assert "1 existing line rewritten" in text and "not touched" not in text
