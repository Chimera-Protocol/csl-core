"""
0.5.1 compatibility contract.

The working tree is captured with tests/contract/capture.py and compared with goldens
generated from tag v0.5.1 by scripts/regen_contract.py. Allowed differences, nothing else:

  * version strings (normalised by the capture itself);
  * the top-level `cslcore` command list may gain the 0.6 commands (setup, venom, policy,
    studio, map, exempt, mode, hook, watch);
  * new MCP tools, resources and prompts may be added (existing ones are frozen);
  * new public names may be added; an existing callable may gain a parameter only if it
    is optional and does not shift existing positional parameters.

Never regenerate the goldens to make this suite pass.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import capture  # noqa: E402

ALLOWED_NEW_COMMANDS = {"setup", "venom", "studio", "policy", "map", "exempt", "mode", "hook", "watch", "wire", "limits", "apply"}
FROZEN_PATHS = [
    "chimera_core/runtime.py",
    "chimera_core/language",
    "chimera_core/engines",
    "chimera_core/factory.py",
]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def current() -> Dict[str, str]:
    data = capture.run_sharded(REPO, GOLDEN)
    return {"__raw__": data, **capture.split_capture(data)}  # type: ignore[dict-item]


def _golden(rel: str) -> str:
    return (GOLDEN / rel).read_text(encoding="utf-8")


def _gjson(rel: str) -> Any:
    return json.loads(_golden(rel))


def _diff(expected: str, actual: str, name: str, limit: int = 80) -> str:
    lines = list(
        difflib.unified_diff(
            expected.splitlines(), actual.splitlines(),
            fromfile=f"golden/{name} (v0.5.1)", tofile=f"current/{name}", lineterm="",
        )
    )
    more = f"\n... {len(lines) - limit} more diff lines" if len(lines) > limit else ""
    return "\n".join(lines[:limit]) + more


_TLC_VERSION = re.compile(r"TLC2 Version [0-9.]+ \(rev: [0-9a-f]+\)")


def _tool_versions(v: Any) -> Any:
    """The TLC release the machine has is not 0.5.1 behaviour: compare it as a placeholder."""
    if isinstance(v, str):
        return _TLC_VERSION.sub("TLC2 Version <TLC>", v)
    if isinstance(v, dict):
        return {k: _tool_versions(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_tool_versions(x) for x in v]
    return v


def _assert_same(expected: Any, actual: Any, name: str) -> None:
    expected, actual = _tool_versions(expected), _tool_versions(actual)
    if expected != actual:
        e = expected if isinstance(expected, str) else json.dumps(expected, indent=1, sort_keys=True, default=str)
        a = actual if isinstance(actual, str) else json.dumps(actual, indent=1, sort_keys=True, default=str)
        pytest.fail(f"0.5.1 contract broken in {name}:\n{_diff(e, a, name)}", pytrace=False)


def _golden_files(prefix: str) -> List[str]:
    base = GOLDEN / prefix
    return sorted(str(p.relative_to(GOLDEN)) for p in base.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# signature compatibility (additive optional parameters only)
# ---------------------------------------------------------------------------

_POSITIONAL = {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}


def signature_problems(golden: List[Dict[str, Any]] | None, cur: List[Dict[str, Any]] | None) -> List[str]:
    if golden is None or cur is None:
        return [] if golden == cur else [f"signature availability changed: {golden!r} -> {cur!r}"]
    problems: List[str] = []
    cur_by_name = {p["name"]: (i, p) for i, p in enumerate(cur)}
    golden_positional = [p["name"] for p in golden if p["kind"] in _POSITIONAL]
    for gp in golden:
        if gp["name"] not in cur_by_name:
            problems.append(f"parameter '{gp['name']}' removed")
            continue
        _, cp = cur_by_name[gp["name"]]
        for field in ("kind", "has_default", "default", "annotation"):
            if gp.get(field) != cp.get(field):
                problems.append(f"parameter '{gp['name']}' {field} changed: {gp.get(field)!r} -> {cp.get(field)!r}")
    cur_positional = [p["name"] for p in cur if p["kind"] in _POSITIONAL]
    if cur_positional[: len(golden_positional)] != golden_positional:
        problems.append(f"positional order changed: {golden_positional} -> {cur_positional}")
    golden_names = {p["name"] for p in golden}
    for cp in cur:
        if cp["name"] in golden_names:
            continue
        if cp["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            if not any(g["kind"] == cp["kind"] for g in golden):
                problems.append(f"new variadic parameter '{cp['name']}'")
        elif not cp["has_default"]:
            problems.append(f"new parameter '{cp['name']}' is required (must be optional)")
    return problems


def describe_problems(g: Dict[str, Any], c: Dict[str, Any], where: str) -> List[str]:
    out: List[str] = []
    if g.get("type") != c.get("type"):
        return [f"{where}: kind changed {g.get('type')} -> {c.get('type')}"]
    if g["type"] == "function":
        out += [f"{where}: {p}" for p in signature_problems(g["params"], c["params"])]
    elif g["type"] == "class":
        if g.get("bases") != c.get("bases"):
            out.append(f"{where}: bases changed {g.get('bases')} -> {c.get('bases')}")
        if g.get("dataclass_fields") != c.get("dataclass_fields") or g.get("frozen") != c.get("frozen"):
            out.append(f"{where}: dataclass fields changed")
        for m, gm in g.get("members", {}).items():
            cm = c.get("members", {}).get(m)
            if cm is None:
                out.append(f"{where}.{m}: removed")
            elif gm.get("type") != cm.get("type"):
                out.append(f"{where}.{m}: kind changed {gm.get('type')} -> {cm.get('type')}")
            elif "params" in gm:
                out += [f"{where}.{m}: {p}" for p in signature_problems(gm["params"], cm["params"])]
    elif g != c:
        out.append(f"{where}: value changed {g!r} -> {c!r}")
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_goldens_come_from_v051_tag():
    manifest = _gjson("manifest.json")
    assert manifest["tag"] == "v0.5.1"
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "v0.5.1^{commit}"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        pytest.skip("tag v0.5.1 not fetched")
    assert manifest["commit"] == commit


def test_api_surface(current):
    golden = _gjson("api_surface.json")
    cur = json.loads(current["api_surface.json"])
    problems: List[str] = []
    for mod, gentry in golden.items():
        if "import_error" in gentry:
            continue
        centry = cur.get(mod, {})
        if "import_error" in centry:
            problems.append(f"{mod}: no longer importable ({centry['import_error']})")
            continue
        if "__all__" in gentry:
            missing = sorted(set(gentry["__all__"]) - set(centry.get("__all__", [])))
            if missing:
                problems.append(f"{mod}.__all__: removed {missing}")
        for name, g in gentry.items():
            if name == "__all__":
                continue
            c = centry.get(name)
            if c is None:
                problems.append(f"{mod}.{name}: removed")
                continue
            problems += describe_problems(g, c, f"{mod}.{name}")
    if problems:
        pytest.fail("0.5.1 public API changed:\n  " + "\n  ".join(problems), pytrace=False)


def test_cli_top_level(current):
    g = _gjson("cli/top_level.json")
    c = json.loads(current["cli/top_level.json"])
    for key in ("description", "prog", "options"):
        _assert_same(g[key], c[key], f"cli/top_level.json:{key}")
    for cmd, help_text in g["commands"].items():
        assert cmd in c["commands"], f"command '{cmd}' removed"
        _assert_same(help_text, c["commands"][cmd], f"cli command help '{cmd}'")
    extra = set(c["commands"]) - set(g["commands"])
    assert extra <= ALLOWED_NEW_COMMANDS, f"unexpected new top-level commands: {sorted(extra - ALLOWED_NEW_COMMANDS)}"


@pytest.mark.parametrize("cmd", ["verify", "simulate", "formal", "repl"])
def test_cli_subcommand_help(current, cmd):
    rel = f"cli/help_{cmd}.txt"
    _assert_same(_golden(rel), current.get(rel, "<missing>"), rel)


def _top_usage(text: str) -> str:
    """Allowed difference: the top-level usage line may list the 0.6 commands."""
    import re

    def drop_new(m):
        cmds = [c for c in m.group(1).split(",") if c not in ALLOWED_NEW_COMMANDS]
        return "{" + ",".join(cmds) + "}"

    text = re.sub(r"(usage: cslcore \[-h\] \[--version\])\n\s+(\{)", r"\1 \2", text)
    text = re.sub(r"(usage: cslcore \[-h\] \[--version\] \{[a-z,]+\})\n\s+(\.\.\.)", r"\1 \2", text)
    return re.sub(r"(?<=usage: cslcore \[-h\] \[--version\] )\{([a-z,]+)\}", drop_new, text)


def test_cli_runs(current):
    rels = _golden_files("cli/runs")
    assert len(rels) > 100
    failures = [rel for rel in rels if _top_usage(current.get(rel, "")) != _golden(rel)]
    if failures:
        first = failures[0]
        pytest.fail(
            f"{len(failures)} CLI outputs changed (first shown):\n" + _diff(_golden(first), current.get(first, ""), first),
            pytrace=False,
        )


def test_mcp_tools_resources_prompts(current):
    g = _gjson("mcp_tools.json")
    c = current["__raw__"]["mcp"]
    assert "import_error" not in g, "goldens were generated without the mcp extra"
    assert "import_error" not in c, "mcp extra missing; install csl-core[mcp] to run the contract suite"
    for section, key in (("tools", "name"), ("resources", "uri"), ("prompts", "name")):
        cur_by = {x[key]: x for x in c[section]}
        for item in g[section]:
            assert item[key] in cur_by, f"MCP {section[:-1]} '{item[key]}' removed"
            _assert_same(item, cur_by[item[key]], f"mcp {section[:-1]} {item[key]}")
    for uri, content in g["resource_content"].items():
        _assert_same(content, c["resource_content"].get(uri), f"mcp resource content {uri}")
    for name, text in g["prompt_text"].items():
        _assert_same(text, c["prompt_text"].get(name), f"mcp prompt {name}")


def test_mcp_tool_outputs(current):
    g = _gjson("mcp_tools.json")
    c = current["__raw__"]["mcp"]
    for policy, outputs in g["tool_outputs"].items():
        for tool, text in outputs.items():
            _assert_same(text, c["tool_outputs"].get(policy, {}).get(tool), f"mcp {tool} on {policy}")
    for policy, text in g["tla_verify_mock"].items():
        _assert_same(text, c["tla_verify_mock"].get(policy), f"mcp tla_verify on {policy}")
    for name, text in g["edge_cases"].items():
        _assert_same(text, c["edge_cases"].get(name), f"mcp edge case {name}")


def test_policy_hashes(current):
    g = _gjson("policy_hashes.json")
    assert len(g) >= 20
    _assert_same(g, json.loads(current["policy_hashes.json"]), "policy_hashes.json")


def test_decisions(current):
    g = _gjson("decisions.json")
    c = json.loads(current["decisions.json"])
    assert set(g) == set(c), "decision corpus changed"
    for policy in sorted(g):
        for config in sorted(g[policy]):
            _assert_same(g[policy][config], c[policy].get(config), f"decisions {policy} [{config}]")


def test_factory(current):
    _assert_same(_gjson("factory.json"), json.loads(current["factory.json"]), "factory.json")


def test_import_isolation():
    code = (
        "import sys, chimera_core, chimera_core.cli; "
        "bad = sorted(m for m in sys.modules if m.startswith('chimera_core.venom') or m == 'chimera_core.mapping'); "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_frozen_files_untouched():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    probe = subprocess.run(["git", "rev-parse", "v0.5.1^{commit}"], cwd=REPO, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("tag v0.5.1 not fetched")
    r = subprocess.run(
        ["git", "diff", "--stat", "v0.5.1", "--", *FROZEN_PATHS], cwd=REPO, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", f"frozen files changed since v0.5.1:\n{r.stdout}"
