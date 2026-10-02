"""
Capture the observable behavior of a csl-core source tree.

Used two ways:
  * scripts/regen_contract.py runs it against a git tag (v0.5.1) to write goldens.
  * tests/contract/test_contract.py runs it against the working tree and compares.

Run as a script in a fresh interpreter so the tree under test is the only
chimera_core on sys.path:

    python tests/contract/capture.py --src <root> --golden <dir> --out <file.json>
    python tests/contract/capture.py --src <root> --golden <dir> --make-inputs

Everything captured is normalised by exactly the rules listed in NORMALISATION.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import inspect
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

NORMALISATION = [
    "package version string replaced by <VERSION>",
    "latency_ms removed from guard results",
    "TLC time, PID and worker lines removed from tla_verify output",
    "memory addresses (' at 0x...') removed from reprs",
    "argparse 'optional arguments:' heading rendered as 'options:'",
    "durations ('<number>ms', '<number>s', 'h:mm:ss') replaced by <T>; space runs collapsed on those lines",
    "TLA+ runs use the Python model checker (real TLC depends on a local Java install)",
]

HELP_COLUMNS = "100"
CONSOLE_WIDTH = 100

# Policy source folders (relative to a source root) whose *.csl files form the corpus.
POLICY_DIRS = ["examples", "examples/community", "quickstart", "chimera_core/mcp/examples"]

# Test-case files shipped with the examples, keyed by corpus policy name.
JSON_CASES = {
    "examples__agent_tool_guard": "examples/json_files/agent_tool_guard_tests.json",
    "examples__chimera_banking_case_study": "examples/json_files/banking_tests.json",
    "examples__dao_treasury_guard": "examples/json_files/dao_treasury_guard_tests.json",
    "examples__openclaw_guard": "examples/json_files/openclaw_guard_tests.json",
}

RUNTIME_CONFIGS = {
    "no_raise": dict(raise_on_block=False),
    "default": dict(),
    "dry_run": dict(dry_run=True, raise_on_block=False),
    "missing_warn": dict(missing_key_behavior="warn", raise_on_block=False),
    "missing_ignore": dict(missing_key_behavior="ignore", raise_on_block=False),
    "eval_warn": dict(evaluation_error_behavior="warn", raise_on_block=False),
    "fast_fail": dict(collect_all_violations=False, raise_on_block=False),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")
_TLC_VOLATILE = re.compile(r"^.*\*\*(Time|PID|Workers):\*\*.*$\n?", re.MULTILINE)
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s)\b|\b\d+:\d\d:\d\d\b")
_PID = re.compile(r"\bpid \d+")


def _norm_line(line: str) -> str:
    new = _PID.sub("pid <PID>", _DURATION.sub("<T>", line))
    if new != line:
        new = re.sub(r" {2,}", " ", new)
    return new


class _Norm:
    version = ""

    @classmethod
    def text(cls, s: str) -> str:
        if cls.version:
            s = s.replace(cls.version, "<VERSION>")
        s = _ADDR.sub("", s)
        if _DURATION.search(s) or _PID.search(s):
            s = "\n".join(_norm_line(x) for x in s.split("\n"))
        return s

    @classmethod
    def deep(cls, obj: Any) -> Any:
        if isinstance(obj, str):
            return cls.text(obj)
        if isinstance(obj, list):
            return [cls.deep(x) for x in obj]
        if isinstance(obj, dict):
            return {k: cls.deep(v) for k, v in obj.items()}
        return obj


def _repr(v: Any) -> str:
    """repr() with sets sorted, so hash randomisation never changes a golden."""
    if isinstance(v, (set, frozenset)):
        inner = ", ".join(sorted(_repr(x) for x in v))
        return ("{" + inner + "}") if isinstance(v, set) and v else f"{type(v).__name__}({{{inner}}})"
    return _Norm.text(repr(v))


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def corpus(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for d in POLICY_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.csl")):
            key = d.replace("/", "__") + "__" + p.stem
            out[key] = p
    return out


# (index, count): this process captures only the policies whose position % count == index.
# Shard 0 also captures the global sections (API surface, CLI help, MCP listings, edge cases).
SHARD = (0, 1)


def golden_corpus(golden: Path, *, all_shards: bool = False) -> Dict[str, Path]:
    paths = sorted((golden / "policies").glob("*.csl"))
    k, n = SHARD
    return {p.stem: p for i, p in enumerate(paths) if all_shards or i % n == k}


def _primary() -> bool:
    return SHARD[0] == 0


# ---------------------------------------------------------------------------
# inputs (generated once, from the golden tag, then frozen)
# ---------------------------------------------------------------------------

def _parse_domain(dom: str):
    dom = dom.strip()
    if dom.startswith("{"):
        vals = re.findall(r'"([^"]*)"', dom)
        return ("enum", vals)
    m = re.fullmatch(r"(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)", dom)
    if m:
        lo, hi = m.group(1), m.group(2)
        if "." in lo or "." in hi:
            return ("float", (float(lo), float(hi)))
        return ("int", (int(lo), int(hi)))
    if dom.upper() in ("BOOLEAN", "BOOL"):
        return ("bool", None)
    return ("other", dom)


def _base_value(kind, spec, which: str):
    if kind == "enum":
        return spec[0] if which == "low" else spec[-1]
    if kind in ("int", "float"):
        return spec[0] if which == "low" else spec[1]
    if kind == "bool":
        return which != "low"
    return "sample" if which == "low" else "other"


def _rule_constants(path: Path, var: str) -> List[float]:
    """Numeric literals on rule lines (comparisons) that mention `var`."""
    consts = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("//")[0]
        if not re.search(r"[<>]|==|!=|\bMUST\b", code) or not re.search(rf"\b{re.escape(var)}\b", code):
            continue
        found = []
        for m in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])", code):
            txt = m.group(1)
            found.append(float(txt) if "." in txt else int(txt))
        consts.update(found)
        # Scaled comparisons such as `amount * 2 >= 5000` have their boundary at 5000 / 2.
        if "*" in code:
            for x in found:
                for y in found:
                    if y not in (0, 1) and x != y and isinstance(x, int) and isinstance(y, int) and x % y == 0:
                        consts.add(x // y)
    return sorted(consts)


def make_inputs(root: Path, golden: Path) -> Dict[str, Any]:
    from chimera_core.language.parser import parse_csl_file

    inputs: Dict[str, Any] = {}
    for name, path in golden_corpus(golden, all_shards=True).items():
        cases: List[Dict[str, Any]] = []
        shipped = JSON_CASES.get(name)
        if shipped and (root / shipped).exists():
            data = json.loads((root / shipped).read_text(encoding="utf-8"))
            for group in ("allow_cases", "block_cases", "cases"):
                for c in data.get(group, []) if isinstance(data, dict) else []:
                    if isinstance(c, dict) and isinstance(c.get("input"), dict):
                        cases.append({"name": f"shipped:{c.get('name', '')}", "input": c["input"]})
        try:
            with _quiet():
                ast = parse_csl_file(str(path))
            decls = [(d.name, _parse_domain(d.domain)) for d in ast.domain.variable_declarations]
        except Exception:
            decls = []

        if decls:
            low = {n: _base_value(k, s, "low") for n, (k, s) in decls}
            high = {n: _base_value(k, s, "high") for n, (k, s) in decls}
            cases.append({"name": "all_low", "input": dict(low)})
            cases.append({"name": "all_high", "input": dict(high)})
            for n, (k, s) in decls:
                if k == "enum":
                    for v in s:
                        cases.append({"name": f"{n}={v}", "input": {**low, n: v}})
                    cases.append({"name": f"{n}=case_variant", "input": {**low, n: s[0].lower() if s[0] != s[0].lower() else s[0].upper()}})
                    cases.append({"name": f"{n}=unknown", "input": {**low, n: "__UNKNOWN_VALUE__"}})
                    cases.append({"name": f"{n}=wrong_type", "input": {**low, n: 1}})
                elif k in ("int", "float"):
                    lo, hi = s
                    step = 1 if k == "int" else 0.5
                    for v in (lo - step, hi, hi + step, (lo + hi) / 2 if k == "float" else (lo + hi) // 2):
                        cases.append({"name": f"{n}={v}", "input": {**low, n: v}})
                    # Boundary values around every constant a rule compares this variable with.
                    for c in _rule_constants(path, n):
                        for v in sorted({c - step, c, c + step}):
                            for base_name, base in (("low", low), ("high", high)):
                                cases.append({"name": f"{n}={v}@{base_name}", "input": {**base, n: v}})
                    cases.append({"name": f"{n}=numeric_string", "input": {**low, n: str(hi)}})
                    cases.append({"name": f"{n}=wrong_type", "input": {**low, n: "not_a_number"}})
                elif k == "bool":
                    cases.append({"name": f"{n}=true", "input": {**low, n: True}})
                    cases.append({"name": f"{n}=string", "input": {**low, n: "YES"}})
                cases.append({"name": f"{n}=missing", "input": {kk: vv for kk, vv in low.items() if kk != n}})
            cases.append({"name": "empty", "input": {}})
        inputs[name] = cases
    return inputs


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def _param(p: inspect.Parameter) -> Dict[str, Any]:
    return {
        "name": p.name,
        "kind": p.kind.name,
        "default": None if p.default is inspect.Parameter.empty else _repr(p.default),
        "has_default": p.default is not inspect.Parameter.empty,
        "annotation": None if p.annotation is inspect.Parameter.empty else (
            p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", repr(p.annotation))
        ),
    }


def _sig(obj) -> Any:
    try:
        return [_param(p) for p in inspect.signature(obj).parameters.values()]
    except (TypeError, ValueError):
        return None


def _describe(obj) -> Dict[str, Any]:
    if obj is None:
        return {"type": "None"}
    if inspect.isclass(obj):
        d: Dict[str, Any] = {"type": "class", "bases": [b.__name__ for b in obj.__bases__]}
        if dataclasses.is_dataclass(obj):
            fields = []
            for f in dataclasses.fields(obj):
                if f.default is not dataclasses.MISSING:
                    default = _repr(f.default)
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    default = "factory:" + _repr(f.default_factory())  # type: ignore[misc]
                else:
                    default = None
                fields.append({"name": f.name, "type": str(f.type), "default": default})
            d["dataclass_fields"] = fields
            d["frozen"] = obj.__dataclass_params__.frozen  # type: ignore[attr-defined]
        members = {}
        for name, val in sorted(vars(obj).items()):
            if name.startswith("_") and name not in ("__init__", "__call__"):
                continue
            if isinstance(val, property):
                members[name] = {"type": "property"}
            elif isinstance(val, (staticmethod, classmethod)):
                members[name] = {"type": type(val).__name__, "params": _sig(val.__func__)}
            elif inspect.isfunction(val):
                members[name] = {"type": "method", "params": _sig(val)}
        d["members"] = members
        return d
    if callable(obj):
        return {"type": "function", "params": _sig(obj)}
    if inspect.ismodule(obj):
        return {"type": "module"}
    return {"type": type(obj).__name__, "value": _repr(obj)}


def api_surface() -> Dict[str, Any]:
    import importlib

    out: Dict[str, Any] = {}

    def add_module(modname: str, names: List[str] | None = None):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # optional extras
            out[modname] = {"import_error": type(e).__name__}
            return
        if names is None:
            names = list(getattr(mod, "__all__", []))
        entry: Dict[str, Any] = {"__all__": sorted(getattr(mod, "__all__", []))} if hasattr(mod, "__all__") else {}
        for n in names:
            entry[n] = _describe(getattr(mod, n, None)) if hasattr(mod, n) else {"type": "missing"}
        out[modname] = entry

    add_module("chimera_core")
    add_module("chimera_core.language")
    add_module("chimera_core.language.parser", ["parse_csl", "parse_csl_file"])
    add_module("chimera_core.language.compiler", ["CSLCompiler", "CompiledConstitution", "CompiledConstraint", "CompilationError"])
    add_module("chimera_core.runtime", ["ChimeraGuard", "ChimeraError", "GuardResult", "RuntimeConfig"])
    add_module("chimera_core.factory", ["load_guard", "create_guard_from_string"])
    add_module("chimera_core.engines")
    add_module("chimera_core.plugins.base", ["ChimeraPlugin", "default_context_mapper", "safe_model_dump"])
    add_module("chimera_core.plugins.langchain", ["ChimeraRunnableGate", "gate", "GuardedTool", "wrap_tool", "guard_tools"])
    add_module("chimera_core.plugins.openclaw")
    add_module("chimera_core.cli", ["main", "build_parser"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _help_text(parser) -> str:
    s = parser.format_help()
    return _Norm.text(s.replace("optional arguments:", "options:"))


def cli_help() -> Dict[str, Any]:
    from chimera_core import cli

    parser = cli.build_parser()
    sub_action = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    top = {
        "description": parser.description,
        "prog": parser.prog,
        "options": sorted(
            [",".join(a.option_strings) for a in parser._actions if a.option_strings]
        ),
        "commands": {
            ca.dest: _Norm.text(ca.help or "") for ca in sub_action._choices_actions
        },
    }
    helps = {name: _help_text(sp) for name, sp in sub_action.choices.items()}
    return {"top": top, "help": helps}


def _run_cli(argv: List[str]) -> Dict[str, Any]:
    from rich.console import Console
    from chimera_core import cli

    buf = io.StringIO()
    saved = cli.console
    cli.console = Console(
        file=buf, width=CONSOLE_WIDTH, color_system=None, force_terminal=False,
        no_color=True, highlight=False, emoji=True, legacy_windows=False,
    )
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                rc = cli.main(argv)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
    finally:
        cli.console = saved
    return {"exit_code": rc, "stdout": _Norm.text(buf.getvalue())}


def cli_runs(golden: Path, inputs: Dict[str, Any], tmp: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cwd = os.getcwd()
    os.chdir(golden)
    try:
        for name in sorted(golden_corpus(golden)):
            rel = f"policies/{name}.csl"
            ctxs = [c["input"] for c in inputs.get(name, [])][:12]
            infile = tmp / f"{name}.inputs.json"
            infile.write_text(json.dumps(ctxs), encoding="utf-8")
            out[name] = {
                "verify": _run_cli(["verify", rel]),
                "simulate": _run_cli(["simulate", rel, "--input-file", str(infile), "--no-raise"]),
                "simulate_json": _run_cli(
                    ["simulate", rel, "--input-file", str(infile), "--json", "--quiet", "--no-raise"]
                ),
                "simulate_raise": _run_cli(["simulate", rel, "--input-file", str(infile), "--quiet"]),
                "simulate_dry_run": _run_cli(
                    ["simulate", rel, "--input-file", str(infile), "--dry-run", "--quiet"]
                ),
            }
        if _primary():
            out["__errors__"] = {
                "missing_file": _run_cli(["verify", "policies/__does_not_exist__.csl"]),
                "no_command": _run_cli([]),
                "bad_flag": _run_cli(["verify", "--nope"]),
            }
    finally:
        os.chdir(cwd)
    return out


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def _dump_model(m) -> Any:
    if hasattr(m, "model_dump"):
        return json.loads(json.dumps(m.model_dump(mode="json", exclude_none=True), default=str))
    return str(m)


def mcp_surface(golden: Path, inputs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from chimera_core.mcp import server
    except Exception as e:
        return {"import_error": type(e).__name__}

    def t(s: str) -> str:
        return _TLC_VOLATILE.sub("", _Norm.text(s))

    out: Dict[str, Any] = {"tool_outputs": {}, "tla_verify_mock": {}}
    for name, path in golden_corpus(golden).items():
        src = path.read_text(encoding="utf-8")
        ctxs = [c["input"] for c in inputs.get(name, [])][:8]
        with _quiet():
            out["tool_outputs"][name] = {
                "verify_policy": t(server.verify_policy(src)),
                "explain_policy": t(server.explain_policy(src)),
                "universe_info": t(server.universe_info(src)),
                "simulate_policy": t(server.simulate_policy(src, json.dumps(ctxs))),
                "simulate_policy_single": t(server.simulate_policy(src, json.dumps(ctxs[0] if ctxs else {}))),
                "simulate_policy_dry_run": t(server.simulate_policy(src, json.dumps(ctxs), dry_run=True)),
            }
            if "tla_demo" in name or "hello_world" in name:
                out["tla_verify_mock"][name] = t(server.tla_verify(src, timeout=60, use_mock=True))

    if not _primary():
        return out

    app = server.mcp

    async def lists():
        tools = await app.list_tools()
        resources = await app.list_resources()
        prompts = await app.list_prompts()
        res_content = {}
        for r in resources:
            contents = await app.read_resource(str(r.uri))
            res_content[str(r.uri)] = [getattr(c, "content", str(c)) for c in contents]
        prompt_text = {}
        for p in prompts:
            if any(getattr(a, "required", False) for a in (p.arguments or [])):
                continue  # prompts that need arguments are listed, not rendered
            got = await app.get_prompt(p.name, {})
            prompt_text[p.name] = _dump_model(got)
        return tools, resources, prompts, res_content, prompt_text

    tools, resources, prompts, res_content, prompt_text = asyncio.run(lists())
    with _quiet():
        out["edge_cases"] = {
            "verify_parse_error": t(server.verify_policy("DOMAIN {")),
            "simulate_bad_json": t(server.simulate_policy("DOMAIN X {}", "{not json")),
            "scaffold_basic": t(server.scaffold_policy("PaymentGuard", "Limit transfers")),
            "scaffold_vars": t(server.scaffold_policy("AgentSafety", "Tool gate", "tool, amount, role")),
        }
    out.update({
        "tools": [
            {"name": x.name, "description": _Norm.text(x.description or ""), "input_schema": x.inputSchema}
            for x in sorted(tools, key=lambda x: x.name)
        ],
        "resources": [
            {"uri": str(r.uri), "name": r.name, "description": r.description, "mime_type": r.mimeType}
            for r in sorted(resources, key=lambda r: str(r.uri))
        ],
        "resource_content": {k: [_Norm.text(c) for c in v] for k, v in sorted(res_content.items())},
        "prompts": [
            {"name": p.name, "description": p.description, "arguments": _dump_model(p).get("arguments")}
            for p in sorted(prompts, key=lambda p: p.name)
        ],
        "prompt_text": _Norm.deep(prompt_text),
    })
    return out


# ---------------------------------------------------------------------------
# policy hashes and decisions
# ---------------------------------------------------------------------------

def _compile(path: Path):
    from chimera_core.language.parser import parse_csl_file
    from chimera_core.language.compiler import CSLCompiler

    with _quiet():
        ast = parse_csl_file(str(path))
        return CSLCompiler().compile(ast)


_META = ("domain_name", "policy_name", "policy_id", "policy_version", "policy_hash", "engine_version")


def _result_dict(r, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    d = _Norm.deep(dataclasses.asdict(r))
    d.pop("latency_ms", None)
    if meta is not None:
        # Policy metadata is recorded once per policy; keep it on a row only if it differs.
        for k in _META:
            if k in d and d[k] == meta.get(k):
                d.pop(k)
    return d


def _row(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def policies_and_decisions(golden: Path, inputs: Dict[str, Any]):
    from chimera_core.runtime import ChimeraGuard, ChimeraError, RuntimeConfig

    hashes: Dict[str, Any] = {}
    decisions: Dict[str, Any] = {}
    for name, path in golden_corpus(golden).items():
        try:
            compiled = _compile(path)
        except Exception as e:
            hashes[name] = {"error": type(e).__name__, "message": _Norm.text(str(e))[:2000]}
            continue
        hashes[name] = {
            "policy_hash": compiled.policy_hash,
            "policy_id": compiled.policy_id,
            "policy_version": compiled.policy_version,
            "domain_name": compiled.domain_name,
            "constraints": [c.name for c in compiled.constraints],
            "variable_domains": {k: str(v) for k, v in compiled.variable_domains.items()},
        }
        meta = _Norm.deep({k: getattr(compiled, k, None) for k in _META})
        per: Dict[str, Any] = {"__meta__": meta}
        for cname, kwargs in RUNTIME_CONFIGS.items():
            guard = ChimeraGuard(compiled, RuntimeConfig(**kwargs))
            rows = []
            for case in inputs.get(name, []):
                try:
                    rows.append(_row({"case": case["name"], "result": _result_dict(guard.verify(case["input"]), meta)}))
                except ChimeraError as e:
                    rows.append(_row({
                        "case": case["name"],
                        "raised": type(e).__name__,
                        "message": _Norm.text(str(e)),
                        "constraint_name": e.constraint_name,
                        "result": _result_dict(e.result, meta) if e.result is not None else None,
                    }))
                except Exception as e:
                    rows.append(_row({"case": case["name"], "raised": type(e).__name__, "message": _Norm.text(str(e))}))
            per[cname] = rows
        decisions[name] = per
    return hashes, decisions


def factory_checks(golden: Path) -> Dict[str, Any]:
    from chimera_core import load_guard, create_guard_from_string

    out = {}
    for name, path in golden_corpus(golden).items():
        try:
            with _quiet():
                g1 = load_guard(str(path))
                g2 = create_guard_from_string(path.read_text(encoding="utf-8"))
            out[name] = {
                "load_guard_hash": g1.constitution.policy_hash,
                "from_string_hash": g2.constitution.policy_hash,
                "config": dataclasses.asdict(g1.config),
            }
        except Exception as e:
            out[name] = {"error": type(e).__name__}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def capture_all(root: Path, golden: Path, tmp: Path) -> Dict[str, Any]:
    inputs = json.loads((golden / "inputs.json").read_text(encoding="utf-8"))
    hashes, decisions = policies_and_decisions(golden, inputs)
    data = {
        "cli_runs": cli_runs(golden, inputs, tmp),
        "mcp": mcp_surface(golden, inputs),
        "policy_hashes": hashes,
        "decisions": decisions,
        "factory": factory_checks(golden),
    }
    if _primary():
        data["api_surface"] = api_surface()
        data["cli_help"] = cli_help()
    return data


def merge(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge shard captures (dict keys are disjoint below the section level)."""
    out: Dict[str, Any] = {}
    for part in parts:
        for k, v in part.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge([out[k], v])
            elif k not in out:
                out[k] = v
    return out


def run_sharded(src: Path, golden: Path, shards: int | None = None, python: str | None = None) -> Dict[str, Any]:
    """Capture `src` in `shards` parallel fresh interpreters and merge the results."""
    import subprocess
    import tempfile

    shards = shards or max(1, min(8, os.cpu_count() or 1))
    python = python or sys.executable
    with tempfile.TemporaryDirectory() as td:
        procs = []
        for k in range(shards):
            outf = Path(td) / f"shard{k}.json"
            cmd = [python, str(Path(__file__).resolve()), "--src", str(src), "--golden", str(golden),
                   "--shard", f"{k}/{shards}", "--out", str(outf)]
            env = {**os.environ, "PYTHONHASHSEED": "0"}
            procs.append((subprocess.Popen(cmd, cwd=str(src), env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True), outf))
        parts = []
        for proc, outf in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"capture shard failed ({proc.returncode}):\n{err[-4000:]}")
            parts.append(json.loads(outf.read_text(encoding="utf-8")))
    return merge(parts)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def split_capture(data: Dict[str, Any]) -> Dict[str, str]:
    """Map a capture onto the golden file layout: relative path -> file content."""
    files = {
        "api_surface.json": _dumps(data["api_surface"]),
        "cli/top_level.json": _dumps(data["cli_help"]["top"]),
        "mcp_tools.json": _dumps(data["mcp"]),
        "policy_hashes.json": _dumps(data["policy_hashes"]),
        "decisions.json": _dumps(data["decisions"]),
        "factory.json": _dumps(data["factory"]),
    }
    for cmd, text in data["cli_help"]["help"].items():
        files[f"cli/help_{cmd}.txt"] = text
    for policy, runs in data["cli_runs"].items():
        for run, res in runs.items():
            files[f"cli/runs/{policy}.{run}.txt"] = f"exit_code: {res['exit_code']}\n---\n{res['stdout']}"
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source root whose chimera_core is captured")
    ap.add_argument("--golden", required=True, help="golden directory (policies/ and inputs.json)")
    ap.add_argument("--out", help="write capture JSON here")
    ap.add_argument("--make-inputs", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: capture every n-th policy starting at k")
    args = ap.parse_args()
    global SHARD
    k, n = (int(x) for x in args.shard.split("/"))
    SHARD = (k, n)

    root = Path(args.src).resolve()
    golden = Path(args.golden).resolve()
    os.environ["COLUMNS"] = HELP_COLUMNS
    os.environ["NO_COLOR"] = "1"
    os.environ["PYTHON_COLORS"] = "0"
    sys.path.insert(0, str(root))
    import chimera_core

    loaded = Path(chimera_core.__file__).resolve()
    if root not in loaded.parents:
        print(f"chimera_core loaded from {loaded}, expected under {root}", file=sys.stderr)
        return 2
    _Norm.version = chimera_core.__version__

    # Real TLC depends on a local Java install; pin the TLA+ path to the Python engine.
    from chimera_core.engines.tla_engine import tlc_runner
    tlc_runner.TLCRunner.is_available = lambda self: False  # type: ignore[method-assign]

    if args.make_inputs:
        inputs = make_inputs(root, golden)
        (golden / "inputs.json").write_text(json.dumps(inputs, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        data = capture_all(root, golden, Path(td))
    text = _dumps(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
