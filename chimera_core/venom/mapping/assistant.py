"""
`cslcore map`: mapping assistant and mapping test.
"""

from __future__ import annotations

import contextlib
import io
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..commands import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE, _load_inventory, console_for, workspace_for
from ..layers.governance import read_policy
from ..model import Agent, Inventory, PolicyRef
from ..policy import draft as D
from ..policy.workbench import _policy_refs, _text, agents_for, confirm, find_agent
from ..render.screen import _section
from . import codegen, harness
from . import tricks as tricks_mod
from .spec import MappingSpec, apply_classify, build_spec


def policy_for(ws, inv: Inventory, agent: Agent, explicit: Optional[str], args) -> Optional[PolicyRef]:
    if explicit:
        p = Path(explicit)
        text = ws.read(p)
        return read_policy(str(p.resolve()), text, "active") if text is not None else None
    refs = _policy_refs(ws, inv)
    key = D.agent_key(agent)
    for status in ("active", "draft", "found"):
        for r in refs:
            if r.status == status and r.error is None and (Path(r.path).stem == key or agent in agents_for(inv, r)):
                return r
    return None


def compile_guard(text: str):
    from ...language.compiler import CSLCompiler
    from ...language.parser import parse_csl
    from ...runtime import ChimeraGuard, RuntimeConfig

    with contextlib.redirect_stdout(io.StringIO()):
        compiled = CSLCompiler().compile(parse_csl(text))
    return ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))


def mapping_path(ws, agent: Agent) -> Path:
    return ws.policies / f"{D.agent_key(agent).replace('-', '_')}_mapping.py"


def render_spec(console, spec: MappingSpec) -> None:
    t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
    t.add_column("Variable", style="head", no_wrap=True)
    t.add_column("Domain", style="text", overflow="fold")
    t.add_column("Source", no_wrap=True)
    t.add_column("From", style="muted", overflow="fold")
    style = {"tool": "brand", "param": "ok", "constant": "muted", "derived": "configured", "context": "warn"}
    for v in spec.variables:
        dom = f"{v.low}..{v.high}" if v.kind == "range" else ("{" + ", ".join(v.values) + "}" if v.values else v.kind)
        if v.source == "tool":
            frm = f"{len(spec.tool_table)} tools"
        elif v.source == "param":
            parts = []
            for tn, p in v.params.items():
                coerce = ""
                if v.kind == "flag" and (p.type or "").lower() in ("bool", "boolean"):
                    coerce = " (bool -> YES/NO)"
                elif v.kind == "range" and (p.type or "").lower() in ("str", "string"):
                    coerce = " (numeric string -> int)"
                rename = "" if p.name == v.name else f" (named {p.name})"
                parts.append(f"{tn}.{p.name}{rename}{coerce}")
            frm = ", ".join(parts)
        elif v.source == "constant":
            frm = f'"{v.constant}"'
        elif v.source == "derived":
            frm = v.note or ""
        else:
            frm = "caller context (required)"
        t.add_row(v.name, dom, Text(v.source, style=style.get(v.source, "text")), frm)
    console.print(_section("VARIABLES"))
    console.print(t)
    if spec.tool_var:
        vt = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
        vt.add_column("Real tool name", style="head")
        vt.add_column(f"Policy value ({spec.tool_var})")
        for real, pol in spec.tool_table.items():
            vt.add_row(real, Text(pol, style="ok" if pol == real else "warn") + (Text("  drift, mapped", style="muted") if pol != real else Text("")))
        for real in spec.unmapped_tools:
            vt.add_row(real, Text("(unmapped: blocked)", style="high"))
        console.print(_section("VALUES"))
        console.print(vt)


def render_results(console, res: harness.HarnessResult, spec: MappingSpec, target: str, policy_rel: str, limit: int = 40) -> None:
    console.print(Text.assemble(("  MAPPING TEST   ", "label"), (spec.agent_name, "head"), (" → ", "muted"),
                                (policy_rel, "text"), (f"   {len(res.cases)} cases", "muted")))
    console.print(Text(f"                 {target}", style="muted"))
    wide = console.width >= 100
    t = Table(box=None, header_style="label", pad_edge=False, show_edge=False, padding=(0, 1, 0, 0))
    for col, w in (("  Variable", 24 if wide else 17), ("Input", 22 if wide else 14), ("Mapped to", 26 if wide else 16),
                   ("Decision", 9), ("Expected", 9), ("", 11)):
        t.add_column(col, no_wrap=True, overflow="ellipsis", width=w, min_width=w, max_width=w)
    plain = [c for c in res.cases if c.kind not in ("bypass", "regression")]
    shown = [c for c in plain if not c.ok]
    shown += [c for c in plain if c.ok][: max(0, limit - len(shown))]
    order = {t: i for i, t in enumerate(dict.fromkeys(c.tool for c in res.cases))}
    shown.sort(key=lambda c: (order[c.tool], res.cases.index(c)))
    current_tool = None
    for c in shown:
        if c.tool != current_tool:
            current_tool = c.tool
            t.add_row(Text(f"  {c.tool}", style="brand.dim"), "", "", "", "", "")
        dec = Text(c.decision, style="ok" if c.decision == "ALLOW" else "high")
        mark = Text("✓", style="ok") if c.ok else Text("✗ fail-open" if c.expected == "BLOCK" else "✗ unmapped", style="high")
        t.add_row("    " + c.variable, c.input, Text(c.mapped, style="muted" if c.mapped.startswith("(") else "text"),
                  dec, Text(c.expected, style="muted" if c.expected == "=" else "text"), mark)
    console.print(t)
    if len(plain) > len(shown):
        console.print(Text(f"  {len(plain) - len(shown)} more passing cases", style="muted"))
    render_bypass(console, res, spec)
    render_regression(console, res)
    fo = res.fail_open
    bad = len(res.failed)
    line = Text("  RESULT       ", style="label")
    line.append(f"{len(res.cases) - bad} of {len(res.cases)} pass", style="ok" if not bad else "text")
    line.append(" · ")
    line.append(f"{len(fo)} fail-open", style="high" if fo else "ok")
    if res.bypass:
        line.append(f" · {len(res.bypass)} bypass tricks", style="muted")
    console.print()
    console.print(line)
    plain_fo = [c for c in fo if c.kind == "malformed"]
    if plain_fo:
        c = plain_fo[0]
        console.print(Text(f"               fix: map {c.variable} with to_enum / to_range / to_flag (fail-closed)", style="muted"))
    for var in dict.fromkeys(c.variable for c in fo if c.kind == "bypass"):
        kind = spec.classify.get(var, ("scope", None))[0]
        _what, helper = tricks_mod.describe(kind)
        console.print(Text(f"               fix: compute {var} with {helper} (passes every trick family)", style="muted"))
    if res.inconclusive:
        console.print(Text(f"  note: the base call of {', '.join(res.inconclusive)} is blocked by the policy, so malformed "
                           "inputs for it cannot show fail-open", style="warn"))


def render_bypass(console, res: harness.HarnessResult, spec: MappingSpec) -> None:
    fams = res.families()
    if not fams and not res.uncovered:
        return
    console.print()
    console.print(Text.assemble(("  BYPASS TRICKS  ", "label"),
                                ("inputs built to land outside what the mapping allows; each must end in BLOCK", "muted")))
    for var, by_family in fams.items():
        leaked = {f: n for f, n in by_family.items() if n}
        kind = spec.classify.get(var, ("?", None))[0]
        tools = sorted({c.tool for c in res.bypass if c.variable == var})
        head = Text.assemble(("  ", ""), ("✗ " if leaked else "✓ ", "high" if leaked else "ok"), (var, "head"),
                             (f"  {kind} · {len(by_family)} families · {len(tools)} tool{'s' if len(tools) != 1 else ''}", "muted"))
        console.print(head)
        line = Text("      ")
        for i, (fam, n) in enumerate(by_family.items()):
            if i:
                line.append("  ")
            line.append(fam, style="high" if n else "ok")
        console.print(line)
        for c in [c for c in res.bypass if c.variable == var and c.decision == "ALLOW"][:6]:
            console.print(Text.assemble(("      ✗ ", "high"), (f"{c.family:<16}", "warn"), (f"{c.tool}  ", "muted"),
                                        (c.input, "text"), ("  → ", "muted"), (c.mapped, "text"), ("  ALLOW", "high")))
    groups: Dict[Tuple[str, str], List[str]] = {}
    for u in res.uncovered:
        groups.setdefault((u.variable, u.reason), []).append(u.tool)
    for (var, reason), tools in groups.items():
        names = ", ".join(tools[:4]) + (f" and {len(tools) - 4} more" if len(tools) > 4 else "")
        console.print(Text.assemble(("  · ", "warn"), (var, "head"), (f" ({names})", "muted"), ("  not covered: ", "warn"),
                                    (reason, "muted")))


def render_regression(console, res: harness.HarnessResult) -> None:
    reg = [c for c in res.cases if c.kind == "regression"]
    if not reg:
        return
    bad = [c for c in reg if not c.ok]
    console.print()
    console.print(Text.assemble(("  REGRESSION     ", "label"), (f"{len(reg) - len(bad)} of {len(reg)} cases keep their decision",
                                                                 "ok" if not bad else "text")))
    for c in bad[:10]:
        console.print(Text.assemble(("      ✗ ", "high"), (f"{c.tool}  ", "muted"), (c.input, "text"),
                                    (f"  expected {c.expected}, got {c.decision}", "high")))


def adapt_mapper(fn, tool_field: str):
    """Turn an existing 0.5.1-style mapper into map_call(tool_name, args, context).

    (args)                          LangChain context_mapper; the tool name is injected as tool_field
    (tool_name, args)               plain function
    (tool_name, args, context)      Venom style
    (tool_name, params, metadata, config)  OpenClaw map_context
    """
    import inspect

    try:
        n = len([p for p in inspect.signature(fn).parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty])
    except (TypeError, ValueError):
        n = 3
    if n <= 1:
        def call(tool_name, args, context=None):
            ctx = dict(fn(args))
            ctx.setdefault(tool_field, tool_name)  # like guard_tools(..., tool_field=...)
            ctx.update(context or {})
            return ctx
        return call
    if n == 2:
        return lambda tool_name, args, context=None: fn(tool_name, args)
    if n >= 4:
        from ...plugins.openclaw.config import OpenClawConfig
        cfg = OpenClawConfig()
        return lambda tool_name, args, context=None: fn(tool_name, args, context or {}, cfg)
    return fn


def _free_names(node) -> set:
    """Names a function reads that it does not define itself (arguments, assignments, comprehensions)."""
    import ast

    local = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
    for extra in (node.args.vararg, node.args.kwarg):
        if extra is not None:
            local.add(extra.arg)
    loads = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (local.add if isinstance(n.ctx, ast.Store) else loads.add)(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not node:
            args = n.args
            local |= {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
            if not isinstance(n, ast.Lambda):
                local.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    local.add(t.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                local.add((a.asname or a.name).split(".")[0])
    return loads - local


def isolated_function(source: str, func: str, filename: str):
    """Compile one top-level function from a file without running the rest of the module.

    What the function needs from its file is brought along only when that is safe without
    running the file: other top-level functions it calls (defined, not run), constants whose
    value is a plain literal (strings, numbers, lists, tuples, sets, dicts), and imports of the
    standard library or chimera_core. Anything else it needs (objects built at import time,
    third-party imports, classes) is reported; --import-module loads the whole file instead.
    """
    import ast
    import builtins
    import sys
    import typing

    import warnings

    with warnings.catch_warnings():  # the operator's file, not ours: its warnings are not ours to print
        warnings.simplefilter("ignore")
        tree = ast.parse(source, filename=filename)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if func not in funcs:
        raise SystemExit(f"{filename} has no top-level function {func}")
    literals, imports = {}, {}
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"chimera_core"}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                literals[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, SyntaxError, TypeError):
                pass
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            try:
                literals[n.target.id] = ast.literal_eval(n.value)
            except (ValueError, SyntaxError, TypeError):
                pass
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in stdlib:
                    imports[(a.asname or a.name).split(".")[0]] = n
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split(".")[0] in stdlib:
            for a in n.names:
                if a.name != "*":
                    imports[a.asname or a.name] = n
    namespace = {k: getattr(typing, k) for k in typing.__all__}
    wanted, todo, missing, used_imports = [], [func], set(), set()
    while todo:
        name = todo.pop()
        if name in wanted:
            continue
        wanted.append(name)
        for free in _free_names(funcs[name]):
            if free in funcs:
                todo.append(free)
            elif free in literals:
                namespace[free] = literals[free]
            elif free in imports:
                used_imports.add(free)
            elif free not in namespace and not hasattr(builtins, free):
                missing.add(free)
    if missing:
        raise SystemExit(f"{func} uses {', '.join(sorted(missing))} from the rest of {filename}, which cannot be brought "
                         f"along without running the file. Pass --import-module to load the whole file (its top-level "
                         f"code will run).")
    body = []
    for name in sorted(used_imports):  # only the imports the function uses (stdlib / chimera_core)
        node = imports[name]
        alias = next(a for a in node.names if (a.asname or a.name).split(".")[0] == name)
        body.append(ast.Import(names=[alias]) if isinstance(node, ast.Import)
                    else ast.ImportFrom(module=node.module, names=[alias], level=0))
    for name in wanted:
        node = funcs[name]
        node.decorator_list = []  # decorators would run code from the file
        body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    try:
        exec(compile(module, filename, "exec"), namespace)  # definitions and the imports they use, nothing else
    except ImportError as e:
        raise SystemExit(f"{func} needs {e.name or e} to be importable here: {e}")
    return namespace[func]


def _module_for(ws, spec: MappingSpec, path: Optional[Path], policy_rel: str, func: Optional[str] = None,
                import_module: bool = False) -> Tuple[types.ModuleType, str]:
    if path is not None and str(path) == "openclaw":
        from ...plugins.openclaw import context_mapper as oc
        mod = types.ModuleType("_venom_openclaw_mapping")
        mod.map_call = adapt_mapper(oc.map_context, spec.tool_var or "tool")  # type: ignore[attr-defined]
        return mod, "OpenClaw map_context (built in)"
    if path is not None and path.exists():
        if func and not import_module:
            target = isolated_function(ws.read(path) or "", func, str(path))
            wrapped = types.ModuleType("_venom_adapted_mapping")
            wrapped.map_call = adapt_mapper(target, spec.tool_var or "tool")  # type: ignore[attr-defined]
            return wrapped, f"{ws.rel(path)}:{func} (function only)"
        mod = ws.load_module(path)
        if func:
            target = getattr(mod, func, None)
            if target is None:
                raise SystemExit(f"{path} has no function {func}")
            wrapped = types.ModuleType("_venom_adapted_mapping")
            wrapped.map_call = adapt_mapper(target, spec.tool_var or "tool")  # type: ignore[attr-defined]
            return wrapped, f"{ws.rel(path)}:{func}"
        return mod, ws.rel(path)
    code = codegen.generate(spec, policy_rel)
    mod = types.ModuleType("_venom_generated_mapping")
    exec(compile(code, "<generated mapping>", "exec"), mod.__dict__)  # the generator's own output, not discovered code
    return mod, "generated mapping (not written yet)"


def record(ws, spec: MappingSpec, res: harness.HarnessResult, target: str) -> None:
    state = ws.load_state()
    state.setdefault("mapping_tests", {})[spec.agent_id] = {
        "cases": len(res.cases), "fail_open": len(res.fail_open), "failed": len(res.failed), "mapping": target,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ws.save_state(state)


def parse_mapping_arg(value: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """"openclaw", "path.py" or "path.py:function"."""
    if not value:
        return None, None
    if value == "openclaw":
        return Path("openclaw"), None
    if ":" in value and not value.endswith(".py"):
        path, func = value.rsplit(":", 1)
        return Path(path), func
    return Path(value), None


def cases_path(ws, spec: MappingSpec) -> Path:
    """Regression cases kept with the workspace: every mapping test of the agent runs them."""
    return ws.venom / "cases" / f"{spec.agent_key}.jsonl"


def load_cases(ws, path: Optional[Path]) -> List[Dict[str, Any]]:
    """JSON lines (or one JSON list) of {"tool", "args", "context", "expect": "BLOCK"|"ALLOW", "note"}."""
    import json

    if path is None:
        return []
    text = ws.read(path) if hasattr(ws, "read") else None
    if text is None:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError:
            raise SystemExit(f"cannot read {path}")
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = []
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except ValueError as e:
                raise SystemExit(f"{path}:{n}: not JSON ({e})")
    bad = [i for i, c in enumerate(items, 1) if not isinstance(c, dict) or "tool" not in c]
    if bad:
        raise SystemExit(f"{path}: case {bad[0]} needs at least a \"tool\"")
    return items


def test_mapping(console, ws, spec: MappingSpec, policy_text: str, policy_rel: str, module_path: Optional[Path],
                 func: Optional[str] = None, import_module: bool = False, allowed: Optional[Dict[str, List[str]]] = None,
                 cases_file: Optional[Path] = None) -> harness.HarnessResult:
    mod, target = _module_for(ws, spec, module_path, policy_rel, func, import_module)
    fn = getattr(mod, "map_call", None)
    if fn is None:
        raise SystemExit(f"{target} has no map_call(tool_name, args, context) function")
    cases = load_cases(ws, cases_path(ws, spec) if cases_path(ws, spec).exists() else None) + load_cases(ws, cases_file)
    res = harness.run(spec, fn, compile_guard(policy_text), mod, allowed=allowed, cases=cases)
    render_results(console, res, spec, target, policy_rel)
    record(ws, spec, res, target)
    return res


def keep_cases(ws, spec: MappingSpec, cases_file: Path, console) -> None:
    """Append a case file to the agent's kept regression cases (duplicates skipped)."""
    import json

    target = cases_path(ws, spec)
    have = {json.dumps(c, sort_keys=True) for c in load_cases(ws, target if target.exists() else None)}
    new = [c for c in load_cases(ws, cases_file) if json.dumps(c, sort_keys=True) not in have]
    if not new:
        console.print(f"  [muted]{ws.rel(target)}: nothing new to keep[/muted]")
        return
    old = ws.read(target) or ""
    ws.write_text(target, old + ("" if not old or old.endswith("\n") else "\n")
                  + "".join(json.dumps(c, sort_keys=True) + "\n" for c in new))
    console.print(f"  [ok]kept[/ok] {len(new)} regression case(s) in {ws.rel(target)}: every mapping test of "
                  f"{spec.agent_name} runs them")


def cmd_map(args) -> int:
    console = console_for(args)
    ws = workspace_for(args)
    inv = _load_inventory(args, console)
    agent = find_agent(inv, args.agent)
    if agent is None:
        console.print("[high]which agent?[/high] pass --agent (names from `cslcore venom`)")
        return EXIT_USAGE
    ref = policy_for(ws, inv, agent, args.policy, args)
    if ref is None:
        console.print(f"[high]no policy for {agent.display_name}[/high]: draft one with [brand]cslcore policy new --agent {agent.display_name}[/brand]")
        return EXIT_USAGE
    text = _text(ws, ref, args)
    policy_rel = ws.rel(ref.path)
    spec = build_spec(agent, ref)
    problems = apply_classify(spec, getattr(args, "classify", None) or [])
    if problems:
        for p in problems:
            console.print(f"[high]--classify {p}[/high]")
        return EXIT_USAGE
    allowed = {"scope": list(getattr(args, "allowed_root", None) or []),
               "command": list(getattr(args, "allowed_command", None) or []),
               "destination": list(getattr(args, "allowed_destination", None) or [])}
    cases_file = Path(args.cases) if getattr(args, "cases", None) else None
    user_path, func = parse_mapping_arg(args.mapping)
    path = user_path or mapping_path(ws, agent)

    if args.test:
        res = test_mapping(console, ws, spec, text, policy_rel, path, func, bool(getattr(args, "import_module", False)),
                           allowed, cases_file)
        if cases_file is not None and getattr(args, "keep_cases", False):
            keep_cases(ws, spec, cases_file, console)
        return EXIT_CHECK_FAILED if res.fail_open else EXIT_OK

    console.print(Panel(Text.assemble(("Mapping assistant  ", "brand"), (agent.display_name, "head"), ("  →  ", "muted"), (policy_rel, "text")),
                        box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
    render_spec(console, spec)
    code = codegen.generate(spec, policy_rel)
    console.print(_section("GENERATED"))
    console.print(Text(f"  {ws.rel(path)}", style="muted"))
    from rich.padding import Padding
    console.print(Padding(Syntax(code, "python", theme="ansi_dark", background_color="default", word_wrap=True), (0, 0, 0, 4)))
    console.print()
    res = test_mapping(console, ws, spec, text, policy_rel, None)
    if res.fail_open:
        console.print("  [high]the generated mapping is fail-open; not writing it[/high]")
        return EXIT_CHECK_FAILED
    if ws.plan_only:
        console.print(f"  [muted]--plan-only: would write {ws.rel(path)}[/muted]")
        return EXIT_OK
    existing = ws.read(path)
    if existing is not None and existing != code and not args.mapping:
        console.print(f"  [warn]{ws.rel(path)} exists and differs[/warn]; it will be replaced")
    if confirm(console, f"Write {ws.rel(path)}?", args.yes, default=True):
        ws.write_text(path, code)
        console.print(f"  [ok]written[/ok] {ws.rel(path)}")
    return EXIT_OK
