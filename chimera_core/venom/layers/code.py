"""
L1 code layer: static analysis of Python agents.

Source is parsed with `ast`, never imported or executed. The unit of discovery is a
project folder; every agent-like file in it contributes tools, frameworks and evidence.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from ..model import Evidence, PromptInfo, Tool, ToolParam

FRAMEWORK_IMPORTS = [
    # (module prefix, label); more specific first
    ("langgraph", "langgraph"),
    ("langchain", "langchain"),
    ("langchain_core", "langchain"),
    ("crewai", "crewai"),
    ("llama_index", "llama_index"),
    ("autogen", "autogen"),
    ("agents", "openai-agents"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("claude_agent_sdk", "claude-agent-sdk"),
    ("google.genai", "google-genai"),
    ("google.generativeai", "google-genai"),
    ("mcp", "mcp"),
    ("fastmcp", "mcp"),
    ("chimera_core", "csl-core"),
]

MODEL_PATTERNS = [
    (re.compile(r"^claude-[\w.\-]+$"), "anthropic"),
    (re.compile(r"^(gpt-[\w.\-]+|o[1-9](-[\w.\-]+)?)$"), "openai"),
    (re.compile(r"^gemini-[\w.\-]+$"), "google"),
    (re.compile(r"^mistral-[\w.\-]+$"), "mistral"),
    (re.compile(r"^command-[\w.\-]+$"), "cohere"),
    (re.compile(r"^llama-?[\w.\-]+$", re.I), "meta"),
    (re.compile(r"^deepseek-[\w.\-]+$"), "deepseek"),
]

GUARD_CALLS = {
    "ChimeraGuard": "wrapper", "load_guard": "wrapper", "create_guard_from_string": "wrapper",
    "guard_tools": "wrapper", "wrap_tool": "wrapper", "wrap_tools": "wrapper", "gate": "wrapper",
    "OpenClawGuard": "plugin", "ChimeraRunnableGate": "wrapper", "venom_guard": "wrapper", "guarded_verify": "wrapper",
}

ROUTE_METHODS = {"get", "post", "put", "patch", "delete", "route", "api_route", "websocket"}
PREFILTER = re.compile(
    r"langchain|langgraph|crewai|llama_index|autogen|openai|anthropic|genai|mcp|tool|chimera|claude|"
    r"gpt-|gemini|fastapi|flask|route|agents",
)


@dataclass
class RouteInfo:
    path: str
    handler: str
    line: int


@dataclass
class CodeFile:
    path: str
    frameworks: List[str] = field(default_factory=list)
    model_ids: List[str] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)
    graph_nodes: List[str] = field(default_factory=list)
    routes: List[RouteInfo] = field(default_factory=list)
    guard_calls: List[Tuple[str, int, Optional[str]]] = field(default_factory=list)  # (call, line, policy path)
    env_names: List[str] = field(default_factory=list)
    prompt: PromptInfo = field(default_factory=PromptInfo)
    has_main: bool = False
    tool_calls: Dict[str, List[str]] = field(default_factory=dict)  # tool name -> call names in its body

    @property
    def agent_like(self) -> bool:
        agentic = [f for f in self.frameworks if f not in ("csl-core", "mcp")]
        return bool(agentic or self.tools or self.graph_nodes)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _name(node: ast.AST) -> str:
    """Dotted name of a Name / Attribute / Call target, '' when not a plain name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _last(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _kw(call: ast.Call, name: str) -> Optional[ast.AST]:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _annotation(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _params_from_def(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> List[ToolParam]:
    out: List[ToolParam] = []
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for a, d in zip(positional, defaults):
        if a.arg in ("self", "cls", "ctx", "context", "run_manager", "config"):
            continue
        out.append(_param(a.arg, a.annotation, d is not None))
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        out.append(_param(a.arg, a.annotation, d is not None))
    return out


def _param(name: str, ann: Optional[ast.AST], has_default: bool) -> ToolParam:
    t = _annotation(ann)
    enum = None
    if isinstance(ann, ast.Subscript) and _last(_name(ann.value)) == "Literal":
        v = _literal(ann.slice)
        if v is not None:
            enum = [str(x) for x in (v if isinstance(v, tuple) else (v,))]
        t = "str"
    return ToolParam(name=name, type=t, required=not has_default, enum=enum)


def _params_from_schema(schema: Any) -> List[ToolParam]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out = []
    for pname, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        enum = spec.get("enum")
        out.append(ToolParam(
            name=str(pname),
            type=spec.get("type") if isinstance(spec.get("type"), str) else None,
            required=pname in required,
            enum=[str(x) for x in enum] if isinstance(enum, list) else None,
            minimum=spec.get("minimum"),
            maximum=spec.get("maximum"),
        ))
    return out


def _docstring(fn: ast.AST) -> Optional[str]:
    try:
        doc = ast.get_docstring(fn)  # type: ignore[arg-type]
    except TypeError:
        return None
    return doc.strip().splitlines()[0][:200] if doc else None


def _body_calls(fn: ast.AST) -> List[str]:
    calls = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            nm = _name(n.func)
            if nm:
                calls.add(nm)
    return sorted(calls)


# ---------------------------------------------------------------------------
# analysis of one file
# ---------------------------------------------------------------------------

TOOL_DECORATORS = {"tool", "function_tool", "kernel_function", "ai_function"}
MCP_DECORATOR_OWNERS = {"mcp", "server", "app", "srv", "fastmcp"}


def analyze_source(path: str, source: str) -> CodeFile:
    """Raises SyntaxError for files that do not parse (the caller counts them)."""
    tree = ast.parse(source, filename=path)
    cf = CodeFile(path=path)
    frameworks: List[str] = []
    functions: Dict[str, ast.AST] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, node)

    for node in ast.walk(tree):
        # imports -> frameworks
        mods: List[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods = [node.module]
        for m in mods:
            for prefix, label in FRAMEWORK_IMPORTS:
                if m == prefix or m.startswith(prefix + "."):
                    if label not in frameworks:
                        frameworks.append(label)
                    break

        # model ids and prompts from string constants
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
            for rx, _prov in MODEL_PATTERNS:
                if rx.match(node.value) and node.value not in cf.model_ids:
                    cf.model_ids.append(node.value)

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                tn = _name(t).lower()
                if any(k in tn for k in ("system_prompt", "system_message", "instructions")):
                    _note_prompt(cf, node.value)

        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            if _name(node.test.left) == "__name__":
                cf.has_main = True

        # class-based tools: class TransferFunds(BaseTool): name = "transfer_funds" ... def _run(self, ...)
        if isinstance(node, ast.ClassDef) and any(_last(_name(b)) in ("BaseTool", "StructuredTool", "Tool") for b in node.bases):
            tname, desc, run = None, None, None
            for item in node.body:
                target = item.targets[0] if isinstance(item, ast.Assign) and item.targets else (item.target if isinstance(item, ast.AnnAssign) else None)
                if target is not None and _name(target) in ("name", "description"):
                    val = _const_str(item.value)  # type: ignore[union-attr]
                    if _name(target) == "name":
                        tname = val
                    else:
                        desc = val
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in ("_run", "_arun") and run is None:
                    run = item
            if tname:
                _add_tool(cf, Tool(
                    name=tname, source="Tool()", params=_params_from_def(run) if run else [],
                    description=(desc or "")[:200] or None, evidence=[Evidence("code", path, node.lineno, f"class {node.name}")],
                ), _body_calls(run) if run else [])

        # decorated functions: tools and routes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dname = _name(dec)
                last = _last(dname)
                owner = dname.rsplit(".", 1)[0].lower() if "." in dname else ""
                if last in TOOL_DECORATORS and (not owner or owner in MCP_DECORATOR_OWNERS or owner.startswith("lang") or owner in ("tools", "crewai", "agents")):
                    explicit = _const_str(dec.args[0]) if isinstance(dec, ast.Call) and dec.args else None
                    explicit = explicit or (_const_str(_kw(dec, "name")) if isinstance(dec, ast.Call) else None)
                    source = "mcp_tool" if owner in MCP_DECORATOR_OWNERS else "decorator"
                    _add_tool(cf, Tool(
                        name=explicit or node.name, source=source, params=_params_from_def(node),
                        description=_docstring(node),
                        evidence=[Evidence("code", path, node.lineno, f"@{dname}")],
                    ), _body_calls(node))
                if last in ROUTE_METHODS and isinstance(dec, ast.Call) and dec.args:
                    rpath = _const_str(dec.args[0])
                    if rpath and rpath.startswith("/"):
                        cf.routes.append(RouteInfo(rpath, node.name, node.lineno))

        if isinstance(node, ast.Call):
            fname = _name(node.func)
            last = _last(fname)
            # Tool(name=..., func=...) and StructuredTool.from_function(...)
            if last in ("Tool", "StructuredTool", "FunctionTool") or fname.endswith("from_function") or fname.endswith("from_defaults"):
                tname = _const_str(_kw(node, "name"))
                func = _kw(node, "func") or _kw(node, "fn") or (node.args[0] if node.args else None)
                fn_node = functions.get(_name(func)) if func is not None else None
                if not tname and fn_node is not None:
                    tname = fn_node.name  # type: ignore[attr-defined]
                if tname:
                    params = _params_from_def(fn_node) if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
                    desc = _const_str(_kw(node, "description")) or (_docstring(fn_node) if fn_node else None)
                    _add_tool(cf, Tool(
                        name=tname, source="Tool()", params=params, description=(desc or None) and desc[:200],
                        evidence=[Evidence("code", path, node.lineno, fname)],
                    ), _body_calls(fn_node) if fn_node is not None else [])
            # server.tool("name")(fn) style or mcp.add_tool(fn)
            if last == "add_tool" and node.args:
                fn_node = functions.get(_name(node.args[0]))
                tname = _const_str(_kw(node, "name")) or (fn_node.name if fn_node else _name(node.args[0]))  # type: ignore[attr-defined]
                if tname:
                    _add_tool(cf, Tool(
                        name=tname, source="mcp_tool",
                        params=_params_from_def(fn_node) if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [],
                        description=_docstring(fn_node) if fn_node else None,
                        evidence=[Evidence("code", path, node.lineno, fname)],
                    ), _body_calls(fn_node) if fn_node is not None else [])
            # LangGraph nodes
            if last == "add_node" and node.args:
                nname = _const_str(node.args[0]) or _name(node.args[0])
                if nname and nname not in cf.graph_nodes:
                    cf.graph_nodes.append(nname)
            # guard call sites (observe() counts only in files that use chimera_core)
            if last == "observe" and fname.endswith("observe") and node.args:
                cf.guard_calls.append(("observe?", node.lineno, None))
            if last in GUARD_CALLS:
                policy = None
                for a in list(node.args) + [k.value for k in node.keywords]:
                    s = _const_str(a)
                    if s and s.endswith(".csl"):
                        policy = s
                cf.guard_calls.append((last, node.lineno, policy))
            # credentials by name
            if fname in ("os.getenv", "os.environ.get", "getenv", "environ.get") and node.args:
                s = _const_str(node.args[0])
                if s and s not in cf.env_names:
                    cf.env_names.append(s)
            # system prompts passed as keywords
            for k in ("system", "system_prompt", "instructions", "system_message"):
                v = _kw(node, k)
                if v is not None:
                    _note_prompt(cf, v)

        if isinstance(node, ast.Subscript) and _name(node.value) in ("os.environ", "environ"):
            s = _const_str(node.slice)
            if s and s not in cf.env_names:
                cf.env_names.append(s)

        # function-calling schemas (OpenAI / Anthropic dicts)
        if isinstance(node, ast.Dict):
            _maybe_schema_tool(cf, node, path)

        # {"role": "system", "content": "..."}
        if isinstance(node, ast.Dict):
            keys = [_const_str(k) for k in node.keys if k is not None]
            if "role" in keys and "content" in keys:
                d = dict(zip([_const_str(k) if k is not None else None for k in node.keys], node.values))
                if _const_str(d.get("role")) == "system":
                    _note_prompt(cf, d.get("content"))

    cf.frameworks = frameworks
    observed = [g for g in cf.guard_calls if g[0] == "observe?"]
    cf.guard_calls = [g for g in cf.guard_calls if g[0] != "observe?"]
    if observed and "csl-core" in frameworks:
        cf.guard_calls += [("observe", line, None) for _, line, _ in observed]
    return cf


def _note_prompt(cf: CodeFile, value: Optional[ast.AST]) -> None:
    text = _const_str(value)
    if not text or len(text) < 20:
        return
    if cf.prompt.present and (cf.prompt.length or 0) >= len(text):
        return
    cf.prompt = PromptInfo(present=True, length=len(text), sha256=hashlib.sha256(text.encode()).hexdigest()[:16])


def _maybe_schema_tool(cf: CodeFile, node: ast.Dict, path: str) -> None:
    keys = {_const_str(k): v for k, v in zip(node.keys, node.values) if k is not None}
    target = keys
    if _const_str(keys.get("type")) == "function" and isinstance(keys.get("function"), ast.Dict):
        fn = keys["function"]
        target = {_const_str(k): v for k, v in zip(fn.keys, fn.values) if k is not None}  # type: ignore[union-attr]
    name = _const_str(target.get("name"))
    schema_node = target.get("parameters") or target.get("input_schema") or target.get("inputSchema")
    if not name or schema_node is None or not re.fullmatch(r"[A-Za-z_][\w\-.]{0,63}", name):
        return
    schema = _literal(schema_node)
    _add_tool(cf, Tool(
        name=name, source="function_schema", params=_params_from_schema(schema),
        description=(_const_str(target.get("description")) or "")[:200] or None,
        evidence=[Evidence("code", path, node.lineno, "function-calling schema")],
    ), [])


_TOOL_NAME = re.compile(r"[A-Za-z_][\w\-. ]{0,63}")


def _add_tool(cf: CodeFile, tool: Tool, body_calls: List[str]) -> None:
    if not _TOOL_NAME.fullmatch(tool.name):
        return
    for existing in cf.tools:
        if existing.name == tool.name:
            if not existing.params and tool.params:
                existing.params = tool.params
            return
    cf.tools.append(tool)
    cf.tool_calls[tool.name] = body_calls


# ---------------------------------------------------------------------------
# project grouping
# ---------------------------------------------------------------------------

PROJECT_MARKERS = ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", ".git", "Pipfile", "package.json")


def project_of(probe, path: str, roots: List[str]) -> str:
    """Nearest ancestor with a project marker, bounded by the scan root; else the file's folder."""
    p = PurePosixPath(path).parent
    root = next((r for r in sorted(roots, key=len, reverse=True) if path.startswith(r.rstrip("/") + "/")), None)
    cur = p
    while True:
        if any(probe.exists(str(cur / m)) for m in PROJECT_MARKERS):
            return str(cur)
        if root is None or str(cur) == root.rstrip("/") or cur.parent == cur:
            break
        cur = cur.parent
    return str(p)


@dataclass
class CodeScan:
    files: List[CodeFile]
    files_scanned: int
    parse_errors: int
    error_paths: List[str]


def scan_code(probe, roots: List[str], budget) -> CodeScan:
    files: List[CodeFile] = []
    scanned = 0
    errors = 0
    error_paths: List[str] = []
    seen = set()
    from .code_js import JS_SUFFIXES, analyze_js, is_js

    for root in roots:
        for path in probe.walk(root, suffixes=(".py",) + JS_SUFFIXES):
            if path in seen:
                continue
            seen.add(path)
            if budget.exceeded():
                budget.partial = True
                break
            scanned += 1
            text = probe.read_text(path)
            if text is None or not PREFILTER.search(text):
                continue
            if path.endswith(JS_SUFFIXES) and not is_js(path):
                continue
            try:
                cf = analyze_js(path, text) if is_js(path) else analyze_source(path, text)
            except (SyntaxError, ValueError, RecursionError, IndexError):
                errors += 1
                error_paths.append(path)
                continue
            if cf.agent_like or cf.guard_calls or cf.routes:
                files.append(cf)
    return CodeScan(files, scanned, errors, error_paths)
