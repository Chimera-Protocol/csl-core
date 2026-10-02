"""
L1 code layer for TypeScript / JavaScript. Never executed: the source is
scanned with a small string- and comment-aware reader, no JS runtime and no parser
dependency.

Recognised tool shapes:
    MCP TS SDK         server.tool("name", "desc", { a: z.string() }, handler)
                       server.registerTool("name", { description, inputSchema: { ... } }, handler)
    Vercel AI SDK      name: tool({ description, parameters | inputSchema: z.object({...}) })
    LangChain.js       new DynamicStructuredTool({ name, description, schema: z.object({...}) })
                       tool(fn, { name, description, schema: z.object({...}) })
    OpenClaw plugins   api.registerTool({ name, description, parameters: Type.Object({...}) })
    function schemas   { type: "function", function: { name, parameters: {...} } } and { name, input_schema }
Parameters come from zod (z.string(), z.number().min().max(), z.enum([...]), z.boolean()),
TypeBox (Type.String(), Type.Number({ minimum, maximum })) or JSON schema literals.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from ..model import Evidence, PromptInfo, Tool, ToolParam
from .code import MODEL_PATTERNS, CodeFile, RouteInfo, _add_tool

JS_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")

FRAMEWORKS = [
    ("@modelcontextprotocol/sdk", "mcp"), ("@langchain/langgraph", "langgraph"), ("@langchain/", "langchain"),
    ("langchain", "langchain"), ("@openai/agents", "openai-agents"), ("@anthropic-ai/claude-agent-sdk", "claude-agent-sdk"),
    ("@anthropic-ai/sdk", "anthropic"), ("openai", "openai"), ("@google/genai", "google-genai"), ("@mastra/", "mastra"),
    ("openclaw", "openclaw"), ("ai", "vercel-ai"),
]
_IMPORT = re.compile(r"""(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\s*\(\s*)['"]([^'"]+)['"]""")
_STRING = re.compile(r"""(['"`])((?:\\.|(?!\1).)*)\1""", re.S)


# ---------------------------------------------------------------------------
# a tiny reader: blank out comments, find matching brackets, split object literals
# ---------------------------------------------------------------------------

def strip_comments(src: str) -> str:
    """Replace comments with spaces (keeping offsets and line numbers), leave strings intact."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


def match_bracket(src: str, start: int) -> int:
    """Index of the bracket closing the one at `start` ((, [ or {); -1 if unbalanced."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [pairs[src[start]]]
    i = start + 1
    while i < len(src):
        c = src[i]
        if c in "'\"`":
            j = i + 1
            while j < len(src) and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1
            continue
        if c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return i
        i += 1
    return -1


def split_top(src: str, sep: str = ",") -> List[str]:
    parts, depth, cur, i = [], 0, [], 0
    while i < len(src):
        c = src[i]
        if c in "'\"`":
            j = i + 1
            while j < len(src) and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            cur.append(src[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def object_entries(obj: str) -> Dict[str, str]:
    """Top-level `key: value` pairs of an object literal (including the braces)."""
    body = obj.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    out: Dict[str, str] = {}
    for part in split_top(body):
        m = re.match(r"""^\s*(?:['"]([^'"]+)['"]|([A-Za-z_$][\w$]*))\s*:\s*(.*)$""", part, re.S)
        if m:
            out[m.group(1) or m.group(2)] = m.group(3).strip()
        else:
            ident = part.strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", ident):
                out[ident] = ident  # shorthand property
    return out


def string_value(expr: str) -> Optional[str]:
    m = _STRING.fullmatch(expr.strip())
    return m.group(2) if m and "${" not in m.group(2) else None


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def _num(expr: str, name: str) -> Optional[float]:
    m = re.search(rf"\.{name}\(\s*(-?\d+(?:\.\d+)?)", expr) or re.search(rf"\b{name}imum\s*:\s*(-?\d+(?:\.\d+)?)", expr)
    return float(m.group(1)) if m else None


def param_from_expr(name: str, expr: str) -> ToolParam:
    e = expr.strip()
    optional = ".optional()" in e or ".default(" in e or ".nullish()" in e or "Type.Optional(" in e
    m = re.match(r"(Type\.Optional|z\.optional|z\.nullable|Type\.Readonly)\s*\(", e)
    if m:  # unwrap the modifier and read the inner type
        close = match_bracket(e, m.end() - 1)
        e = e[m.end(): close].strip() if close > 0 else e
    typ = None
    enum = None
    if re.match(r"(z|Type)\.(string|String)\b", e) or "Type.String" in e:
        typ = "string"
    elif re.match(r"(z\.(number|bigint)|Type\.(Number|Integer))", e):
        typ = "integer" if (".int()" in e or "Type.Integer" in e) else "number"
    elif re.match(r"(z\.boolean|Type\.Boolean)", e):
        typ = "boolean"
    elif re.match(r"z\.(enum|nativeEnum)\(", e) or e.startswith("Type.Union("):
        typ = "string"
        enum = re.findall(r"""['"]([^'"]+)['"]""", e[: match_bracket(e, e.find("(")) + 1 if "(" in e else len(e)])
    elif re.match(r"(z\.array|Type\.Array)", e):
        typ = "array"
    elif re.match(r"(z\.object|Type\.Object)", e):
        typ = "object"
    elif e.startswith("{"):
        entries = object_entries(e)
        t = string_value(entries.get("type", "") or "")
        typ = t
        if "enum" in entries:
            enum = re.findall(r"""['"]([^'"]+)['"]""", entries["enum"])
    lo = _num(e, "min")
    hi = _num(e, "max")
    if "minimum" in e or "maximum" in e:
        ent = object_entries(e[e.find("{"):]) if "{" in e else {}
        lo = float(ent["minimum"]) if re.fullmatch(r"-?\d+(\.\d+)?", ent.get("minimum", "") or "") else lo
        hi = float(ent["maximum"]) if re.fullmatch(r"-?\d+(\.\d+)?", ent.get("maximum", "") or "") else hi
    if typ == "string":
        lo = hi = None  # z.string().min(n) is a length, not a value range
    return ToolParam(name=name, type=typ, required=not optional, enum=enum or None,
                     minimum=int(lo) if lo is not None and lo == int(lo) else lo,
                     maximum=int(hi) if hi is not None and hi == int(hi) else hi)


def params_from_schema_expr(expr: str) -> List[ToolParam]:
    """z.object({...}) / Type.Object({...}) / a raw zod shape {...} / a JSON schema literal."""
    e = expr.strip()
    m = re.match(r"(z\.object|Type\.Object)\s*\(", e)
    if m:
        inner_start = e.find("{", m.end() - 1)
        if inner_start < 0:
            return []
        e = e[inner_start: match_bracket(e, inner_start) + 1]
    if not e.startswith("{"):
        return []
    entries = object_entries(e)
    if "properties" in entries and entries["properties"].startswith("{"):
        req = set(re.findall(r"""['"]([^'"]+)['"]""", entries.get("required", "")))
        out = []
        for k, v in object_entries(entries["properties"]).items():
            p = param_from_expr(k, v)
            p.required = k in req
            out.append(p)
        return out
    return [param_from_expr(k, v) for k, v in entries.items()]


# ---------------------------------------------------------------------------
# analysis of one file
# ---------------------------------------------------------------------------

def _line(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def _args_of(src: str, open_paren: int) -> List[str]:
    close = match_bracket(src, open_paren)
    return split_top(src[open_paren + 1: close]) if close > 0 else []


def analyze_js(path: str, source: str) -> CodeFile:
    cf = CodeFile(path=path)
    src = strip_comments(source)
    for m in _IMPORT.finditer(src):
        mod = m.group(1)
        for prefix, label in FRAMEWORKS:
            if mod == prefix or mod.startswith(prefix if prefix.endswith("/") else prefix + "/") or (prefix != "ai" and mod.startswith(prefix)):
                if label not in cf.frameworks:
                    cf.frameworks.append(label)
                break
    for m in _STRING.finditer(src):
        val = m.group(2)
        if len(val) < 80:
            for rx, _p in MODEL_PATTERNS:
                if rx.match(val) and val not in cf.model_ids:
                    cf.model_ids.append(val)

    def add(name: Optional[str], params: List[ToolParam], desc: Optional[str], pos: int, how: str, source_kind: str = "decorator"):
        if name:
            _add_tool(cf, Tool(name=name, source=source_kind, params=params, description=(desc or None) and desc[:200],
                               evidence=[Evidence("code", path, _line(src, pos), how)]), [])

    # MCP: server.tool("name", ...) and server.registerTool("name", {...})
    for m in re.finditer(r"\.(tool|registerTool)\s*\(", src):
        args = _args_of(src, m.end() - 1)
        if not args:
            continue
        name = string_value(args[0])
        if name is None and m.group(1) == "registerTool" and args[0].startswith("{"):
            ent = object_entries(args[0])  # OpenClaw: api.registerTool({ name, parameters })
            add(string_value(ent.get("name", "") or ""), params_from_schema_expr(ent.get("parameters", "") or ent.get("inputSchema", "") or ""),
                string_value(ent.get("description", "") or ""), m.start(), "registerTool({...})", "plugin_tool")
            continue
        if name is None:
            continue
        desc, params = None, []
        for a in args[1:]:
            if string_value(a) is not None and desc is None:
                desc = string_value(a)
            elif a.startswith("{"):
                ent = object_entries(a)
                if m.group(1) == "registerTool" and ("inputSchema" in ent or "description" in ent):
                    desc = string_value(ent.get("description", "") or "") or desc
                    params = params_from_schema_expr(ent.get("inputSchema", "") or "")
                else:
                    params = params_from_schema_expr(a)
                break
        add(name, params, desc, m.start(), f".{m.group(1)}()", "mcp_tool")

    # Vercel AI SDK: key: tool({...}) / const key = tool({...}); LangChain.js: tool(fn, {...})
    for m in re.finditer(r"(?:([A-Za-z_$][\w$]*)\s*[:=]\s*)?\btool\s*\(", src):
        if src[max(0, m.start() - 1)] == ".":
            continue
        args = _args_of(src, m.end() - 1)
        cfg = next((a for a in args if a.startswith("{")), None)
        if cfg is None:
            continue
        ent = object_entries(cfg)
        name = string_value(ent.get("name", "") or "") or m.group(1)
        schema = ent.get("parameters") or ent.get("inputSchema") or ent.get("schema") or ""
        add(name, params_from_schema_expr(schema), string_value(ent.get("description", "") or ""), m.start(), "tool({...})")

    # LangChain.js classes
    for m in re.finditer(r"new\s+(DynamicStructuredTool|DynamicTool|StructuredTool)\s*\(", src):
        args = _args_of(src, m.end() - 1)
        if args and args[0].startswith("{"):
            ent = object_entries(args[0])
            add(string_value(ent.get("name", "") or ""), params_from_schema_expr(ent.get("schema", "") or ""),
                string_value(ent.get("description", "") or ""), m.start(), f"new {m.group(1)}()", "Tool()")

    # function-calling schemas
    for m in re.finditer(r"""['"]?(parameters|input_schema)['"]?\s*:\s*\{""", src):
        brace = src.find("{", m.end() - 1)
        # find the enclosing object and its name
        depth, i = 0, m.start()
        while i > 0:
            i -= 1
            if src[i] == "}":
                depth += 1
            elif src[i] == "{":
                if depth == 0:
                    break
                depth -= 1
        close = match_bracket(src, i) if src[i:i + 1] == "{" else -1
        if close < 0:
            continue
        ent = object_entries(src[i: close + 1])
        name = string_value(ent.get("name", "") or "")
        if name and re.fullmatch(r"[A-Za-z_][\w\-.]{0,63}", name):
            add(name, params_from_schema_expr(src[brace: match_bracket(src, brace) + 1]),
                string_value(ent.get("description", "") or ""), m.start(), "function-calling schema", "function_schema")

    # routes, env, prompts, entry points
    for m in re.finditer(r"""\b(?:app|router|server|hono)\.(get|post|put|patch|delete|all)\s*\(\s*['"](/[^'"]*)['"]""", src):
        cf.routes.append(RouteInfo(m.group(2), m.group(1), _line(src, m.start())))
    for m in re.finditer(r"""process\.env(?:\.([A-Z_][A-Z0-9_]*)|\[\s*['"]([A-Z_][A-Z0-9_]*)['"]\s*\])""", src):
        n = m.group(1) or m.group(2)
        if n not in cf.env_names:
            cf.env_names.append(n)
    for m in re.finditer(r"""\b(system|systemPrompt|system_prompt|instructions)\s*:\s*(['"`])((?:\\.|(?!\2).){20,})\2""", src, re.S):
        text = m.group(3)
        if not cf.prompt.present or (cf.prompt.length or 0) < len(text):
            cf.prompt = PromptInfo(True, len(text), hashlib.sha256(text.encode()).hexdigest()[:16])
    cf.has_main = bool(re.search(r"require\.main\s*===\s*module|\.connect\s*\(\s*(new\s+)?\w*Transport|\.listen\s*\(|process\.argv", src))
    return cf


def is_js(path: str) -> bool:
    return path.endswith(JS_SUFFIXES) and not path.endswith((".d.ts", ".min.js"))
