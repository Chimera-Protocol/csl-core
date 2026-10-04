"""
`cslcore wire`: put the guard in an agent's call path, for real.

Two kinds of agents are wired automatically:

    Claude Code    a PreToolUse hook in the project's .claude/settings.local.json (the user-level
                   assistant: ~/.claude/settings.json). Every tool call, including tools a plugin or
                   MCP server adds later, goes through `cslcore hook` first.
    Python tools   the guard, created once at the top of the file, and a decorator directly above
                   each tool function: `@_csl_guard.tool("name")`. Works under LangChain's @tool,
                   OpenAI Agents' @function_tool, CrewAI's @tool and for plain functions named
                   after the tool.

Anything else (a tool that exists only as a schema, JavaScript, other assistants) gets the exact
snippet instead, and the reason it could not be wired.

Nothing changes until the operator has seen the diff and confirmed it. Every file is copied into
the workspace first (.csl/venom/wire/); `cslcore wire --undo` puts it back. A guard that cannot
load its policy refuses the call (fail closed), so a wired agent never runs unguarded by accident.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .model import Agent
from .workspace import Workspace

MARK = "# cslcore wire"


@dataclass
class Change:
    path: str  # the real file
    shown: str  # the path as the scan saw it
    before: Optional[str]  # None: the file does not exist yet
    after: str
    why: str

    def diff(self, label: Optional[str] = None) -> str:
        a = (self.before or "").splitlines(keepends=True)
        b = self.after.splitlines(keepends=True)
        name = label or self.shown
        return "".join(difflib.unified_diff(a, b, fromfile=name, tofile=name, n=2))


@dataclass
class Plan:
    key: str
    agent: str
    kind: str  # hook | code | done | manual
    changes: List[Change] = field(default_factory=list)
    wrapped: List[str] = field(default_factory=list)  # tools the guard will decide
    missing: List[str] = field(default_factory=list)  # tools it cannot reach automatically
    shared: List[str] = field(default_factory=list)  # tool functions another agent's guard decides
    requires: str = ""  # what the agent's own environment needs for the change to load
    note: str = ""


def cslcore_command() -> str:
    """The cslcore the hook should run: the one running now (next to this Python), by its full
    path, else the first on PATH."""
    import shutil
    import sys
    from pathlib import Path

    here = Path(sys.executable).parent / "cslcore"
    found = str(here) if here.exists() else shutil.which("cslcore")
    return shlex.quote(found) if found else "cslcore"


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_for(agent: Agent, key: str, ws: Workspace, probe, command: Optional[str] = None) -> Plan:
    plan = _plan_for(agent, key, ws, probe, command)
    return _inside_scan(plan, ws)


def _inside_scan(plan: Plan, ws: Workspace) -> Plan:
    """Only files inside the folder the last scan covered are changed (`--root`). A scan record
    that names files elsewhere (a workspace copied from another folder, an old scan) changes
    nothing there."""
    import os

    root = ws.load_state().get("scan_root")
    if not root or not plan.changes:
        return plan
    real_root = os.path.realpath(root)
    outside = [ch for ch in plan.changes
               if os.path.commonpath([os.path.realpath(ch.path), real_root]) != real_root]
    if not outside:
        return plan
    plan.changes = [ch for ch in plan.changes if ch not in outside]
    note = (f"{', '.join(ch.shown for ch in outside)}: outside the scanned folder ({root}), not changed; "
            "scan again (cslcore setup) if the agent moved")
    plan.note = (plan.note + "; " if plan.note else "") + note
    if not plan.changes:
        plan.kind = "manual"
    return plan


def _plan_for(agent: Agent, key: str, ws: Workspace, probe, command: Optional[str] = None) -> Plan:
    product = agent.framework[0] if agent.framework else ""
    if agent.kind == "assistant" and product == "claude-code":
        return _plan_hook(agent, key, ws, probe, command or cslcore_command())
    if agent.kind in ("code", "service"):
        return _plan_code(agent, key, ws, probe)
    return Plan(key, agent.display_name, "manual",
                note=f"{product or agent.kind} has no hook CSL-Core can install; use the snippet in .csl/venom/wiring.md")


def _hook_matchers(settings_text: Optional[str]) -> List[str]:
    """The matchers of the cslcore PreToolUse hooks in a settings file."""
    try:
        data = json.loads(settings_text) if settings_text and settings_text.strip() else {}
    except ValueError:
        return []
    hooks = data.get("hooks") if isinstance(data, dict) else None
    entries = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    out = []
    for entry in entries if isinstance(entries, list) else []:
        for h in entry.get("hooks", []) if isinstance(entry, dict) else []:
            cmd = str(h.get("command", "")) if isinstance(h, dict) else ""
            if "cslcore" in cmd and " hook" in cmd:
                out.append(str(entry.get("matcher", "")))
    return out


EVERY_TOOL = ("*", "", ".*")


def _plan_hook(agent: Agent, key: str, ws: Workspace, probe, command: str) -> Plan:
    plan = Plan(key, agent.display_name, "hook", wrapped=["every tool call"])
    base = agent.project or probe.home()
    narrow = []
    for name in ("settings.json", "settings.local.json"):  # a cslcore hook for every tool: wired already
        matchers = _hook_matchers(ws.read(probe.real_path(f"{base}/.claude/{name}")))
        if any(m in EVERY_TOOL for m in matchers):
            plan.kind, plan.note = "done", f"already wired in {base}/.claude/{name}"
            return plan
        narrow += matchers
    if narrow:
        plan.note = (f"a cslcore hook there covers only {', '.join(narrow)}; the other tools skip it. "
                     "This one covers every tool.")
    if agent.project:
        shown = f"{agent.project}/.claude/settings.local.json"
    else:
        shown = f"{probe.home()}/.claude/settings.json"
    path = probe.real_path(shown)
    before = ws.read(path)
    try:
        data: Dict[str, Any] = json.loads(before) if before and before.strip() else {}
    except ValueError:
        plan.kind, plan.note = "manual", f"{shown} is not valid JSON; it was not touched"
        return plan
    if not isinstance(data, dict):
        plan.kind, plan.note = "manual", f"{shown} is not a settings object; it was not touched"
        return plan
    entries = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    if not isinstance(entries, list):
        plan.kind, plan.note = "manual", f"hooks.PreToolUse in {shown} is not a list; it was not touched"
        return plan
    hook = f"{command} hook --agent {shlex.quote(key)} --workspace {shlex.quote(str(ws.root))}"
    entries.append({"matcher": "*", "hooks": [{"type": "command", "command": hook}]})
    after = json.dumps(data, indent=2) + "\n"
    plan.changes.append(Change(path, shown, before, after, "a PreToolUse hook: every tool call is decided first"))
    return plan


def _defs(tree: ast.AST) -> List[ast.AST]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _tool_name_of(fn) -> List[str]:
    """Names a function is registered under: its own, and any string given to a decorator call."""
    names = [fn.name]
    for d in fn.decorator_list:
        if isinstance(d, ast.Call):
            for a in list(d.args) + [k.value for k in d.keywords if k.arg in ("name", "name_override")]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    names.append(a.value)
    return names


_GUARD_DEF = re.compile(r'(?m)^(_csl_guard\w*) = venom_guard\(("[^"]*")')
_GUARD_USE = re.compile(r"^(_csl_guard\w*)\.tool\(")


def _wrapper(fn) -> Optional[str]:
    """The guard variable a tool function is decorated with (`@_csl_guard.tool(...)`), if any."""
    for d in fn.decorator_list:
        m = _GUARD_USE.match(ast.unparse(d))
        if m:
            return m.group(1)
    return None


FRAMEWORK_MODULES = ("langchain", "langgraph", "agents", "crewai", "llama_index", "semantic_kernel", "autogen",
                     "pydantic_ai", "smolagents", "strands")


def _framework_names(tree: ast.AST) -> set:
    """Names imported from an agent framework in this file (`from langchain_core.tools import tool`,
    `from agents import function_tool as ft`, `import crewai`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and node.module.split(".")[0].startswith(FRAMEWORK_MODULES):
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names
                      if a.name.split(".")[0].startswith(FRAMEWORK_MODULES)}
    return names


def _under_framework(fn, framework_names: set) -> bool:
    """A tool registered by a framework's decorator (LangChain @tool, OpenAI Agents @function_tool,
    CrewAI @tool): a stopped call returns readable text there, so the agent loop goes on. A decorator
    of the file's own (a plain function) keeps raising PermissionError, as in 0.6.8."""
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        root = ast.unparse(target).split(".", 1)[0]
        if root in framework_names and "_csl_guard" not in ast.unparse(d):
            return True
    return False


def _guard_var(source: str, key: str) -> str:
    """The guard variable for this agent in a file: its own if the file has one already; the plain
    `_csl_guard` if no other agent's guard is there; else one named after the agent, so two agents
    whose tools share a file each keep their own policy."""
    others = []
    for var, quoted in _GUARD_DEF.findall(source):
        if json.loads(quoted) == key:
            return var
        others.append(var)
    return "_csl_guard" if "_csl_guard" not in others else "_csl_guard_" + re.sub(r"\W", "_", key)


def _relative(root, path: str) -> str:
    """The workspace as seen from the agent's file, so the line holds in a clone of the repository
    on any machine (venom_guard also reads CSL_WORKSPACE and searches upwards for .csl)."""
    import os

    return os.path.relpath(str(root), os.path.dirname(os.path.realpath(path)))


DEPENDENCY_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile")


def _requirement_note(agent: Agent, probe, ws: Workspace) -> str:
    """The wired code imports chimera_core: the Python environment the agent runs in needs csl-core.
    Said unless its dependency files list it already (and then where to add it, if it has any)."""
    base = agent.project or ""
    listed, files = False, []
    for name in DEPENDENCY_FILES:
        text = ws.read(probe.real_path(f"{base}/{name}")) if base else None
        if text is not None:
            files.append(name)
            listed = listed or "csl-core" in text or "csl_core" in text
    need = "the environment this agent runs in needs csl-core (pip install csl-core); without it the agent stops at import"
    if files and not listed:
        need += f"; add csl-core to its {files[0]}"
    return need if not listed else ""


def _header_line(tree: ast.Module) -> int:
    """After the imports at the top (or the docstring): where the guard is created."""
    at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            at = node.end_lineno or node.lineno
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and at == 0:
            at = node.end_lineno or node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            break
    return at


def _plan_code(agent: Agent, key: str, ws: Workspace, probe) -> Plan:
    plan = Plan(key, agent.display_name, "code")
    files: List[str] = []
    for t in agent.tools:
        for e in t.evidence:
            if e.path.endswith(".py") and e.path not in files:
                files.append(e.path)
    for e in agent.evidence:
        if e.layer == "code" and e.path.endswith(".py") and e.path not in files:
            files.append(e.path)
    wanted = {t.name for t in agent.tools if t.source != "builtin"}
    found: Dict[str, str] = {}
    shared: List[str] = []
    for shown in files:
        path = probe.real_path(shown)
        before = ws.read(path)
        if before is None:
            continue
        try:
            tree = ast.parse(before)
        except SyntaxError:
            continue
        lines = before.splitlines(keepends=True)
        inserts: List[tuple] = []  # (line index, text)
        var = _guard_var(before, key)
        frameworks = _framework_names(tree)
        for fn in _defs(tree):
            names = [n for n in _tool_name_of(fn) if n in wanted and n not in found]
            if not names:
                continue
            found[names[0]] = shown
            owner = _wrapper(fn)
            if owner and owner != var:
                shared.append(names[0])  # another agent's guard decides it already
                continue
            if owner:
                continue
            row = fn.lineno - 1
            indent = lines[row][: len(lines[row]) - len(lines[row].lstrip())]
            framed = ', on_block="return"' if _under_framework(fn, frameworks) else ""
            inserts.append((row, f'{indent}@{var}.tool({json.dumps(names[0])}{framed})  {MARK}\n'))
        if not inserts:
            continue
        if f"{var} = venom_guard(" not in before:
            at = _header_line(tree)
            header = "" if "from chimera_core.venom.observe import venom_guard" in before else \
                f"from chimera_core.venom.observe import venom_guard  {MARK}\n"
            header += f"{var} = venom_guard({json.dumps(key)}, workspace={json.dumps(_relative(ws.root, path))}, near=__file__)  {MARK}\n"
            inserts.append((at, ("\n" if at else "") + header))
        for row, text in sorted(inserts, key=lambda r: r[0], reverse=True):
            lines.insert(row, text)
        after = "".join(lines)
        try:
            ast.parse(after)
        except SyntaxError:
            continue  # never leave an agent that does not parse
        plan.changes.append(Change(path, shown, before, after, "the guard decides each tool call before the function runs"))
    plan.wrapped = sorted(n for n in found if n not in shared)
    plan.missing = sorted(wanted - set(found))
    plan.shared = sorted(shared)
    if not plan.changes:
        if found:
            plan.kind, plan.note = "done", "its tool functions already go through the guard"
        else:
            plan.kind = "manual"
            plan.note = ("its tools exist only as schemas (no function by that name in its code); add "
                         "guard.check(tool, arguments) where your code runs a tool call (see .csl/venom/wiring.md)")
    elif plan.missing:
        plan.note = (f"{', '.join(plan.missing)}: no function by that name in its code; add guard.check where your "
                     "code runs it (see .csl/venom/wiring.md)")
    if plan.changes:
        plan.requires = _requirement_note(agent, probe, ws)
    if shared:
        plan.note = (plan.note + "; " if plan.note else "") + (
            f"{', '.join(shared)}: one function shared with another agent, decided by that agent's policy")
    return plan


# ---------------------------------------------------------------------------
# applying and undoing
# ---------------------------------------------------------------------------

def apply(plan: Plan, ws: Workspace) -> List[Dict[str, Any]]:
    """Write the plan's changes (each one checked against what the diff was made from)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = ws.venom / "wire" / f"{stamp}-{plan.key}"
    done = []
    for ch in plan.changes:
        rec = ws.change_file(ch.path, ws.sha(ch.before), ch.after, backup)
        if rec is not None:
            done.append(rec)
    if done:
        state = ws.load_state()
        wiring = state.setdefault("wiring", {})
        wiring.setdefault(plan.key, {"files": []})
        wiring[plan.key].update({"kind": plan.kind, "at": stamp, "tools": plan.wrapped})
        wiring[plan.key]["files"] = wiring[plan.key]["files"] + done
        ws.save_state(state)
    return done


def apply_many(items, ws: Workspace, probe, on_error=None) -> int:
    """Apply several (agent, plan) pairs in turn. A plan whose file an earlier one in the batch
    changed (two agents' tools in one file) is made again from the file as it is now, so each
    agent gets its own guard there. Returns how many agents were wired."""
    touched: set = set()
    done = 0
    for agent, plan in items:
        if touched & {ch.path for ch in plan.changes}:
            plan = plan_for(agent, plan.key, ws, probe)
        try:
            if apply(plan, ws):
                done += 1
        except (RuntimeError, OSError) as e:
            if on_error is not None:
                on_error(plan, e)
            continue
        touched |= {ch.path for ch in plan.changes}
    return done


def undo(ws: Workspace, key: Optional[str] = None) -> List[tuple]:
    """Put back what `cslcore wire` changed: one agent, or all. Returns (agent, path, result)."""
    state = ws.load_state()
    wiring = state.get("wiring") or {}
    out = []
    for k in [key] if key else list(wiring):
        rec = wiring.get(k)
        if not rec:
            continue
        kept = []
        for f in reversed(rec.get("files", [])):
            result = ws.undo_change(f)
            out.append((k, f["path"], result))
            if result == "skipped":
                kept.append(f)
        if kept:
            rec["files"] = kept
        else:
            wiring.pop(k, None)
    state["wiring"] = wiring
    ws.save_state(state)
    return out


def wired_keys(ws: Workspace) -> List[str]:
    return sorted((ws.load_state().get("wiring") or {}).keys())
