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

    def diff(self) -> str:
        a = (self.before or "").splitlines(keepends=True)
        b = self.after.splitlines(keepends=True)
        return "".join(difflib.unified_diff(a, b, fromfile=self.shown, tofile=self.shown, n=2))


@dataclass
class Plan:
    key: str
    agent: str
    kind: str  # hook | code | done | manual
    changes: List[Change] = field(default_factory=list)
    wrapped: List[str] = field(default_factory=list)  # tools the guard will decide
    missing: List[str] = field(default_factory=list)  # tools it cannot reach automatically
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


def _wired(fn) -> bool:
    return any("_csl_guard" in ast.unparse(d) for d in fn.decorator_list)


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
        for fn in _defs(tree):
            names = [n for n in _tool_name_of(fn) if n in wanted and n not in found]
            if not names:
                continue
            found[names[0]] = shown
            if _wired(fn):
                continue
            row = fn.lineno - 1
            indent = lines[row][: len(lines[row]) - len(lines[row].lstrip())]
            inserts.append((row, f'{indent}@_csl_guard.tool({json.dumps(names[0])})  {MARK}\n'))
        if not inserts:
            continue
        if "_csl_guard = venom_guard(" not in before:
            at = _header_line(tree)
            header = (f"from chimera_core.venom.observe import venom_guard  {MARK}\n"
                      f"_csl_guard = venom_guard({json.dumps(key)}, workspace={json.dumps(str(ws.root))})  {MARK}\n")
            inserts.append((at, ("\n" if at else "") + header))
        for row, text in sorted(inserts, key=lambda r: r[0], reverse=True):
            lines.insert(row, text)
        after = "".join(lines)
        try:
            ast.parse(after)
        except SyntaxError:
            continue  # never leave an agent that does not parse
        plan.changes.append(Change(path, shown, before, after, "the guard decides each tool call before the function runs"))
    plan.wrapped = sorted(n for n in found)
    plan.missing = sorted(wanted - set(found))
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
