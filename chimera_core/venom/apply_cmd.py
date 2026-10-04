"""
`cslcore apply`: the limits a repository keeps in csl-limits.ini, made into active policies.

    cslcore apply --init        write csl-limits.ini from the limits each agent has now
    cslcore apply               scan the repository, make each agent's policy from the file, check it
                                (Z3, the mapping test), show the diff, activate it, run the check
    cslcore apply --check       for CI: change nothing. The policies the file makes pass the gate, their
                                mappings have no fail-open case and their sample calls are decided as
                                the limits say (else exit 3); where policies are active, exit 1 when
                                one is not what the file says

The file is the place a team reviews changes to what its agents may do, in a pull request:

    ; What each agent in this repository may do. Apply with: cslcore apply
    [payments]                        ; an agent, as cslcore limits names it
    profile = standard                ; standard | strict
    mode = block                      ; block stops what the limits do not allow; log only records it
    transfer_funds = 1k..5k           ; money: free up to 1k, with approval up to 5k, never above
    export_rows.limit = ..1000        ; any number a tool takes (a list: its length)
    delete_customer = block           ; allow | approval | block | standard
    scope = data, reports             ; folders it may write under, from this file's folder
    commands = git status, git log *  ; strict profile: the only commands it runs
    destinations = api.example.com    ; only these destinations
    extra_tools = wire_transfer:spend:amount   ; tools the scan does not see (NAME:KIND[:AMOUNT_PARAM])

An agent the file does not name keeps the limits it has. Each section starts from the standard
limits, so the file says everything that is not standard. Without `mode` an agent keeps the mode it
has; one activated for the first time starts in block.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.text import Text

from .commands import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE, console_for, workspace_for
from .model import Agent
from .policy import limits as L
from .render.words import n as _n

FILE = "csl-limits.ini"
EXIT_DIFFERS = 1
SPECIAL = {"profile", "mode", "scope", "commands", "destinations", "extra_tools"}
DECISIONS = {"allow", "approval", "block", "standard"}


class LimitsFileError(ValueError):
    pass


def find_file(start: Optional[str] = None) -> Optional[Path]:
    """csl-limits.ini in this folder or the nearest one above it, up to the repository root."""
    cur = Path(start or os.getcwd()).resolve()
    while True:
        if (cur / FILE).is_file():
            return cur / FILE
        if (cur / ".git").exists() or cur.parent == cur:
            return None
        cur = cur.parent


def repo_root(start: Optional[str] = None) -> Path:
    """The repository this folder is in (the nearest folder with .git), else this folder."""
    here = Path(start or os.getcwd()).resolve()
    return next((p for p in [here, *here.parents] if (p / ".git").exists()), here)


def parse(text: str) -> Dict[str, Dict[str, str]]:
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";", "#"), comment_prefixes=(";", "#"),
                                   delimiters=("=",), strict=True)
    cp.optionxform = str  # tool names keep their case
    try:
        cp.read_string(text)
    except configparser.Error as e:
        raise LimitsFileError(f"{FILE}: {str(e).splitlines()[0]}")
    return {section: dict(cp.items(section)) for section in cp.sections()}


def _items(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def limits_from(section: Dict[str, str], agent: Agent, key: str, scope: List[str], base: Path) -> L.Limits:
    """The limits a section says, on top of the standard ones."""
    lim = L.defaults(agent, scope)
    mode = section.get("mode", "").strip()
    if mode and mode not in ("block", "log"):
        raise LimitsFileError(f"[{key}] mode: block or log, not {mode!r}")
    profile = section.get("profile", "standard").strip()
    if profile not in ("standard", "strict"):
        raise LimitsFileError(f"[{key}] profile: standard or strict, not {profile!r}")
    adds = [f"{key}:{e}" for e in _items(section.get("extra_tools", ""))]
    L.apply_flags(lim, key, (), adds, profile)
    lim = L.defaults(agent, scope, lim)
    if "scope" in section:
        lim.scope = [str((base / s).resolve()) if not os.path.isabs(s) else s for s in _items(section["scope"])]
    lim.commands = _items(section.get("commands", ""))
    lim.destinations = _items(section.get("destinations", ""))
    for name, value in section.items():
        if name in SPECIAL:
            continue
        value = value.strip()
        tool = name.split(".", 1)[0]
        if tool not in lim.tools:
            known = ", ".join(sorted(lim.tools)) or "none"
            raise LimitsFileError(f"[{key}] {name}: {key} has no tool {tool!r} (its tools: {known})")
        if value in DECISIONS:
            if "." in name:
                raise LimitsFileError(f"[{key}] {name}: a decision is for a whole tool ({tool} = {value})")
            lim.tools[tool].decide = None if value == "standard" else value
        else:
            L.apply_flags(lim, key, [f"{key}.{name}={value}"])
    return lim


def render(rows: List[Tuple[Agent, str, L.Limits]], root: Path, modes: Optional[Dict[str, str]] = None) -> str:
    """A csl-limits.ini that says what each agent's limits are now (what is not standard is written
    as a setting, the rest as a comment)."""
    out = ["; What each agent in this repository may do. Apply with: cslcore apply",
           "; Review changes in pull requests; in CI, cslcore apply --check fails when the active",
           "; policies are not what this file says.", ""]
    modes = modes or {}
    for agent, key, lim in rows:
        std = L.defaults(agent, lim.scope)
        out.append(f"[{key}]")
        if lim.profile != "standard":
            out.append(f"profile = {lim.profile}")
        out.append(f"mode = {modes.get(key, 'log')}   ; block stops what the limits do not allow; log only records")
        rel = [os.path.relpath(s, root) if s.startswith(str(root)) else s for s in lim.scope]
        out.append(f"scope = {', '.join(rel) or '.'}" if lim.scope != std.scope else f"; scope = {', '.join(rel) or '.'}")
        if lim.commands:
            out.append(f"commands = {', '.join(lim.commands)}")
        if lim.destinations:
            out.append(f"destinations = {', '.join(lim.destinations)}")
        if lim.extra_tools:
            out.append("extra_tools = " + ", ".join(
                ":".join(filter(None, [e["name"], _kind_word(e["risk"]), e.get("amount_param")])) for e in lim.extra_tools))
        described = {name: what for name, _kind, what in L.describe(lim)}
        for name, tl in sorted(lim.tools.items()):
            s = std.tools.get(name)
            if tl.decide:
                out.append(f"{name} = {tl.decide}")
            if tl.kind == "spend" and tl.amount_param and tl.decide != "block":
                line = f"{name} = {_num(tl.allow_up_to)}..{_num(tl.never_above)}"
                same = s is not None and (s.allow_up_to, s.never_above) == (tl.allow_up_to, tl.never_above)
                out.append(("; " if same else "") + line + ("   ; standard" if same else ""))
            for param, (lo, hi) in sorted((tl.numbers or {}).items()):
                out.append(f"{name}.{param} = {_num(lo)}..{_num(hi)}")
            if not tl.decide and not (tl.kind == "spend" and tl.amount_param) and not tl.numbers:
                out.append(f"; {name}: {described.get(name, tl.kind)}")
        out.append("")
    return "\n".join(out)


def _kind_word(risk: str) -> str:
    return {"SPEND": "spend", "EXEC": "shell", "WRITE": "write", "READ": "read", "EXTERNAL": "send",
            "DESTRUCTIVE": "destroy", "IDENTITY": "identity"}.get(risk, "other")


def _num(n: Optional[int]) -> str:
    n = int(n or 0)
    for div, suffix in ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k")):
        if n >= div and n % div == 0:
            return f"{n // div}{suffix}"
    return str(n)


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def _agents(args, console, ws, root: Path) -> List[Agent]:
    """The repository's agents, from a fresh scan of it."""
    from argparse import Namespace

    from .commands import remember_root, run_scan, save_report

    scan_args = Namespace(**{**vars(args), "root": str(root), "no_anim": True})
    inv = run_scan(scan_args, console, live=False).inventory
    if not ws.plan_only:
        save_report(ws, inv)
        remember_root(ws, scan_args)
    return [a for a in inv.agents if a.tools]


def _match(agents: List[Agent], name: str) -> Optional[Agent]:
    from .policy.draft import agent_key

    return next((a for a in agents if name in (agent_key(a), a.display_name, a.id)), None)


def cmd_apply(args) -> int:
    from .limits_cmd import apply_limits, show_check
    from .policy.draft import agent_key
    from .policy.gate import verify_text

    console = console_for(args)
    ws = workspace_for(args)
    given = getattr(args, "file", None)
    path = Path(given).resolve() if given else find_file()
    if getattr(args, "init", False):
        return _init(args, console, ws, path or repo_root() / FILE)
    if path is None or not path.is_file():
        console.print(f"  [warn]no {FILE} here or above (up to the repository root)[/warn]: "
                      "cslcore apply --init writes one from the limits your agents have now")
        return EXIT_USAGE
    root = path.parent
    try:
        sections = parse(path.read_text(encoding="utf-8"))
    except LimitsFileError as e:
        console.print(f"  [high]{e}[/high]")
        return EXIT_USAGE
    agents = _agents(args, console, ws, Path(args.root).resolve() if getattr(args, "root", None) else root)
    checking = bool(getattr(args, "check", False))
    plan: List[Tuple[Agent, str, L.Limits, str]] = []
    try:
        for name, section in sections.items():
            agent = _match(agents, name)
            if agent is None:
                known = ", ".join(sorted(agent_key(a) for a in agents if a.tools)) or "none"
                raise LimitsFileError(f"[{name}]: no agent by that name (the scanned agents: {known})")
            from .board import scope_of
            key = agent_key(agent)
            lim = limits_from(section, agent, key, scope_of(args, ws, agent), root)
            text, _notes = L.policy_text(agent, lim)
            gate = verify_text(text)
            if not gate.ok:
                issue = gate.issues[0].message if gate.issues else gate.stage
                raise LimitsFileError(f"[{name}]: the policy for these limits does not pass the check: {issue}")
            plan.append((agent, key, lim, text))
    except (LimitsFileError, L.LimitError) as e:
        console.print(f"  [high]{e}[/high]")
        return EXIT_USAGE
    console.print(Text.assemble(("  ", ""), (str(path), "head"),
                                (f"   {len(plan)} agent{'s' if len(plan) != 1 else ''}", "muted")))
    differs, failed = [], []
    for agent, key, lim, text in plan:
        active = ws.read(ws.policies / f"{key}.csl")
        same = active == text
        console.print()
        if checking:
            if not _check_file(console, agent, key, lim, text):
                failed.append(key)
            line = Text.assemble(("    ", ""))
            if active is None:
                line.append("not active in this workspace", style="muted")
            elif same:
                line.append("the active policy is what the file says", style="ok")
            else:
                line.append("the active policy is not what the file says: run cslcore apply", style="high")
                differs.append(key)
            console.print(line)
            continue
        elif not same:
            if not apply_limits(console, ws, args, agent, lim, bool(getattr(args, "yes", False))):
                differs.append(key)
                continue
        else:
            L.save(ws, lim)
            console.print(Text.assemble(("  ", ""), (key, "head"), ("  already active", "muted")))
        if not checking:
            _mode(console, ws, key, sections[_section_of(sections, agent, key)].get("mode", "").strip())
            if getattr(args, "wire", False):
                _wire(args, console, ws, agent)
        if not show_check(console, ws, agent, lim, compact=same):
            failed.append(key)
    unnamed = sorted(agent_key(a) for a in agents if a.tools and not any(a is p[0] for p in plan))
    if unnamed:
        console.print()
        console.print(Text(f"  not in {FILE} (they keep the limits they have): {', '.join(unnamed)}", style="muted"))
    if failed:
        return EXIT_CHECK_FAILED
    if differs:
        return EXIT_DIFFERS
    return EXIT_OK


def _check_file(console, agent: Agent, key: str, lim: L.Limits, text: str) -> bool:
    """For CI: the policy the file makes passes the gate (done before), its generated mapping has no
    fail-open case, and its sample calls are decided as the limits say. Needs nothing active."""
    from . import check

    report, fail_open = check.run_text(agent, lim, text)
    if fail_open:
        console.print(Text.assemble(("  ✗ ", "high"), (key, "head"),
                                    (f"   its mapping would be fail-open in {_n(fail_open, 'case')}", "high")))
        return False
    check.show(console, report, key, compact=True)
    return report.ok


def _section_of(sections: Dict[str, Dict[str, str]], agent: Agent, key: str) -> str:
    return next(n for n in sections if n in (key, agent.display_name, agent.id))


def _mode(console, ws, key: str, wanted: str) -> None:
    """The mode the file says, else the one the agent has (a first activation started it in block);
    said as it is."""
    from .controls import mode_on_activation

    mode = mode_on_activation(ws, key, False, wanted or None)
    console.print(Text.assemble(("  ", ""), (key, "head"), ("  mode ", "muted"),
                                ("block: stops what its limits do not allow", "ok") if mode == "block"
                                else ("log: records what its limits do not allow, stops nothing", "warn")))


def _wire(args, console, ws, agent: Agent) -> None:
    from . import board as B
    from . import wiring
    from .policy.workbench import confirm
    from .wire_cmd import show_plan

    plan = B.wire_plan(args, ws, agent)
    if plan.kind == "done" or not plan.changes:
        if plan.kind == "manual":
            console.print(Text("  " + plan.note, style="warn"))
        return
    show_plan(console, plan)
    if confirm(console, f"Wire {agent.display_name}?", bool(getattr(args, "yes", False)), default=True):
        try:
            wiring.apply(plan, ws)
        except (RuntimeError, OSError) as e:
            console.print(f"  [high]not wired: {e}[/high]")


def _init(args, console, ws, path: Path) -> int:
    from .board import scope_of
    from .policy.draft import agent_key

    if path.exists() and not getattr(args, "force", False):
        console.print(f"  [warn]{path} exists; edit it, or pass --force to write it again[/warn]")
        return EXIT_USAGE
    root = path.parent
    agents = _agents(args, console, ws, Path(args.root).resolve() if getattr(args, "root", None) else root)
    rows = []
    for a in sorted((a for a in agents if a.tools), key=lambda a: agent_key(a)):
        key = agent_key(a)
        rows.append((a, key, L.defaults(a, scope_of(args, ws, a), L.load(ws, key))))
    if not rows:
        console.print(f"  [warn]no agent with tools in {root}[/warn]")
        return EXIT_USAGE
    from .controls import Controls

    controls = Controls(ws)
    modes = {key: controls.get(key).mode for _a, key, _l in rows}
    text = render(rows, root, modes)
    if ws.plan_only:
        console.print(text)
        return EXIT_OK
    path.write_text(text, encoding="utf-8")
    console.print(Text.assemble(("  ✓ wrote ", "ok"), (str(path), "head"),
                                (f"  {_n(len(rows), 'agent')} · edit it, then: cslcore apply", "muted")))
    return EXIT_OK
