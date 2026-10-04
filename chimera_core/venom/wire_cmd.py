"""
`cslcore wire`: show, then make, the change that puts the guard in each agent's call path.

    cslcore wire                   every agent that has an active policy
    cslcore wire --agent KEY       one agent
    cslcore wire --yes             apply without asking (each diff is still printed)
    cslcore wire --undo [--agent KEY]   put the files back as they were

Only agents with an active policy are wired: a guard without its policy refuses every call.
After a change the host is scanned again, so the map and the live panel show what is guarded.
"""

from __future__ import annotations

import os
import sys
from typing import List

from rich.syntax import Syntax
from rich.text import Text

from . import wiring
from .commands import EXIT_OK, EXIT_USAGE, console_for, run_scan, save_report, workspace_for


def scan_probe(args, ws):
    """The probe for the files the scan saw: --root, else the folder the last scan covered."""
    from .probe import probe_for

    root = getattr(args, "root", None) or ws.load_state().get("scan_root")
    probe, _roots = probe_for(root if root and os.path.isdir(root) else None)
    return probe, root


def rescan(args, console, ws, root) -> None:
    """Scan again (quietly) so every view shows the wiring as it now is."""
    from argparse import Namespace

    scan_args = Namespace(**{**vars(args), "root": root, "no_anim": True})
    inv = run_scan(scan_args, console, live=False).inventory
    if not ws.plan_only:
        save_report(ws, inv)


def guarded_keys(ws) -> List[str]:
    from .bindings import Bindings

    return sorted(Bindings(ws).all())


def env_ready(console, args, ws, agent, plan: wiring.Plan, ask=None) -> bool:
    """Before a Python agent is wired: can the interpreter it runs with import chimera_core? If not,
    say how to install it and ask (install now, or wire anyway); without a terminal (`ask` None) the
    agent is not wired, since a wired agent without csl-core stops at import."""
    from .probe import agent_python, can_import, install_command, run_install

    if plan.kind != "code" or not plan.changes:
        return True
    probe, _root = scan_probe(args, ws)
    project = probe.real_path(agent.project) if agent.project else None
    python, how = agent_python(project)
    if can_import(python):
        return True
    cmd = install_command(python, project)
    console.print(Text.assemble(("  ", ""), (agent.display_name, "head"),
                                (f"  {python} ({how}) cannot import chimera_core: wired now, the agent would stop "
                                 "at its first import", "warn")))
    console.print(Text.assemble(("    install it with: ", "muted"), (" ".join(cmd), "brand")))
    if ask is None:
        console.print(Text(f"    not wired; after installing: cslcore wire --agent {plan.key}", style="muted"))
        return False
    if ask("Install csl-core there now (runs the command above)?", True):
        ok, last = run_install(cmd)
        if ok and can_import(python):
            console.print(Text("    ✓ csl-core installed there", style="ok"))
            return True
        console.print(Text(f"    the install did not finish: {last}", style="high"))
    return ask(f"Wire {agent.display_name} anyway? it stops at import until csl-core is installed there", False)


def show_plan(console, plan: wiring.Plan) -> None:
    head = Text.assemble(("  ", ""), (plan.agent, "head"))
    if plan.kind == "hook":
        head.append("  a hook: every tool call is decided first", style="muted")
    elif plan.kind == "code" and plan.changes:
        head.append(f"  {len(plan.wrapped)} tool function{'s' if len(plan.wrapped) != 1 else ''} wrapped: "
                    f"{', '.join(plan.wrapped)}", style="muted")
    console.print(head)
    for ch in plan.changes:
        console.print(Syntax(ch.diff(), "diff", theme="ansi_dark", background_color="default", word_wrap=True))
    if plan.note:
        console.print(Text("    " + plan.note, style="warn" if plan.kind == "manual" else "muted"))
    if getattr(plan, "requires", ""):
        console.print(Text("    " + plan.requires, style="warn"))


def cmd_wire(args) -> int:
    from .observe import current_mode
    from .policy.workbench import confirm

    console = console_for(args)
    ws = workspace_for(args)
    probe, root = scan_probe(args, ws)
    if getattr(args, "undo", False):
        results = wiring.undo(ws, getattr(args, "agent", None))
        if not results:
            console.print("  [muted]nothing was wired here by cslcore wire[/muted]")
            return EXIT_OK
        for key, path, result in results:
            style = "ok" if result in ("restored", "removed") else "warn"
            note = "  (it changed since; left as it is)" if result == "skipped" else ""
            console.print(Text.assemble(("  ", ""), (key, "head"), (f"  {result} ", style), (path, "muted"), (note, "warn")))
        rescan(args, console, ws, root)
        return EXIT_OK

    from .watch import _inventory
    from .policy.draft import agent_key

    inv = _inventory(ws)
    if inv is None:
        inv = run_scan(args, console, live=False).inventory
        save_report(ws, inv)
    want = getattr(args, "agent", None)
    agents = [a for a in inv.agents if not want or want in (agent_key(a), a.display_name, a.id)]
    if want and not agents:
        console.print(f"[high]no agent matches '{want}'[/high]")
        return EXIT_USAGE
    keys = set(guarded_keys(ws))
    applied, manual = [], []
    for a in agents:
        key = agent_key(a)
        if key not in keys:
            if want or a.tools:
                console.print(Text.assemble(("  ", ""), (a.display_name, "head"),
                                            ("  no policy yet: ", "muted"), (f"cslcore setup --agent {key}", "brand"),
                                            (" writes and checks one first", "muted")))
            continue
        plan = wiring.plan_for(a, key, ws, probe)
        if plan.kind == "done":
            console.print(Text.assemble(("  ", ""), (a.display_name, "head"), (f"  {plan.note}", "ok")))
            continue
        show_plan(console, plan)
        if plan.kind == "manual" or not plan.changes:
            manual.append(plan)
            continue
        interactive = sys.stdin.isatty() and not getattr(args, "yes", False)
        if not env_ready(console, args, ws, a, plan, (lambda q, d: confirm(console, q, False, default=d)) if interactive
                         else None):
            console.print()
            continue
        if confirm(console, f"Wire {a.display_name}?", bool(getattr(args, "yes", False)), default=True):
            try:
                wiring.apply(plan, ws)
            except (RuntimeError, OSError) as e:
                console.print(f"  [high]not wired: {e}[/high]")
                continue
            applied.append((plan, current_mode(ws, key)))
        console.print()
    if applied:
        rescan(args, console, ws, root)
        console.print(Text("  WIRED", style="label"))
        for plan, mode in applied:
            line = Text.assemble(("    ", ""), (plan.agent, "head"), ("  ", ""))
            if mode == "block":
                line.append("block mode: policy violations are stopped", style="ok")
            else:
                line.append("log mode: every call recorded, nothing stopped yet", style="warn")
                line.append(f"   cslcore mode --agent {plan.key} block", style="brand")
            console.print(line)
        console.print(Text("    see them live: cslcore watch · undo: cslcore wire --undo", style="muted"))
    elif not manual:
        console.print("  [muted]nothing to wire[/muted]")
    return EXIT_OK


def guard_one(console, args, ws, agent) -> bool:
    """Put one agent under a guard, all the way, asking at each step: a policy if it has none
    (drafted from its tools, checked by the gate, shown, then activated and bound with a mapping
    that passes the fail-open test), then the wiring change. True when the guard is in its call
    path afterwards. Used by the map and the live panel when someone wants to stop an agent that
    nothing guards yet."""
    from . import board as B
    from .policy import limits as L
    from .policy.draft import agent_key
    from .policy.workbench import confirm

    key = agent_key(agent)
    console.print(Text.assemble(("\n  ", ""), (agent.display_name, "head"), ("  nothing guards it yet", "warn")))
    if key not in guarded_keys(ws):
        lim = L.defaults(agent, B.scope_of(args, ws, agent), L.load(ws, key))
        console.print(Text("  LIMITS   the standard ones for what its tools do (change them later with l, or "
                           f"cslcore limits --agent {key})", style="label"))
        B.show_limits(console, agent, lim)
        if not confirm(console, f"Activate a policy made from these limits for {key}?", False, default=True):
            console.print("  [muted]nothing changed[/muted]")
            return False
        L.save(ws, lim)
        if B.make_policy(console, ws, agent, lim) is None:
            return False
    probe, root = scan_probe(args, ws)
    plan = wiring.plan_for(agent, key, ws, probe)
    if plan.kind == "manual":
        console.print(Text("  " + plan.note, style="warn"))
        return False
    if plan.kind == "hook" or plan.changes:
        show_plan(console, plan)
        if not env_ready(console, args, ws, agent, plan, lambda q, d: confirm(console, q, False, default=d)):
            console.print("  [muted]not wired[/muted]")
            return False
        if not confirm(console, f"Wire {agent.display_name}?", False, default=True):
            console.print("  [muted]not wired[/muted]")
            return False
        try:
            wiring.apply(plan, ws)
        except (RuntimeError, OSError) as e:
            console.print(f"  [high]not wired: {e}[/high]")
            return False
    rescan(args, console, ws, root)
    from .watch import _inventory

    inv = _inventory(ws)
    now = next((a for a in inv.agents if a.id == agent.id), None) if inv is not None else None
    ok = now is not None and now.guard.status != "none"
    console.print(Text("  ✓ its guard is in the call path" if ok else "  the scan does not see the guard yet",
                       style="ok" if ok else "warn"))
    return ok
