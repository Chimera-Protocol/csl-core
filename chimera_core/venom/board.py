"""
The protection board: every agent, the riskiest first, and how far its protection has come.

    group      high: moves money, runs commands or queries, deletes, changes credentials
               medium: writes files, sends or publishes; low: reads only
    stages     limits (set by the operator, or the standard ones) · policy (made from them, checked
               with Z3) · wired (the guard is in the agent's call path) · mode (block stops, log only
               records) · check (sample calls decided by the active policy, as the limits say)

Every stage is read from what is on disk (the limits, the bound policy and mapping, the last scan,
the controls, the stored check), so the board always opens where the operator left it.

One agent at a time: its limits, the policy made from them, the mapping and its fail-open test, the
mode, the wiring change (shown as a diff, confirmed) and the check. Or standard protection for every
agent still unprotected, with one confirmation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.table import Table
from rich.text import Text

from .model import Agent
from .policy import limits as L

HIGH = {"spend", "shell", "sql", "destroy", "identity"}
MEDIUM = {"write", "send", "publish"}
WORDS = {"spend": "moves money", "shell": "runs commands", "sql": "queries databases", "destroy": "deletes",
         "identity": "changes credentials", "write": "writes files", "send": "sends data out", "publish": "publishes",
         "read": "reads", "other": "other tools"}
GROUPS = ("high", "medium", "low")


@dataclass
class Row:
    agent: Agent
    key: str
    risk: str
    does: List[str] = field(default_factory=list)  # what its tools do, in words, riskiest first
    limits: bool = False  # set by the operator or the standard ones saved
    policy: str = ""  # "limits" | "hand" | "adopted" | ""
    wired: str = "none"  # wired | partly | manual | none
    wiring_note: str = ""
    own: str = ""  # a policy its own code already enforces (a 0.5.1 integration)
    mode: str = "log"
    frozen: bool = False
    check: str = ""  # ok | failed | "" (not run for the active policy)
    skipped: bool = False  # the operator chose to leave it untouched on the board

    @property
    def protected(self) -> bool:
        """Its limits are enforced: a policy, the guard in its call path, block mode, a passing check
        (or a policy of the operator's own, which the check does not judge)."""
        if self.own and not self.policy:
            return False
        if self.policy == "adopted":
            return not self.frozen  # its own guard, in its own code, with its own policy
        return (bool(self.policy) and self.wired == "wired" and self.mode == "block" and not self.frozen
                and (self.check == "ok" or self.policy in ("hand", "adopted")))

    @property
    def state(self) -> Text:
        if self.frozen:
            return Text("frozen", style="high")
        if self.skipped and not self.policy:
            return Text("skipped, left untouched", style="muted")
        if self.own and not self.policy:
            return Text(f"its code enforces {self.own}; keep it", style="warn")
        if self.policy == "adopted" and self.protected:
            return Text("protected by its own code", style="ok")
        if self.protected:
            return Text("protected", style="ok")
        if self.policy and self.wired == "partly":
            return Text("partly wired", style="warn")
        if self.policy and self.wired == "wired" and self.mode == "log":
            return Text("recording only (log)", style="warn")
        if self.policy and self.wired == "wired" and self.check == "failed":
            return Text("check failed", style="high")
        if self.policy and self.wired == "manual":
            return Text("policy ready, wire by hand", style="warn")
        if self.policy:
            return Text("policy ready, not wired", style="warn")
        return Text("not protected", style="muted")


def kinds_of(agent: Agent, lim: Optional[L.Limits] = None) -> List[str]:
    kinds = {(lim.tools[t.name].kind if lim and t.name in lim.tools else L.kind_of(t)) for t in L.tools_of(agent, lim)}
    order = list(WORDS)
    return sorted(kinds, key=order.index)


def risk_of(kinds: List[str]) -> str:
    if set(kinds) & HIGH:
        return "high"
    if set(kinds) & MEDIUM:
        return "medium"
    return "low"


def _fingerprint(ws, key: str) -> str:
    from .bindings import Bindings

    b = Bindings(ws).get(key)
    if b is None:
        return ""
    h = hashlib.sha256()
    for path in (b.policy, b.mapping or ""):
        h.update((ws.read(Bindings(ws).abs(path)) or "").encode() if path else b"")
    return h.hexdigest()[:16]


def stored_check(ws, key: str) -> str:
    c = (ws.load_state().get("checks") or {}).get(key) or {}
    return c.get("result", "") if c.get("of") == _fingerprint(ws, key) and c.get("of") else ""


def store_check(ws, key: str, report) -> None:
    state = ws.load_state()
    runs, stops, bad = report.counts()
    state.setdefault("checks", {})[key] = {"of": _fingerprint(ws, key), "result": "ok" if report.ok else "failed",
                                           "runs": runs, "stops": stops, "failed": bad}
    ws.save_state(state)


def own_policy(agent: Agent, policies) -> str:
    """The policy an agent's own code already enforces (a 0.5.1 integration), by file name."""
    from pathlib import Path

    from .analysis.coverage import link_policies

    if agent.guard.status == "none" or agent.kind == "assistant":
        return ""
    linked = [p for p in link_policies(agent, policies or []) if p.status != "draft"]
    return Path(linked[0].path).name if linked else ""


def wiring_of(args, ws, agent: Agent):
    """(wired | partly | manual | none, note) from the wiring plan: is this agent's guard, with its
    own binding, in the call path of every tool?"""
    plan = wire_plan(args, ws, agent)
    if plan.kind == "done":
        return ("partly" if plan.missing else "wired"), plan.note
    if plan.kind == "manual":
        return "manual", plan.note
    return "none", plan.note


def row_for(ws, agent: Agent, setup_state: Optional[Dict[str, Any]] = None, args=None, policies=None) -> Row:
    from .bindings import Bindings
    from .controls import Controls
    from .policy.draft import agent_key

    key = agent_key(agent)
    lim = L.load(ws, key)
    kinds = kinds_of(agent, lim)
    r = Row(agent, key, risk_of(kinds), [WORDS[k] for k in kinds if k not in ("read", "other")] or ["reads"])
    r.limits = lim is not None
    st = (setup_state or {}).get(agent.id) or {}
    r.skipped = bool(st.get("board_skip"))
    b = Bindings(ws).get(key)
    if b is not None and b.mapping:
        text = ws.read(Bindings(ws).abs(b.policy)) or ""
        r.policy = "limits" if "made from its limits" in text else ("adopted" if st.get("adopted") else "hand")
    elif st.get("adopted") and st.get("policy"):
        r.policy = "adopted"
    r.own = own_policy(agent, policies) if r.policy != "limits" else ""
    if r.policy == "adopted" or (r.own and not r.policy):
        r.wired = "wired"  # its own guard, in its own code
    else:
        r.wired, r.wiring_note = wiring_of(args, ws, agent)
    ctl = Controls(ws).get(key, Controls(ws).default_mode() or "log")
    r.mode, r.frozen = ctl.mode, ctl.disabled
    r.check = stored_check(ws, key) if r.policy == "limits" else ""
    return r


def rows_for(ws, agents: List[Agent], setup_state: Optional[Dict[str, Any]] = None, args=None,
             policies=None) -> List[Row]:
    rows = [row_for(ws, a, setup_state, args, policies) for a in agents]
    return sorted(rows, key=lambda r: (GROUPS.index(r.risk), r.protected, r.key))


def table(rows: List[Row], width: int = 120) -> Table:
    """The board; below 110 columns without what each agent does (the group still says how risky)."""
    wide = width >= 110
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 2 if wide else 1, 0, 0))
    t.add_column("", style="brand", no_wrap=True, justify="right", min_width=2)
    t.add_column("AGENT", style="head", no_wrap=True, min_width=10)
    if wide:
        t.add_column("WHAT IT DOES", style="muted", ratio=1)
    t.add_column("LIMITS", no_wrap=True)
    t.add_column("POLICY", no_wrap=True)
    t.add_column("WIRED", no_wrap=True)
    t.add_column("MODE", no_wrap=True)
    t.add_column("CHECK", no_wrap=True)
    t.add_column("STATE", overflow="fold", ratio=1, min_width=12)
    group = None
    for i, r in enumerate(rows, 1):
        if r.risk != group:
            group = r.risk
            t.add_row("", Text(f"{group.upper()} RISK", style={"high": "high", "medium": "warn", "low": "muted"}[group]),
                      *([""] * (7 if wide else 6)))
        yes, no = ("✓", "ok"), ("·", "muted")
        t.add_row(str(i), r.key, *([", ".join(r.does)] if wide else []),
                  Text(*(yes if r.limits else no)),
                  Text({"limits": "✓", "hand": "yours", "adopted": "yours"}.get(r.policy, "in code" if r.own else "·"),
                       style="ok" if r.policy else ("warn" if r.own else "muted")),
                  Text(*(yes if r.wired == "wired" else ("partly", "warn") if r.wired == "partly"
                         else ("by hand", "warn") if r.wired == "manual" else no)),
                  (Text("its own", style="muted") if r.policy == "adopted" else
                   Text(r.mode, style="ok" if r.mode == "block" else "warn") if r.policy else Text("·", style="muted")),
                  Text({"ok": "✓", "failed": "✗"}.get(r.check, "·"), style={"ok": "ok", "failed": "high"}.get(r.check, "muted")),
                  r.state)
    return t


class TerminalUI:
    """Questions at a terminal, for the board outside setup (the live panel, the map)."""

    def __init__(self, console) -> None:
        self.console = console

    def ask(self, question: str, default: bool) -> bool:
        from rich.markup import escape

        from .policy.workbench import confirm
        return confirm(self.console, escape(question), False, default=default)

    def choose(self, question: str, choices: List[str], default: str) -> str:
        from rich.markup import escape
        from rich.prompt import Prompt
        return Prompt.ask(f"  {escape(question)}", choices=choices, default=default, console=self.console)

    def text_input(self, question: str, default: str = "") -> str:
        from rich.markup import escape
        from rich.prompt import Prompt
        return Prompt.ask(f"  {escape(question)}", default=default, show_default=bool(default), console=self.console)


# ---------------------------------------------------------------------------
# asking for limits
# ---------------------------------------------------------------------------

def ask_limits(ui, console, agent: Agent, lim: L.Limits) -> None:
    """At a terminal: what each tool may do, the money limits in the operator's own numbers, any
    change per tool (allow, approval, block, a number's range), and tools the scan did not see.
    Enter keeps what is shown. `ui` asks: text_input(question, default) and choose(question, choices, default)."""
    from .policy.draft import agent_key

    key = agent_key(agent)
    show_limits(console, agent, lim)
    for name, tl in sorted(lim.tools.items()):
        if tl.kind != "spend" or not tl.amount_param or tl.decide == "block":
            continue
        while True:
            lo = ui.text_input(f"{name}: allowed freely up to", f"{tl.allow_up_to:,}")
            hi = ui.text_input(f"{name}: never above", f"{tl.never_above:,}")
            try:
                tl.allow_up_to, tl.never_above = L.parse_range(f"{lo}..{hi}")
                break
            except L.LimitError as e:
                console.print(f"  [warn]{e}[/warn]")
    while True:
        change = (ui.text_input("Change a tool? TOOL=allow|approval|block|standard, TOOL.PARAM=FREE..MAX "
                                "(Enter when done)", "") or "").strip()
        if not change:
            break
        try:
            done = apply_change(lim, key, change)
        except L.LimitError as e:
            console.print(f"  [warn]{e}[/warn]")
            continue
        for item in done:
            console.print(Text.assemble(("    ✓ ", "ok"), (item, "text")))
    while True:
        extra = (ui.text_input("Another tool this agent can call that is not listed (name, Enter for none)", "") or "").strip()
        if not extra:
            break
        risk = ui.choose(f"What does {extra} do", ["spend", "shell", "sql", "write", "send", "destroy", "other"], "other")
        item = {"name": extra, "risk": L.RISKS[risk]}
        if risk == "spend":
            item["amount_param"] = (ui.text_input("Its amount parameter", "amount") or "amount").strip()
        lim.extra_tools = [e for e in lim.extra_tools if e["name"] != extra] + [item]
        lim.tools.pop(extra, None)
        L.defaults(agent, lim.scope, lim)
        if risk == "spend":
            tl = lim.tools[extra]
            lo = ui.text_input(f"{extra}: allowed freely up to", f"{tl.allow_up_to:,}")
            hi = ui.text_input(f"{extra}: never above", f"{tl.never_above:,}")
            try:
                tl.allow_up_to, tl.never_above = L.parse_range(f"{lo}..{hi}")
            except L.LimitError as e:
                console.print(f"  [warn]{e}; the defaults are kept[/warn]")
        console.print(Text.assemble(("    ✓ ", "ok"), (extra, "head"), (f"  {risk}", "muted")))


def apply_change(lim: L.Limits, key: str, change: str) -> List[str]:
    """TOOL=allow|approval|block|standard, or TOOL[.PARAM]=FREE..MAX."""
    tool, sep, value = change.partition("=")
    if not sep:
        raise L.LimitError(f"expected TOOL=allow|approval|block|standard or TOOL.PARAM=FREE..MAX, got {change!r}")
    if value in ("allow", "approval", "block", "standard"):
        if tool not in lim.tools:
            raise L.LimitError(f"{key} has no tool {tool!r}")
        lim.tools[tool].decide = None if value == "standard" else value
        return [f"{tool}: {value}"]
    return L.apply_flags(lim, key, [f"{key}.{change}"])


def show_limits(console, agent: Agent, lim: L.Limits) -> None:
    console.print(Text.assemble(("  ", ""), (agent.display_name, "head"), (f"   {lim.profile} profile", "muted")))
    if lim.scope:
        console.print(Text.assemble(("    writes under  ", "muted"), (", ".join(lim.scope), "text")))
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 0))
    t.add_column(style="text", no_wrap=True)
    t.add_column(style="label", no_wrap=True)
    t.add_column(style="muted")
    for name, kind, what in L.describe(lim):
        t.add_row(name, kind, what)
    from rich.padding import Padding
    console.print(Padding(t, (0, 0, 0, 4)))


# ---------------------------------------------------------------------------
# protecting one agent
# ---------------------------------------------------------------------------

def make_policy(console, ws, agent: Agent, lim: L.Limits) -> Optional[str]:
    """Write the policy made from the limits and bind it (mapping + fail-open test). Returns the
    policy's workspace path, or None when it does not pass. A policy written by hand is kept."""
    from .policy.binder import bind
    from .policy.draft import agent_key
    from .policy.gate import verify_text

    key = agent_key(agent)
    path = ws.policies / f"{key}.csl"
    old = ws.read(path)
    if old and "made from its limits" not in old and "Generated by CSL-Core Venom" not in old:
        console.print(f"  [muted]{ws.rel(path)} was written by hand; it is kept as it is[/muted]")
    else:
        text, _notes = L.policy_text(agent, lim)
        gate = verify_text(text)
        if not gate.ok:
            issue = gate.issues[0].message if gate.issues else gate.stage
            console.print(f"  [high]the policy for these limits does not pass the check: {issue}[/high]")
            return None
        if old != text:
            ws.write_text(path, text)
        if not old:
            from .controls import mode_on_activation
            mode_on_activation(ws, key, True)
        console.print(Text.assemble(("  ✓ policy ", "ok"), (ws.rel(path), "head"),
                                    (f"  {plural(gate.rules, 'rule')}, Z3: no contradictions", "muted")))
    plan = bind(ws, path, [agent])
    res = plan.results[0] if plan.results else None
    if res is None or not res.ok:
        console.print(f"  [high]the mapping did not pass its test: {res.message if res else 'no result'}[/high]")
        return None
    console.print(Text.assemble(("  ✓ mapping ", "ok"), (f"{res.message}", "muted")))
    return ws.rel(path)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def wire_plan(args, ws, agent: Agent):
    from . import wiring
    from .policy.draft import agent_key
    from .wire_cmd import scan_probe

    probe, _root = scan_probe(args, ws)
    return wiring.plan_for(agent, agent_key(agent), ws, probe)


def approval_note(args, ws, agent: Agent) -> str:
    """Where a call that needs an approval gets it, on this agent as it is wired."""
    from argparse import Namespace

    plan = wire_plan(args if args is not None else Namespace(), ws, agent)
    if agent.kind == "assistant" and plan.kind in ("hook", "done"):
        return ("with approval: Claude Code asks you before it runs" if plan.kind == "done"
                else "with approval: Claude Code will ask you, once it is wired")
    if plan.kind == "done":
        return "with approval: the call waits; a person approves it in cslcore watch (a), then it runs once"
    if plan.kind == "code" and plan.changes:
        return "with approval: once wired, the call waits for a person in cslcore watch (a)"
    return ("with approval: needs approval, but this agent has no place to approve yet (its code needs "
            "guard.check), so such calls stop")


def run_check(console, ws, agent: Agent, lim: Optional[L.Limits] = None, compact: bool = False, args=None) -> bool:
    from . import check
    from .policy.draft import agent_key

    report = check.run(ws, agent, lim)
    if report.cases:
        store_check(ws, agent_key(agent), report)
    check.show(console, report, agent_key(agent), compact=compact, approval=approval_note(args, ws, agent))
    return report.ok or not report.cases


def protect(ui, console, args, ws, agent: Agent, *, ask: bool = True) -> bool:
    """One agent, all the way: limits, policy, mapping, mode, wiring (diff, confirmation), check.
    True when its limits are enforced afterwards (or recorded, in log mode, if the operator chose it)."""
    from . import wiring
    from .controls import Controls
    from .policy.draft import agent_key
    from .wire_cmd import rescan, scan_probe, show_plan

    key = agent_key(agent)
    console.print()
    lim = L.defaults(agent, scope_of(args, ws, agent), L.load(ws, key))
    if ask:
        ask_limits(ui, console, agent, lim)
    else:
        show_limits(console, agent, lim)
    if ws.plan_only:
        console.print("  [muted]--plan-only: nothing is written[/muted]")
        return False
    L.save(ws, lim)
    if make_policy(console, ws, agent, lim) is None:
        return False
    controls = Controls(ws)
    current = controls.get(key).mode  # its own, else the workspace default, else log
    if ask:
        mode = ui.choose("Mode: block stops what its limits do not allow; log only records it", ["block", "log"], current)
    else:
        mode = getattr(args, "mode", None) or current
    if mode != current:
        controls.set_mode(key, mode)
    plan = wire_plan(args, ws, agent)
    if plan.kind == "manual":
        console.print(Text("  " + plan.note, style="warn"))
    elif plan.changes:
        show_plan(console, plan, ws=ws)
        from .wire_cmd import env_ready
        if not env_ready(console, args, ws, agent, plan, ui.ask):
            console.print("  [muted]not wired[/muted]")
        elif ui.ask(f"Wire {agent.display_name}? (undo any time: cslcore wire --undo)", True):
            try:
                wiring.apply(plan, ws)
            except (RuntimeError, OSError) as e:
                console.print(f"  [high]not wired: {e}[/high]")
            else:
                _probe, root = scan_probe(args, ws)
                rescan(args, console, ws, root)
                console.print(Text.assemble(("  ✓ wired ", "ok"), (agent.display_name, "head")))
    return run_check(console, ws, agent, lim, args=args)


def protect_rest(ui, console, args, ws, agents: List[Agent]) -> int:
    """Standard protection for every agent given: the standard limits (or what was set before),
    their policies, block mode and the wiring changes, all shown first and confirmed once."""
    from . import wiring
    from .controls import Controls
    from .policy.draft import agent_key
    from .wire_cmd import rescan, scan_probe, show_plan

    if not agents:
        return 0
    console.print()
    console.print(Text(f"  Standard protection for {plural(len(agents), 'agent')}: ", style="head"))
    for a in agents:
        kinds = kinds_of(a, L.load(ws, agent_key(a)))
        console.print(Text.assemble(("    ", ""), (agent_key(a), "head"), (f"  {', '.join(WORDS[k] for k in kinds)}", "muted")))
    chosen = getattr(args, "mode", None)  # --mode; else a first activation starts in block (controls)
    console.print(Text(f"    standard limits, a policy for each checked with Z3 (in the workspace only), "
                       f"{chosen or 'block'} mode, then the wiring changes below, which you confirm. "
                       "Change any of it later: a number here, cslcore limits, cslcore mode", style="muted"))
    if ws.plan_only:
        console.print("  [muted]--plan-only: nothing is written[/muted]")
        return 0
    ready = []
    for a in agents:
        key = agent_key(a)
        lim = L.defaults(a, scope_of(args, ws, a), L.load(ws, key))
        L.save(ws, lim)
        console.print(Text.assemble(("  ", ""), (key, "head")))
        if make_policy(console, ws, a, lim) is not None:
            if chosen and Controls(ws).get(key).mode != chosen:
                Controls(ws).set_mode(key, chosen)
            ready.append(a)
    plans = [wire_plan(args, ws, a) for a in ready]
    agent_of = {agent_key(a): a for a in ready}
    from .wire_cmd import env_ready

    todo = [p for p in plans if p.changes and p.kind in ("hook", "code")
            and env_ready(console, args, ws, agent_of[p.key], p, ui.ask)]
    for p in plans:
        if p.kind == "manual":
            console.print(Text.assemble(("  ", ""), (p.agent, "head"), (f"  {p.note}", "warn")))
    if todo:
        console.print()
        for p in todo:
            show_plan(console, p, ws=ws)
        if ui.ask(f"Wire {plural(len(todo), 'agent')}? (undo any time: cslcore wire --undo)", True):
            probe, root = scan_probe(args, ws)
            wiring.apply_many([(agent_of[p.key], p) for p in todo], ws, probe, on_error=lambda p, e: console.print(
                f"  [high]{p.agent} not wired: {e}[/high]"))
            rescan(args, console, ws, root)
    console.print()
    for a in ready:
        run_check(console, ws, a, compact=True, args=args)
    return len(ready)


def scope_of(args, ws, agent: Agent) -> List[str]:
    from .wire_cmd import scan_probe

    probe, _root = scan_probe(args, ws)
    base = agent.project or (probe.home() if agent.kind == "assistant" else None)
    return [probe.real_path(base)] if base else []


# ---------------------------------------------------------------------------
# the board itself
# ---------------------------------------------------------------------------

def fresh_agents(ws, agents: List[Agent]):
    """The same agents from the latest scan (wiring changes what the scan sees), and its policies."""
    from .model import Inventory

    data = ws.latest_inventory()
    if data is None:
        return agents, []
    inv = Inventory.from_dict(data)
    now = {a.id: a for a in inv.agents}
    return [now.get(a.id, a) for a in agents], inv.policies


def skip(setup_state: Optional[Dict[str, Any]], r: Row, on: bool) -> None:
    """"Skip / leave it untouched": kept with the setup's own state (the workspace), never in the
    agent's files; standard protection (a) passes it by."""
    if setup_state is None:
        return
    st = setup_state.setdefault(r.agent.id, {"key": r.key})
    if on:
        st["board_skip"] = True
    else:
        st.pop("board_skip", None)
        st.pop("skipped", None)


def run(ui, console, args, ws, agents: List[Agent], setup_state: Optional[Dict[str, Any]] = None,
        other_ways=None) -> List[Row]:
    """The board at a terminal, until the operator leaves it. `other_ways(agent)` offers the other
    ways to give an agent a policy (setup: an existing policy, the studio, an editor, an assistant)."""
    while True:
        agents, policies = fresh_agents(ws, agents)
        rows = rows_for(ws, agents, setup_state, args, policies)
        console.print()
        console.print(Text("  PROTECTION   the riskiest agents first", style="label"))
        console.print(table(rows, console.width))
        open_rows = [r for r in rows if not r.policy and not r.own and not r.skipped]
        done = sum(1 for r in rows if r.protected)
        console.print(Text.assemble(("  ", ""), (f"{done} of {len(rows)} protected", "ok" if done == len(rows) else "text"),
                                    ("   number: one agent (limits, policy, wiring, check)", "muted")))
        keys = [str(i) for i in range(1, len(rows) + 1)]
        if open_rows:  # Enter does the recommended thing: standard protection for the rest
            console.print(Text.assemble(("  ", ""), ("Enter", "brand"),
                                        (f"  standard protection for the {plural(len(open_rows), 'agent')} without a "
                                         "policy (you see each wiring change first)", "muted")))
            console.print(Text.assemble(("  ", ""), ("c", "brand"), ("  continue without it", "muted")))
        else:
            console.print(Text.assemble(("  ", ""), ("Enter", "brand"), ("  continue", "muted")))
        pick = (ui.text_input("choice", "a" if open_rows else "") or "").strip().lower()
        if not pick or pick == "c":
            return rows
        if pick == "a" and open_rows:
            protect_rest(ui, console, args, ws, [r.agent for r in open_rows])
            agents, policies = fresh_agents(ws, agents)
            rows = rows_for(ws, agents, setup_state, args, policies)
            if not [r for r in rows if not r.policy and not r.own and not r.skipped]:
                console.print()
                console.print(table(rows, console.width))  # where it ended; a number later: cslcore setup, b
                return rows
            continue
        if pick in keys:
            r = rows[int(pick) - 1]
            if r.own and not r.policy and other_ways is not None:
                console.print(f"  [muted]{r.key}: its code already enforces {r.own}; keeping it is the first choice[/muted]")
                other_ways(r.agent)
                continue
            if r.policy in ("", "hand") and not r.limits or r.skipped:
                other = other_ways is not None
                how = ui.choose(f"{r.key}: Enter sets its limits"
                                + ("; o other ways (an existing policy, the studio, an editor, your assistant)" if other else "")
                                + "; s skip it (nothing is changed for it)", ["l", "o", "s"] if other else ["l", "s"],
                                "l")
                if how == "s":
                    skip(setup_state, r, True)
                    console.print(f"  [muted]{r.key} skipped: no limits, policy, mode or wiring change; "
                                  "a number brings it back[/muted]")
                    continue
                skip(setup_state, r, False)
                if how == "o":
                    other_ways(r.agent)
                    continue
            if r.policy == "adopted":
                console.print(f"  [muted]{r.key} keeps its own policy (enforced in its code)[/muted]")
                continue
            protect(ui, console, args, ws, r.agent)
            continue
        console.print(f"  [warn]{pick!r}: a number, a, c or Enter[/warn]")
