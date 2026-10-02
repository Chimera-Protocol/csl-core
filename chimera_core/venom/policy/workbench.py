"""
`cslcore policy`: the policy workbench.

Every change goes through the same gate: parse, validate, Z3, a diff against the active
version, and operator confirmation. Drafts live in .csl/venom/drafts/ and are never used
at runtime; activation writes to policies/. Existing hand-written policies are never
modified in place: edit, extend and fix produce drafts.
"""

from __future__ import annotations

import difflib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..analysis.coverage import link_policies, tool_variable
from ..commands import EXIT_OK, EXIT_USAGE, _load_inventory, console_for, workspace_for
from ..layers.governance import policy_label, read_policy
from ..model import Agent, Inventory, PolicyRef
from ..render.screen import _section
from . import draft as D
from .gate import GateResult, verify_text

EXIT_GATE = 4

# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

_KEYWORDS = re.compile(r"\b(CONFIG|DOMAIN|VARIABLES|STATE_CONSTRAINT|WHEN|THEN|ALWAYS|MUST|NOT|BE|MAY|AND|OR|True|False)\b")


def highlight_csl(text: str, start_line: int = 1, numbers: bool = True) -> Text:
    out = Text()
    for i, line in enumerate(text.splitlines(), start_line):
        if numbers:
            out.append(f"{i:>4}  ", style="muted")
        code, _, comment = line.partition("//")
        pos = 0
        for m in re.finditer(r'"[^"]*"|\b\d+(?:\.\d+)?\b|' + _KEYWORDS.pattern, code):
            out.append(code[pos:m.start()], style="text")
            tok = m.group(0)
            style = "ok" if tok.startswith('"') else ("warn" if tok[0].isdigit() else "brand")
            out.append(tok, style=style)
            pos = m.end()
        out.append(code[pos:], style="text")
        if comment or line.strip().startswith("//"):
            out.append("//" + comment, style="muted")
        out.append("\n")
    out.rstrip()
    return out


def diff_text(old: str, new: str, old_name: str, new_name: str) -> Optional[Text]:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), old_name, new_name, lineterm="", n=2))
    if not lines:
        return None
    out = Text()
    for l in lines:
        style = "muted"
        if l.startswith("+") and not l.startswith("+++"):
            style = "ok"
        elif l.startswith("-") and not l.startswith("---"):
            style = "high"
        elif l.startswith("@@"):
            style = "brand.dim"
        out.append(l + "\n", style=style)
    out.rstrip()
    return out


def gate_panel(g: GateResult, name: str) -> Panel:
    if g.ok:
        body = Text.assemble(("verified  ", "ok"), (f"{g.rules} rules · {g.variables} variables · Z3: no contradictions", "text"),
                             ("\nhash ", "muted"), (g.policy_hash or "", "muted"))
        return Panel(body, title=Text(f" gate · {name} ", style="ok"), title_align="left", box=box.ROUNDED, border_style="ok", padding=(0, 1))
    rows = Table.grid(padding=(0, 1))
    rows.add_column(style="high", no_wrap=True)
    rows.add_column(style="text")
    for i in g.issues[:6]:
        msg = i.message + (f"  (rules: {', '.join(i.rules)})" if i.rules else "")
        if i.model:
            msg += "\n" + "example: " + ", ".join(f"{k}={v}" for k, v in sorted(i.model.items())[:6])
        rows.add_row(i.kind, msg)
    return Panel(Group(Text(f"failed at {g.stage}: this draft cannot be activated", style="high"), rows),
                 title=Text(f" gate · {name} ", style="high"), title_align="left", box=box.ROUNDED, border_style="high", padding=(0, 1))


def confirm(console, question: str, yes: bool, default: bool = False) -> bool:
    if yes:
        console.print(f"  [muted]{question}[/muted] [ok]yes (--yes)[/ok]")
        return True
    if not sys.stdin.isatty():
        console.print(f"  [muted]{question}[/muted] [warn]no[/warn] [muted](not a terminal; pass --yes to confirm)[/muted]")
        return False
    from rich.prompt import Confirm
    return Confirm.ask(f"  {question}", console=console, default=default)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def _policy_refs(ws, inv: Inventory) -> List[PolicyRef]:
    refs: Dict[str, PolicyRef] = {}
    for path, text, status in ws.policy_items():
        refs[path] = read_policy(path, text, status)
    for p in inv.policies:
        if p.path not in refs and p.status == "found":
            refs[p.path] = p
    order = {"active": 0, "draft": 1, "found": 2}
    return sorted(refs.values(), key=lambda p: (order.get(p.status, 3), p.path))


def find_policy(ws, inv: Inventory, target: Optional[str], prefer: Optional[str] = None) -> Optional[PolicyRef]:
    if not target:
        return None
    refs = _policy_refs(ws, inv)
    if prefer:
        refs = sorted(refs, key=lambda p: p.status != prefer)
    p = Path(target)
    if p.suffix == ".csl" and p.exists():
        full = str(p.resolve())
        for r in refs:
            if r.path == full:
                return r
        return read_policy(full, p.read_text(encoding="utf-8"), "found")
    for r in refs:
        if target in (Path(r.path).stem, Path(r.path).name, r.policy_id, r.domain, r.path):
            return r
    agent = find_agent(inv, target)
    if agent is not None:
        for r in refs:
            if Path(r.path).stem == D.agent_key(agent):
                return r
        linked = link_policies(agent, refs)
        if linked:
            return linked[0]
    return None


def find_agent(inv: Inventory, target: Optional[str]) -> Optional[Agent]:
    if not target:
        return None
    a = inv.agent(target)
    if a is None:
        matches = [x for x in inv.agents if target in x.id or target == D.agent_key(x)]
        a = matches[0] if len(matches) == 1 else None
    return a


def agents_for(inv: Inventory, ref: PolicyRef) -> List[Agent]:
    stem = Path(ref.path).stem
    out = []
    for a in inv.agents:
        if D.agent_key(a) == stem or ref.path in [p.path for p in link_policies(a, [ref])] and a.guard.status != "none":
            out.append(a)
    return out


def _text(ws, ref: PolicyRef, args) -> str:
    """Policy source: workspace files directly, host files through the probe (fixture-aware)."""
    text = ws.read(ref.path)
    if text is None:
        from ..probe import probe_for
        probe, _ = probe_for(getattr(args, "root", None))
        text = probe.read_text(ref.path)
    return text or ""


def _record(ws, path: str, g: GateResult) -> None:
    state = ws.load_state()
    ver = state.setdefault("verifications", {})
    ver[ws.rel(path)] = {"ok": g.ok, "hash": g.policy_hash, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    ws.save_state(state)


def _save_draft(console, ws, name: str, text: str, yes: bool, base: Optional[str] = None, base_name: str = "") -> Optional[Path]:
    path = ws.drafts / f"{name}.csl"
    g = verify_text(text)
    old = base if base is not None else ws.read(path)
    if old:
        d = diff_text(old, text, base_name or ws.rel(path), f"{ws.rel(path)} (new)")
        if d is None:
            console.print("  [muted]no changes[/muted]")
            return None
        console.print(Panel(d, title=Text(" diff ", style="label"), title_align="left", box=box.ROUNDED, border_style="muted", padding=(0, 1)))
    console.print(gate_panel(g, f"{name}.csl"))
    if ws.plan_only:
        console.print(f"  [muted]--plan-only: would write {ws.rel(path)}[/muted]")
        return None
    if not confirm(console, f"Save draft to {ws.rel(path)}?", yes, default=True):
        console.print("  [muted]nothing written[/muted]")
        return None
    ws.write_text(path, text)
    _record(ws, str(path), g)
    console.print(f"  [ok]saved[/ok] {ws.rel(path)}" + ("" if g.ok else "  [warn](draft only: fix the gate errors before activating)[/warn]"))
    return path


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def act_list(console, ws, inv: Inventory) -> int:
    refs = _policy_refs(ws, inv)
    state = ws.load_state().get("verifications", {})
    wanted = [r for r in refs if r.status in ("active", "draft") or agents_for(inv, r)]
    if not wanted:
        console.print(Panel(Text.assemble(("No policies in this workspace yet (first install).\n", "text"),
                                          ("Draft one per agent: ", "muted"), ("cslcore policy new --all", "brand"),
                                          ("  or run the guided flow: ", "muted"), ("cslcore setup", "brand")),
                            box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
        return EXIT_OK
    t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
    for col, just in (("Policy", "left"), ("Status", "left"), ("Rules", "right"), ("Agents", "left"), ("Coverage", "right"),
                      ("Drift", "right"), ("Verified", "left")):
        t.add_column(col, justify=just, overflow="ellipsis", no_wrap=True)  # type: ignore[arg-type]
    for r in wanted:
        agents = agents_for(inv, r)
        tools = [tt for a in agents for tt in a.tools if not tt.name.endswith("/*")]
        covered = [tt for tt in tools if tt.coverage in ("guarded", "exempt")] if r.status == "active" else []
        if r.status != "active":
            from ..analysis.coverage import _covers
            tv = {a.id: tool_variable(r, a.tools) for a in agents}
            covered = [tt for a in agents for tt in a.tools if not tt.name.endswith("/*") and _covers(r, tv[a.id], tt)]
        label = policy_label(r)
        drift = [d for d in inv.drift if d.policy == label and d.kind in ("unknown_value", "coercion")]
        v = state.get(ws.rel(r.path))
        verified = Text("-", style="muted") if not v else Text(("ok " if v["ok"] else "failed ") + v["at"][:16].replace("T", " "), style="ok" if v["ok"] else "high")
        status = Text(r.status if not r.error else "error", style={"active": "ok", "draft": "warn", "found": "muted"}.get(r.status, "high"))
        cov = f"{len(covered)}/{len(tools)}" if tools else "-"
        t.add_row(Text(label, style="head"), status, str(len(r.rules)), ", ".join(a.display_name for a in agents) or "-", cov,
                  Text(str(len(drift)), style="warn" if drift else "muted"), verified)
    console.print(t)
    return EXIT_OK


def act_show(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    text = _text(ws, ref, args)
    console.print(Panel(highlight_csl(text), title=Text(f" {policy_label(ref)} · {ref.status} ", style="brand"),
                        subtitle=Text(f" {ws.rel(ref.path)} ", style="muted"), title_align="left", box=box.ROUNDED,
                        border_style="brand.dim", padding=(0, 1)))
    agents = agents_for(inv, ref)
    if ref.rules:
        t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
        t.add_column("Rule", style="head", no_wrap=True)
        t.add_column("Speaks about", style="text")
        t.add_column("Tools covered", style="ok")
        names = {tt.name for a in agents for tt in a.tools}
        for rule, values in ref.rule_values.items():
            lits = [v for v in values if "=" in v]
            tools = sorted({v.split("=", 1)[1] for v in lits if v.split("=", 1)[1] in names})
            t.add_row(rule, ", ".join(lits) or ", ".join(values), ", ".join(tools) or Text("-", style="muted"))
        console.print(_section("RULES"))
        console.print(t)
    return EXIT_OK


def act_new(console, ws, inv: Inventory, args) -> int:
    targets: List[Agent]
    if getattr(args, "all", False):
        targets = [a for a in inv.agents if D.needs_policy(a)]
    else:
        a = find_agent(inv, args.agent or args.target)
        if a is None:
            console.print("[high]which agent?[/high] pass --agent ID (see: cslcore venom)")
            return EXIT_USAGE
        targets = [a]
    if not targets:
        console.print("  [ok]every agent is covered or exempt[/ok]; nothing to draft")
        return EXIT_OK
    exemptions = ws.load_exemptions()
    ids = {a.id: D.agent_key(a) for a in inv.agents}
    written = 0
    for a in targets:
        d = D.draft_for(a, exemptions, exec_mode=getattr(args, "exec_mode", "allowlist"), agent_ids=ids)
        console.print()
        console.print(Panel(highlight_csl(d.text), title=Text(f" draft · {a.display_name} ", style="brand"),
                            subtitle=Text(f" {len(d.rules)} rules from {len(a.tools)} tools ", style="muted"),
                            title_align="left", box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
        if d.skipped:
            console.print(Text("  no rule: " + "; ".join(d.skipped), style="muted"))
        if _save_draft(console, ws, d.name, d.text, args.yes):
            written += 1
    if written:
        console.print("\n  [label]NEXT[/label]  [brand]cslcore policy activate <name>[/brand] [muted]after reviewing the limits[/muted]")
    return EXIT_OK


def _draft_name(ref: PolicyRef) -> str:
    return Path(ref.path).stem


def act_edit(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    base = _text(ws, ref, args)
    name = _draft_name(ref)
    path = ws.drafts / f"{name}.csl"
    if ref.status != "draft":
        if ws.plan_only:
            console.print(f"  [muted]--plan-only: would copy to {ws.rel(path)} and open the editor[/muted]")
            return EXIT_OK
        if not path.exists():
            ws.write_text(path, base)
    console.print(f"  opening [brand]{ws.rel(path)}[/brand] in {ws.editor()}")
    ws.open_in_editor(path)
    new = ws.read(path) or ""
    g = verify_text(new)
    _record(ws, str(path), g)
    d = diff_text(base, new, ws.rel(ref.path), ws.rel(path))
    if d:
        console.print(Panel(d, title=Text(" diff ", style="label"), title_align="left", box=box.ROUNDED, border_style="muted", padding=(0, 1)))
    console.print(gate_panel(g, path.name))
    return EXIT_OK if g.ok else EXIT_GATE


def act_extend(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    agents = [find_agent(inv, args.agent)] if args.agent else agents_for(inv, ref)
    agents = [a for a in agents if a is not None]
    if not agents:
        console.print("[high]no agent is linked to this policy[/high]; pass --agent ID")
        return EXIT_USAGE
    base = _text(ws, ref, args)
    from ..analysis.coverage import _covers
    missing = []
    for a in agents:
        tv = tool_variable(ref, a.tools)
        for t in a.tools:
            if t.name.endswith("/*") or t.coverage == "exempt" or _covers(ref, tv, t):
                continue
            if t.risk_class == "READ" or any(m.name == t.name for m in missing):
                if t.risk_class == "READ" and tv and t.name not in ref.vocabulary.get(tv, []):
                    missing.append(t)
                continue
            missing.append(t)
    if not missing:
        console.print("  [ok]every discovered tool is already covered by a rule[/ok]")
        return EXIT_OK
    tv = tool_variable(ref, agents[0].tools) or "tool"
    new, rules = D.extend_text(base, ref.variables, tv, missing)
    console.print(f"  adding {len(rules)} rules for {len(missing)} tools: [head]{', '.join(t.name for t in missing)}[/head]")
    _save_draft(console, ws, _draft_name(ref), new, args.yes, base=base, base_name=ws.rel(ref.path))
    return EXIT_OK


def act_fix(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    label = policy_label(ref)
    items = [d for d in inv.drift if d.policy == label]
    renames = {d.value: d.suggestion for d in items if d.kind == "unknown_value" and d.value and d.suggestion and d.value != d.variable}
    var_renames = {d.variable: d.suggestion for d in items if d.kind == "unknown_value" and d.value == d.variable and d.suggestion}
    coercions = [d for d in items if d.kind == "coercion"]
    if not renames and not var_renames:
        console.print("  [ok]no drift that a policy edit can fix[/ok]")
    else:
        t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
        t.add_column("Policy says")
        t.add_column("Real tool / parameter", style="ok")
        t.add_column("Where", style="muted")
        for old, new in {**renames, **var_renames}.items():
            t.add_row(Text(old, style="high"), new, label)
        console.print(t)
        base = _text(ws, ref, args)
        new = D.fix_text(base, renames, var_renames)
        _save_draft(console, ws, _draft_name(ref), new, args.yes, base=base, base_name=ws.rel(ref.path))
    for d in coercions:
        console.print(f"  [warn]mapping needed[/warn] {d.detail}  [muted](cslcore map --agent {next((a.display_name for a in inv.agents if a.id == d.agent_id), d.agent_id)})[/muted]")
    return EXIT_OK


def act_verify(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    g = verify_text(_text(ws, ref, args))
    if ref.path.startswith(str(ws.root)):
        _record(ws, ref.path, g)
    console.print(gate_panel(g, Path(ref.path).name))
    return EXIT_OK if g.ok else EXIT_GATE


def act_diff(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    draft = ws.drafts / f"{_draft_name(ref)}.csl"
    active = ws.policies / f"{_draft_name(ref)}.csl"
    old = ws.read(active) or (_text(ws, ref, args) if ref.status == "found" else "") or ""
    new = ws.read(draft)
    if new is None:
        console.print(f"  [muted]no draft for {_draft_name(ref)}[/muted]")
        return EXIT_OK
    d = diff_text(old, new, ws.rel(active) if old else "(no active version)", ws.rel(draft))
    console.print(d if d else Text("  draft equals the active version", style="muted"))
    return EXIT_OK


def act_activate(console, ws, inv: Inventory, ref: PolicyRef, args) -> int:
    name = _draft_name(ref)
    draft = ws.drafts / f"{name}.csl"
    text = ws.read(draft)
    if text is None:
        console.print(f"  [high]no draft named {name}[/high] in {ws.rel(ws.drafts)}")
        return EXIT_USAGE
    g = verify_text(text)
    _record(ws, str(draft), g)
    target = ws.policies / f"{name}.csl"
    old = ws.read(target) or (_text(ws, ref, args) if ref.status == "found" else "")
    d = diff_text(old, text, ws.rel(target) if ws.read(target) else (ref.path if old else "(new policy)"), ws.rel(draft))
    if d:
        console.print(Panel(d, title=Text(" diff ", style="label"), title_align="left", box=box.ROUNDED, border_style="muted", padding=(0, 1)))
    console.print(gate_panel(g, draft.name))
    if not g.ok:
        console.print("  [high]not activated[/high]: a draft must pass the gate first")
        return EXIT_GATE
    if ws.plan_only:
        console.print(f"  [muted]--plan-only: would write {ws.rel(target)}[/muted]")
        return EXIT_OK
    if not confirm(console, f"Activate as {ws.rel(target)}?", args.yes, default=False):
        console.print("  [muted]nothing written[/muted]")
        return EXIT_OK
    ws.write_text(target, text)
    ws.remove(draft)
    _record(ws, str(target), g)

    console.print(f"  [ok]active[/ok] {ws.rel(target)}  [muted]hash {g.policy_hash[:16] if g.policy_hash else ''}[/muted]")
    if ref.status == "found" and not ref.path.startswith(str(ws.root)):
        console.print(f"  [warn]note[/warn] the agent still loads {ref.path}; point it at {target} (Venom never writes outside the workspace)")
    linked = agents_for(inv, ref)
    console.print(f"  [label]NEXT[/label]  [brand]cslcore map --agent {linked[0].display_name if linked else name} --test[/brand]")
    return EXIT_OK


def act_bind(console, ws, inv: Inventory, args) -> int:
    import fnmatch

    from ..bindings import Bindings
    from . import binder

    if not args.target:
        console.print("[high]which policy?[/high] cslcore policy bind <policy.csl or name> --agent ID | --match P | --unbound")
        return EXIT_USAGE
    p = Path(args.target)
    ref = None
    if p.suffix == ".csl" and p.exists():
        path = p.resolve()
    else:
        ref = find_policy(ws, inv, args.target)
        if ref is None:
            console.print(f"[high]no policy matches '{args.target}'[/high]")
            return EXIT_USAGE
        path = Path(ref.path)
    if args.action == "unbind":
        for key in Bindings(ws).agents_of(str(path)) if not args.agent else args.agent:
            Bindings(ws).unbind(key)
            console.print(f"  [ok]unbound[/ok] {key}")
        return EXIT_OK
    bound = Bindings(ws).all()
    pool = inv.agents
    chosen = []
    for a in pool:
        key = D.agent_key(a)
        if args.agent and (key in args.agent or a.display_name in args.agent or a.id in args.agent):
            chosen.append(a)
        elif args.match and fnmatch.fnmatch(key, args.match):
            chosen.append(a)
        elif args.unbound and key not in bound and D.needs_policy(a):
            chosen.append(a)
    if not chosen:
        console.print("  [warn]no agent selected[/warn] (use --agent, --match or --unbound)")
        return EXIT_USAGE
    text = _text(ws, ref, args) if ref is not None else (ws.read(path) or "")
    plan = binder.bind(ws, path, chosen, write=False, policy_text=text)
    t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
    for col in ("Agent", "", "Mapping test", "Note"):
        t.add_column(col, overflow="fold")
    for r in plan.results:
        t.add_row(Text(r.agent, style="head"), Text("✓" if r.ok else "✗", style="ok" if r.ok else "high"),
                  f"{r.cases} cases · {r.fail_open} fail-open" if r.cases else "-",
                  Text(("agent_id extended · " if r.edited_policy else "") + r.message, style="muted" if r.ok else "high"))
    console.print(Text.assemble(("  bind ", "label"), (path.name, "head"), (f"  to {len(chosen)} agents", "muted")))
    console.print(t)
    good = [r for r in plan.results if r.ok]
    if not good:
        return EXIT_GATE
    if ws.plan_only:
        console.print("  [muted]--plan-only: nothing written[/muted]")
        return EXIT_OK
    if not confirm(console, f"Bind {len(good)} agents? running ones switch on their next call", args.yes, default=True):
        console.print("  [muted]nothing changed[/muted]")
        return EXIT_OK
    binder.bind(ws, path, [a for a in chosen if D.agent_key(a) in {r.agent for r in good}], write=True, policy_text=text)
    console.print(f"  [ok]bound[/ok] {len(good)} agents to {Bindings(ws).rel(path)}")
    return EXIT_OK if len(good) == len(plan.results) else EXIT_GATE


def cmd_policy(args) -> int:
    console = console_for(args)
    ws = workspace_for(args)
    inv = _load_inventory(args, console)
    if isinstance(getattr(args, "agent", None), list):
        agents_list = args.agent
        args.agent = agents_list[0] if args.action not in ("bind", "unbind") else agents_list
    if args.action in ("bind", "unbind"):
        return act_bind(console, ws, inv, args)
    if args.action == "list":
        return act_list(console, ws, inv)
    if args.action == "new":
        return act_new(console, ws, inv, args)
    ref = find_policy(ws, inv, args.target or args.agent, prefer="draft" if args.action in ("activate", "diff") else None)
    if ref is None:
        console.print(f"[high]no policy matches '{args.target or ''}'[/high]. See: cslcore policy list")
        return EXIT_USAGE
    return {
        "show": lambda: act_show(console, ws, inv, ref, args),
        "edit": lambda: act_edit(console, ws, inv, ref, args),
        "extend": lambda: act_extend(console, ws, inv, ref, args),
        "fix": lambda: act_fix(console, ws, inv, ref, args),
        "verify": lambda: act_verify(console, ws, inv, ref, args),
        "diff": lambda: act_diff(console, ws, inv, ref, args),
        "activate": lambda: act_activate(console, ws, inv, ref, args),
    }[args.action]()
