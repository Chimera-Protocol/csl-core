"""`cslcore exempt`, `cslcore mode` and `cslcore watch`."""

from __future__ import annotations


from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import exemptions as ex
from .commands import EXIT_OK, EXIT_USAGE, console_for, workspace_for
from .model import Exemption


def _agent_known(ws, agent: str) -> bool:
    data = ws.latest_inventory() or {}
    ids = {a.get("id") for a in data.get("agents", [])} | {a.get("display_name") for a in data.get("agents", [])}
    return agent == "*" or agent in ids or not ids


def _render_list(console, items, today) -> None:
    if not items:
        console.print("  [muted]No exemptions.[/muted] Add one with: [brand]cslcore exempt add <agent> --reason ... --approved-by ...[/brand]")
        return
    t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
    for col in ("#", "Agent", "Scope", "Reason", "Approved by", "Expires", "Status"):
        t.add_column(col, overflow="fold")
    for i, e in enumerate(items, 1):
        status = e.status
        style = "ok" if status == "approved" else "warn"
        if ex.expired(e, today):
            status, style = "expired", "high"
        t.add_row(str(i), e.agent, e.scope + (f": {e.tool}" if e.tool else ""), e.reason or "-", e.approved_by or "-",
                  e.expires or "-", Text(status, style=style))
    console.print(t)


def cmd_exempt(args) -> int:
    console = console_for(args)
    ws = workspace_for(args)
    items = ws.load_exemptions()
    today = ws.today()

    if args.action == "list":
        _render_list(console, items, today)
        return EXIT_OK

    if args.action == "add":
        if not args.target:
            console.print("[high]exempt add needs an agent id[/high] (or '*' with --scope tool)")
            return EXIT_USAGE
        e = Exemption(agent=args.target, scope=args.scope, tool=args.tool, reason=args.reason,
                      approved_by=args.approved_by, expires=args.expires,
                      status="proposed" if args.propose else "approved")
        try:
            if e.status == "approved":
                ex.validate(e)
            elif not e.reason:
                raise ex.ExemptionError("an exemption needs --reason (why this agent or tool is trusted)")
        except ex.ExemptionError as err:
            console.print(f"[high]Not recorded:[/high] {err}")
            return EXIT_USAGE
        if not _agent_known(ws, e.agent):
            console.print(f"[warn]note:[/warn] '{e.agent}' is not in the latest inventory; recorded as given")
        items.append(e)
        ws.save_exemptions(items)
        verb = "proposed" if e.status == "proposed" else "approved"
        console.print(Panel(Text.assemble(
            (f"Exemption {verb}: ", "head"), (e.agent + (f" / {e.tool}" if e.tool else ""), "brand"),
            ("\nreason: ", "muted"), (e.reason or "", "text"),
            ("\nIt is visible in every report under \"Exempted by operator\". Generated policies encode it as ", "muted"),
            (f'agent_id != "{e.agent}"' if e.scope == "agent" else f"no rule for {e.tool}", "code"), (".", "muted"),
        ), box=box.ROUNDED, border_style="exempt", padding=(0, 1)))
        return EXIT_OK

    try:
        idx = int(args.target or "0") - 1
        target = items[idx]
        if idx < 0:
            raise IndexError
    except (ValueError, IndexError):
        console.print("[high]give the exemption number from `cslcore exempt list`[/high]")
        return EXIT_USAGE

    if args.action == "approve":
        target.approved_by = args.approved_by or target.approved_by
        target.status = "approved"
        try:
            ex.validate(target)
        except ex.ExemptionError as err:
            console.print(f"[high]Not approved:[/high] {err}")
            return EXIT_USAGE
        ws.save_exemptions(items)
        console.print(f"  [ok]approved[/ok] exemption {idx + 1} for {target.agent}")
        return EXIT_OK

    if args.action == "remove":
        items.pop(idx)
        ws.save_exemptions(items)
        console.print(f"  [ok]removed[/ok] exemption {idx + 1}")
        return EXIT_OK
    return EXIT_USAGE


def _control_table(console, controls) -> None:
    items = controls.all()
    if not items:
        console.print("  [muted]no agent is wired yet: cslcore setup[/muted]")
        return
    t = Table(box=box.SIMPLE_HEAD, header_style="label", pad_edge=False, show_edge=False, border_style="muted")
    for col in ("Agent", "Mode", "Status", "Disabled tools"):
        t.add_column(col)
    for key, c in items.items():
        status = Text("DISABLED", style="high") if c.disabled else Text("active", style="ok")
        t.add_row(Text(key, style="head"), Text(c.mode, style="warn" if c.mode == "log" else "brand"), status,
                  Text(", ".join(sorted(c.disabled_tools)) or "-", style="high" if c.disabled_tools else "muted"))
    console.print(t)


def known_agents(ws, controls) -> list:
    """Agent keys from the latest scan, the control plane and the decision logs."""
    from .policy.draft import agent_key
    from .model import Inventory

    keys = set(controls.all())
    data = ws.latest_inventory()
    if data:
        keys |= {agent_key(a) for a in Inventory.from_dict(data).agents}
    keys |= {p.stem for p in ws.decision_logs()}
    setup = (ws.load_state().get("setup") or {}).get("agents") or {}
    keys |= {st.get("key") for st in setup.values() if st.get("key")}
    return sorted(k for k in keys if k)


def _bulk_mode(console, ws, controls, args) -> int:
    import fnmatch

    from .policy.workbench import confirm

    if not args.mode:
        console.print("[high]give the mode:[/high] cslcore mode --all log   or   cslcore mode --match 'pay-*' block")
        return EXIT_USAGE
    agents = known_agents(ws, controls)
    if getattr(args, "match", None):
        hit = [k for k in agents if fnmatch.fnmatch(k, args.match)]
        if not hit:
            console.print(f"  [warn]no agent matches {args.match}[/warn]")
            return EXIT_USAGE
        controls.set_many(hit, args.mode)
        console.print(f"  [ok]{len(hit)} agents[/ok] → [brand]{args.mode}[/brand]: {', '.join(hit)}")
        return EXIT_OK
    losing = [k for k in agents if controls.get(k).mode == "block"] if args.mode == "log" else []
    question = f"Switch all {len(agents)} agents to {args.mode.upper()}?"
    if losing:
        question += f" {len(losing)} of them stop blocking and only record ({', '.join(losing[:5])}{' ...' if len(losing) > 5 else ''})."
    if not confirm(console, question, bool(getattr(args, "yes", False)), default=args.mode == "log" and not losing):
        console.print("  [muted]nothing changed[/muted]")
        return EXIT_OK
    controls.set_all(args.mode)
    console.print(f"  [ok]all agents[/ok] → [brand]{args.mode}[/brand] (new agents too). "
                  "[muted]Single agents can still be switched: cslcore mode --agent ID block[/muted]")
    return EXIT_OK


def cmd_mode(args) -> int:
    from .controls import Controls

    console = console_for(args)
    ws = workspace_for(args)
    controls = Controls(ws)
    if getattr(args, "all", False) or getattr(args, "match", None):
        return _bulk_mode(console, ws, controls, args)
    if not args.agent:
        _control_table(console, controls)
        return EXIT_OK
    agent = args.agent
    changed = False
    if args.mode:
        controls.set_mode(agent, args.mode)
        note = ("nothing is blocked; every decision is recorded as ALLOW or WOULD BLOCK" if args.mode == "log"
                else "policy violations are blocked")
        console.print(f"  [ok]{agent}[/ok] → [brand]{args.mode}[/brand] mode: {note}")
        changed = True
    if args.disable or args.enable:
        controls.set_disabled(agent, bool(args.disable))
        console.print(f"  [{'high' if args.disable else 'ok'}]{agent} {'DISABLED' if args.disable else 'enabled'}[/]"
                      + (": every action is blocked, in any mode" if args.disable else ""))
        changed = True
    for tool in args.disable_tool:
        controls.set_tool(agent, tool, True)
        console.print(f"  [high]{agent} / {tool} disabled[/high]")
        changed = True
    for tool in args.enable_tool:
        controls.set_tool(agent, tool, False)
        console.print(f"  [ok]{agent} / {tool} enabled[/ok]")
        changed = True
    if changed:
        console.print("  [muted]applies to the next tool call of running agents (no restart); recorded in .csl/venom/audit.jsonl[/muted]")
    else:
        c = controls.get(agent)
        console.print(f"  {agent}: [brand]{c.mode}[/brand]" + ("  [high]DISABLED[/high]" if c.disabled else "")
                      + (f"  disabled tools: {', '.join(sorted(c.disabled_tools))}" if c.disabled_tools else ""))
    return EXIT_OK


def cmd_watch(args) -> int:
    from .watch import run_watch
    return run_watch(args)
