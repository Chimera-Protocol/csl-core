"""`cslcore studio [policy] [--agent ID] [--new]`."""

from __future__ import annotations

from rich.text import Text

from ..commands import EXIT_OK, EXIT_USAGE, console_for, workspace_for
from ..model import Inventory


def launch(ws, inv=None, path=None, agent=None, use_real_tlc: bool = True) -> str:
    """Open the studio on one policy (or the agent's) and return its closing message.

    Used by setup (policy step), the watch panel ('o') and `cslcore studio`; the caller's
    flow continues once the studio closes."""
    from .app import StudioApp
    from .session import StudioSession

    session = StudioSession(ws, inv)
    if path:
        session.open(str(path), agent=agent)
    elif agent:
        session.open(agent=agent)
    else:
        session.new()
    return StudioApp(session, use_real_tlc=use_real_tlc).run() or ""


def cmd_studio(args) -> int:
    from .app import StudioApp
    from .session import StudioSession

    console = console_for(args)
    ws = workspace_for(args)
    data = ws.latest_inventory()
    inv = Inventory.from_dict(data) if data else None
    session = StudioSession(ws, inv)
    agent = (args.agent[0] if isinstance(args.agent, list) else args.agent) if getattr(args, "agent", None) else None
    if agent and inv is not None and agent not in session.all_agents():
        match = [k for k in session.all_agents() if agent in k]
        if len(match) != 1:
            console.print(f"[high]no single agent matches '{agent}'[/high]; known: {', '.join(session.all_agents()[:12])}")
            return EXIT_USAGE
        agent = match[0]
    if getattr(args, "policy", None):
        session.open(args.policy, agent=agent)
    elif getattr(args, "new", False) or agent:
        session.open(agent=agent) if agent else session.new()
    else:
        items = session.policies()
        session.open(items[0]["path"]) if items else session.new()
    if inv is None:
        console.print(Text("  no scan in this workspace yet: the studio runs without agent checks "
                           "(cslcore venom adds them)", style="#94a3b8"))
    message = StudioApp(session, use_real_tlc=not getattr(args, "mock", False)).run()
    if message:
        console.print(f"  {message}")
    return EXIT_OK
