"""
The rooms of CSL-Core and the moves between them: the reach map and the live panel (watch),
reached from the end of a scan and from setup.

One screen for all rooms: leaving the alternate screen between rooms would flash the shell, so
the rooms share one Live and one key reader. A room draws frames and takes keys; when it wants to
leave it plays its own way out (the map shrinks into the panel's corner, the panel's map grows to
fill the screen) and then names the next room in `exit_to`.

    frame(now, width, height)   the renderable for this moment
    handle(key) -> bool         False quits CSL-Core
    wait() -> float             how long to wait for a key before the next frame
    exit_to                     the next room, once the way out has played
    enter(came_from)            the way in
    external                    something to run outside the screen (the studio, an editor)
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, Optional

ROOMS = ("map", "watch")


def interactive(console) -> bool:
    """Only a person at a terminal is asked where to go next; scripts, pipes and CI never are."""
    from .probe import prompts_disabled

    if prompts_disabled():
        return False
    return bool(console.is_terminal and sys.stdin.isatty() and sys.stdout.isatty())


def run(console, args, start: str, inv=None, came_from: str = "command", made: Optional[Dict[str, object]] = None) -> int:
    """Open `start` ("map" or "watch") and follow the moves between rooms until one quits.
    `came_from` is how it was reached: "command" (on its own), "scan" or "setup". `made` receives
    the rooms, for what the caller says after (the panel counts the checks it saw)."""
    from rich.live import Live

    from .render.keys import Keys

    made = made if made is not None else {}

    def room(name: str):
        if name not in made:
            made[name] = _make(name, console, args, inv)
        return made[name]

    current = start
    r = room(current)
    r.enter(came_from)
    try:
        _loop(console, Keys, Live, room, r, current)
    except KeyboardInterrupt:
        pass
    return 0


def _loop(console, Keys, Live, room, r, current) -> None:
    import time

    with Keys() as keys, Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            key = keys.read(r.wait())
            if key is not None and not r.handle(key):
                break
            if r.exit_to:
                came_from, current = current, r.exit_to
                r.exit_to = None
                r = room(current)
                r.enter(came_from)
            if r.external is not None:
                job, r.external = r.external, None
                live.stop()  # the studio and the editor need the real terminal
                try:
                    job()
                finally:
                    live.start()
            live.update(r.frame(time.monotonic(), console.width, console.height), refresh=True)


def _make(name: str, console, args, inv):
    if name == "map":
        from .commands import workspace_for
        from .render.mapview import MapRoom
        ws = workspace_for(args) if args is not None else None
        if inv is None and ws is not None:
            from .watch import _inventory
            inv = _inventory(ws)
        return MapRoom(inv, console, ws=ws)
    if name == "watch":
        from .watch import WatchRoom
        return WatchRoom(args, console)
    raise ValueError(f"no room {name!r}")


def ask_next(console, choices: Dict[str, str], keys_source: Optional[Callable] = None) -> Optional[str]:
    """One line of choices, one key: returns the chosen key, or None (Enter, Esc, q, anything else)."""
    from rich.text import Text

    line = Text("  ")
    for i, (k, label) in enumerate(choices.items()):
        if i:
            line.append(" · ", style="muted")
        line.append(k, style="brand")
        line.append(f" {label}", style="muted")
    console.print(line)
    if keys_source is None:
        from .render.keys import Keys
        keys_source = Keys
    with keys_source() as keys:
        while True:
            key = keys.read(60.0)
            if key is None:
                continue
            return key if key in choices and key != "q" else None
