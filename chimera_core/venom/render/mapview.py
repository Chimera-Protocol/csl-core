"""
`cslcore venom map`: the reach map, full screen and interactive.

    arrows / Tab   select a node; the panel shows what reaches it and what it reaches
    1 to 9         jump to that agent
    Enter          dive into an agent: its tools around it, and what each tool reaches
    Esc            back out
    s              turn the map into a slowly rotating sphere, and back
    n              names on the map on / off (on by default)
    r              replay the spread
    x              freeze the selected agent: every action blocked, in any mode (x again unfreezes)
    m              switch the selected agent between log and block mode
    w              the live panel (watch): the map shrinks away, the decisions take its place
    q              quit

Read-only: it draws the latest scan of the workspace (or scans when there is none). From the end
of a scan, from setup and from the live panel it opens as one of the rooms (see venom/rooms.py).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..model import Agent, Inventory
from ..reach import ReachGraph, build
from .topo import AGENT, CHAIN, FROZEN, GUARDED, IMPACT_STYLE, INPUT, Sphere, Then, Topo, ViewCanvas, Zoom
from .web import Canvas, _walk
from .words import n as _n

RISK_COLOR = {"READ": "#94a3b8", "WRITE": "#fbbf24", "EXTERNAL": "#7dd3fc", "IDENTITY": "#f0abfc",
              "EXEC": "#f87171", "SPEND": "#fb923c", "DESTRUCTIVE": "#f87171", "UNCLASSIFIED": "#e879f9"}
TOOL_IMPACT = {"EXEC": "impact:exec", "SPEND": "impact:spend", "DESTRUCTIVE": "impact:destroy",
               "EXTERNAL": "impact:publish"}
ZOOM_S = 0.4
GROW_S = 0.6  # arriving from a scan: the map grows out of one point
LEAVE_S = 0.4  # leaving for the live panel: it shrinks into the panel's map corner
SETTLE_S = 0.3  # arriving from the panel's map: it settles to full size
DRIFT = 0.125  # radians per second when nobody touches the globe


@dataclass
class Spot:
    key: str
    label: str
    kind: str  # tool | impact | agent
    x: float
    y: float
    color: str
    open: bool = True  # no rule decides it


class Dive:
    """One agent's world: the agent in the middle, its tools on a ring, and what each tool reaches
    (impacts, other agents it can change) on an outer ring. Spreads outward like the map."""

    def __init__(self, g: ReachGraph, agent: Agent, w: int, h: int) -> None:
        self.g, self.agent, self.w, self.h = g, agent, w, h
        self.W, self.H = w * 2, h * 4
        rng = random.Random(agent.id)
        cx, cy = self.W / 2, self.H / 2
        rx, ry = self.W * 0.24, self.H * 0.27
        tools = sorted(agent.tools, key=lambda t: (t.risk_class == "READ", t.name))[:10]
        self.hidden = max(0, len(agent.tools) - len(tools))
        self.tools: List[Spot] = []
        for i, t in enumerate(tools):
            a = -math.pi / 2 + i / max(1, len(tools)) * 2 * math.pi
            open_ = t.coverage in (None, "unguarded", "wired_no_rule")
            self.tools.append(Spot(t.name, t.name, "tool", cx + math.cos(a) * rx, cy + math.sin(a) * ry,
                                   RISK_COLOR.get(t.risk_class, "#94a3b8"), open_))
        # what each tool reaches
        targets: Dict[str, Spot] = {}
        self.links: List[Tuple[int, str]] = []  # (tool index, target key)
        reach_agents = [e for e in g.out(agent.id) if g.nodes[e.dst].kind == "agent"]
        for i, t in enumerate(tools):
            keys = []
            imp = TOOL_IMPACT.get(t.risk_class)
            if imp and imp in g.nodes and self.tools[i].open:
                keys.append(imp)
            if t.risk_class in ("WRITE", "EXEC", "DESTRUCTIVE") and self.tools[i].open:
                keys += [e.dst for e in reach_agents if t.name in e.evidence]
            for k in keys:
                targets.setdefault(k, None)  # type: ignore[arg-type]
                self.links.append((i, k))
        for imp in ("impact:root", "impact:cloud", "impact:payment"):
            if any(e.dst == imp for e in g.out(agent.id)):
                targets.setdefault(imp, None)  # type: ignore[arg-type]
                self.links.append((-1, imp))
        ox, oy = self.W * 0.45, self.H * 0.44
        keys = list(targets)
        for j, k in enumerate(keys):
            a = -math.pi / 2 + (j + 0.5) / max(1, len(keys)) * 2 * math.pi
            node = g.nodes[k]
            color = IMPACT_STYLE.get(k, "#f87171") if node.kind == "impact" else AGENT
            targets[k] = Spot(k, node.label, node.kind, cx + math.cos(a) * ox, cy + math.sin(a) * oy, color)
        self.targets = targets
        self.center = (cx, cy)
        self.spokes = [_walk(rng, (cx, cy), (s.x, s.y), wobble=0.7) for s in self.tools]
        self.outer = []
        for i, k in self.links:
            src = self.center if i < 0 else (self.tools[i].x, self.tools[i].y)
            self.outer.append((i, k, _walk(rng, src, (targets[k].x, targets[k].y), wobble=0.8)))
        self.dust = [(rng.random() * self.W, rng.random() * self.H) for _ in range(int(self.W * self.H * 0.006))]

    def render(self, st: float, view=None) -> Text:
        c = ViewCanvas(self.w, self.h, view) if view is not None else Canvas(self.w, self.h)
        for x, y in self.dust:
            c.dot(x, y, "#1e293b", -1)
        speed = 70.0
        for s, path in zip(self.tools, self.spokes):
            n = min(len(path), int(st * speed) + 1)
            for i, (x, y) in enumerate(path[:n]):
                tip = n - i
                c.dot(x, y, "bold #fdf4ff" if n < len(path) and tip <= 1 else ("#0f766e" if s.open else "#334155"), 2)
        for i, k, path in self.outer:
            t0 = (len(self.spokes[i]) / speed if i >= 0 else 0.0) + 0.1
            n = min(len(path), int(max(0.0, st - t0) * speed))
            hot = self.targets[k].kind == "impact"
            for j, (x, y) in enumerate(path[:n]):
                c.dot(x, y, "#c026d3" if hot else "#14b8a6", 3)
                if n < len(path) and n - j <= 2:
                    c.dot(x, y, "bold #fdf4ff", 6)
        for s, path in zip(self.tools, self.spokes):
            if st * speed >= len(path):
                c.put(s.x, s.y, "●" if s.open else "◆", s.color if s.open else "#2dd4bf")
                self._label(c, s.x, s.y, s.label, "text" if s.open else "muted")
        for i, k, path in self.outer:
            t0 = (len(self.spokes[i]) / speed if i >= 0 else 0.0) + 0.1
            if st - t0 >= len(path) / speed:
                sp = self.targets[k]
                c.put(sp.x, sp.y, "▲" if sp.kind == "impact" else "●", f"bold {sp.color}")
                self._label(c, sp.x, sp.y, sp.label, sp.color)
        cx, cy = self.center
        c.ring(cx, cy, 2.4 + math.sin(st * 4) * 0.6, CHAIN, 5, step=st)
        c.put(cx, cy, "◉", "bold #fdf4ff")
        return c.render()

    def _label(self, c, x: float, y: float, text: str, style: str) -> None:
        """A short name beside a node, outward from the centre, kept on the canvas."""
        label = text if len(text) <= 13 else text[:12] + "…"
        right = x >= self.W / 2
        lx = x + 2.5 if right else x - 2.5 - 2 * len(label)
        lx = min(max(0.0, lx), self.W - 2 * len(label) - 1)
        for q, ch in enumerate(label):
            c.put(lx + 2 * q, y, ch, style)

    def panel(self) -> List[Text]:
        a = self.agent
        rows = [Text(a.display_name, style="bold #f0abfc"),
                Text(f"{a.kind} · {a.state}" + (f" · runs as {a.process_user}" if a.process_user else ""), style="muted"),
                Text("")]
        rows.append(Text("TOOLS", style="label"))
        rows.append(Text.assemble(("●", RISK_COLOR["EXEC"]), (" commands ", "muted"), ("●", RISK_COLOR["WRITE"]),
                                  (" writes ", "muted"), ("●", RISK_COLOR["EXTERNAL"]), (" outside ", "muted"),
                                  ("●", RISK_COLOR["READ"]), (" reads ", "muted"), ("◆", "#2dd4bf"), (" a rule decides", "muted")))
        for s in self.tools:
            rows.append(Text.assemble(("● " if s.open else "◆ ", s.color if s.open else "#2dd4bf"), (s.label, "text"),
                                      ("  no rule" if s.open else "  a rule decides", "muted")))
        if self.hidden:
            rows.append(Text(f"  + {self.hidden} more", style="muted"))
        reach = [sp for sp in self.targets.values()]
        if reach:
            rows += [Text(""), Text("REACHES", style="label")]
            for sp in reach:
                rows.append(Text.assemble(("▲ " if sp.kind == "impact" else "● ", sp.color), (sp.label, "text")))
        return rows


class MapView:
    """State and frames of the interactive map; `handle(key)` returns False to quit."""

    def __init__(self, inv: Inventory, width: int, height: int, seed: str = "map", ws=None) -> None:
        self.inv = inv
        self.ws = ws  # the workspace whose controls x and m change (None: the map only shows)
        self.dims = (width, height, seed)
        self.args = None  # set by the room: what a guard-it-now job scans with
        self.console = None
        self.g = build(inv)
        self.w = max(40, min(width - 46, 96))
        self.h = max(13, min(height - 9, 34))
        self.topo = Topo(self.g, w=self.w, h=self.h, max_agents=14, seed=seed)
        self.order = ([p.id for p in self.topo.placed.values() if p.kind == "input"]
                      + [p.id for p in sorted((q for q in self.topo.placed.values() if q.kind == "agent"), key=lambda q: q.number or 0)]
                      + [p.id for p in self.topo.placed.values() if p.kind == "impact"])
        self.sel = 0
        self.t0 = time.monotonic()
        self.spread0 = self.t0
        self.mode = "map"  # map | dive
        self.sphere = False
        self.zoom: Optional[Tuple[float, str]] = None  # (start, "in" | "out")
        self.dive: Optional[Dive] = None
        self.labels = True  # agent names on the canvas
        self.angle = 0.0  # the sphere's turn; it eases toward the selected node
        self.last_frame = self.t0
        self.idle0 = self.t0  # last key: the globe drifts on its own only after a while
        self.back0: Optional[float] = None  # when the map zooms back out after a dive
        # rooms: the way in, the way out, and where to go once it has played
        self.arrive: Optional[Tuple[float, str]] = None
        self.leave: Optional[Tuple[float, str]] = None
        self.exit_to: Optional[str] = None
        self.external = None
        # control, as in the live panel: (agent key, its control, whether a policy guards it)
        self.control: Dict[str, tuple] = {}
        self.pending: Optional[Tuple[str, str, str]] = None  # (action, agent key, question)
        self.message: Optional[Tuple[str, str, float]] = None  # (text, style, when)
        self.synced = 0.0

    # -- rooms ------------------------------------------------------------------------------
    def enter(self, came_from: str) -> None:
        """`command`: opened on its own, the spread plays as always. `scan`: the spread was just
        seen, the map grows out of one point complete. `watch`: back from the panel's map."""
        now = time.monotonic()
        self.leave = None
        if came_from == "command":
            self.spread0, self.arrive = now, None
            return
        self.spread0 = now - 99.0
        self.arrive = (now, came_from)

    def wait(self) -> float:
        return 0.04

    # -- input ------------------------------------------------------------------------------
    def handle(self, key: str) -> bool:
        now = time.monotonic()
        self.idle0 = now
        if self.pending is not None:  # a question is open: y does it, any other key cancels
            action, target, _q = self.pending
            self.pending = None
            if key in ("y", "Y"):
                self._apply(action, target)
            else:
                self.message = ("cancelled", "muted", now)
            return True
        if key in ("q", "ctrl-c"):
            return False
        if self.leave is not None or (self.zoom and now - self.zoom[0] < ZOOM_S):
            return True
        if self.mode == "dive":
            if key in ("esc", "backspace", "enter"):
                self.zoom = (now, "out")
            return True
        if key in ("down", "right", "tab", "j", "l") and self.order:
            self.sel = (self.sel + 1) % len(self.order)
        elif key in ("up", "left", "k", "h") and self.order:
            self.sel = (self.sel - 1) % len(self.order)
        elif key == "enter" and self.selected_agent() is not None:
            self.zoom = (now, "in")
        elif key in ("s", "S"):
            self.sphere = not self.sphere
        elif key in ("n", "N", "l"):
            self.labels = not self.labels
        elif key.isdigit() and key != "0":  # jump to the agent with that number
            target = next((p.id for p in self.topo.placed.values() if p.number == int(key)), None)
            if target in self.order:
                self.sel = self.order.index(target)
        elif key == "r":
            self.spread0 = now
        elif key in ("w", "W"):
            self.leave = (now, "watch")
        elif key in ("x", "X", "m"):
            self._control_key(key.lower(), now)
        return True

    # -- control: freeze and mode, as in the live panel ----------------------------------------
    def _sync(self, now: float, force: bool = False) -> None:
        """Read the controls (about once a second, so changes made elsewhere show here too)."""
        if self.ws is None or (not force and now - self.synced < 1.0):
            return
        from ..bindings import Bindings
        from ..controls import Controls
        from ..policy.draft import agent_key

        self.synced = now
        controls, bound = Controls(self.ws), set(Bindings(self.ws).all())
        marks = {}
        for a in self.inv.agents:
            k = agent_key(a)
            ctl = controls.get(k)
            in_path = a.guard.status != "none"  # what the scan saw: a guard in its code, or a cslcore hook
            self.control[a.id] = (k, ctl, k in bound, in_path)
            if not in_path:
                continue  # nothing would stop it: no ice, no guard ring
            if ctl.disabled:
                marks[a.id] = "frozen"
            elif ctl.mode == "block" and k in bound:
                marks[a.id] = "block"
        self.topo.marks = marks

    def _control_key(self, key: str, now: float) -> None:
        a = self.selected_agent()
        if a is None:
            self.message = ("select an agent first (↑↓ or 1-9)", "muted", now)
            return
        if self.ws is None:
            self.message = ("open the map from a workspace to change controls", "muted", now)
            return
        self._sync(now, force=True)
        k, ctl, bound, in_path = self.control[a.id]
        if not in_path:  # freezing it would stop nothing: offer to put it under a guard first
            if self.args is None:
                self.message = (f"{a.display_name} is not wired: cslcore wire --agent {k}", "warn", now)
                return
            what = "wire it" if bound else "write its policy and wire it"
            self.pending = ("guard", a.id, f"{a.display_name} is not wired: freezing it would stop nothing. "
                                           f"Put it under a guard now ({what}; you see each step)?")
            return
        if key == "x":
            if ctl.disabled:
                self._apply("enable", k)
            else:
                self.pending = ("disable", k, f"Freeze {a.display_name}? every action is blocked, in any mode, until x again")
        else:
            to = "block" if ctl.mode == "log" else "log"
            q = (f"Switch {a.display_name} to BLOCK mode? policy violations will be blocked" if to == "block"
                 else f"Switch {a.display_name} to LOG mode? nothing will be blocked, only recorded")
            self.pending = (f"mode_{to}", k, q)

    def _apply(self, action: str, target: str) -> None:
        from ..controls import Controls

        controls, now = Controls(self.ws), time.monotonic()
        if action == "guard":
            self.external = lambda: self._guard_it(target)
            return
        if action == "disable":
            controls.set_disabled(target, True)
            self.message = (f"{target} FROZEN: every action is blocked (x unfreezes)", "high", now)
        elif action == "enable":
            controls.set_disabled(target, False)
            self.message = (f"{target} unfrozen", "ok", now)
        elif action.startswith("mode_"):
            mode = action[5:]
            controls.set_mode(target, mode)
            self.message = (f"{target} → {mode.upper()} mode (next call, no restart)", "ok" if mode == "log" else "brand", now)
        self._sync(now, force=True)

    def _guard_it(self, agent_id: str) -> None:
        """Outside the screen: put the agent under a guard (policy, wiring), then freeze it, as asked."""
        from rich.prompt import Prompt

        from ..controls import Controls
        from ..watch import _inventory
        from ..wire_cmd import guard_one

        agent = next(a for a in self.inv.agents if a.id == agent_id)
        console = self.console
        ok = guard_one(console, self.args, self.ws, agent)
        inv = _inventory(self.ws) or self.inv
        self.reload(inv)
        self._sync(time.monotonic(), force=True)
        key = self.control.get(agent_id, (None,))[0]
        if ok and key:
            Controls(self.ws).set_disabled(key, True)
            console.print(f"  [high]{agent.display_name} FROZEN[/high] [muted]every action is blocked until x again[/muted]")
            self.message = (f"{agent.display_name} is guarded now, and FROZEN (x unfreezes)", "high", time.monotonic())
        else:
            self.message = (f"{agent.display_name} is still not wired", "warn", time.monotonic())
        Prompt.ask("  [muted]Enter: back to the map[/muted]", default="", show_default=False, console=console)
        self._sync(time.monotonic(), force=True)

    def reload(self, inv: Inventory) -> None:
        """The map of a new scan (after wiring), keeping what was selected and how it is shown."""
        sel = self.selected()
        keep = (self.sphere, self.labels, self.angle, self.args, self.console)
        width, height, seed = self.dims
        MapView.__init__(self, inv, width, height, seed=seed, ws=self.ws)  # not a room's own __init__
        self.sphere, self.labels, self.angle, self.args, self.console = keep
        self.spread0 -= 99.0
        if sel in self.order:
            self.sel = self.order.index(sel)

    def selected(self) -> Optional[str]:
        return self.order[self.sel] if self.order else None

    def selected_agent(self) -> Optional[Agent]:
        sid = self.selected()
        return next((a for a in self.inv.agents if a.id == sid), None)

    # -- frames -----------------------------------------------------------------------------
    def _facing(self) -> float:
        """The sphere angle that brings the selected node to the front."""
        p = self.topo.placed.get(self.selected() or "")
        return -(p.x / self.topo.W - 0.5) * 2 * math.pi if p is not None else 0.0

    def _turn(self, now: float) -> None:
        dt = min(0.2, now - self.last_frame)
        self.last_frame = now
        if now - self.idle0 > 4.0:  # left alone, it drifts slowly: a full turn in about 50 seconds
            self.angle += DRIFT * dt
            return
        diff = (self._facing() - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle += diff * min(1.0, dt * 2.5)  # otherwise it turns the selected node to the front

    def _map_view(self, zoom: float = 1.0):
        """The camera for the map: flat or the sphere, zoomed around the selected node."""
        p = self.topo.placed.get(self.selected() or "")
        W, H = self.topo.W, self.topo.H
        if self.sphere:
            globe = Sphere(self.angle, W, H)
            if zoom == 1.0 or p is None:
                return globe
            X, Y, _ = globe(p.x, p.y)
            return Then(globe, Zoom(X, Y, zoom, W, H))
        if zoom == 1.0 or p is None:
            return None
        return Zoom(p.x, p.y, zoom, W, H)

    def _passage(self, now: float):
        """The camera of a way in or out, or None. Arriving from a scan the map grows out of one
        point; leaving for the panel it shrinks toward the panel's map, top right; arriving from
        the panel's map it settles from slightly too close."""
        W, H = self.topo.W, self.topo.H
        base = Sphere(self.angle, W, H) if self.sphere else None
        cam = None
        if self.leave is not None:
            k = min(1.0, (now - self.leave[0]) / LEAVE_S)
            if k >= 1.0:
                self.exit_to, self.leave = self.leave[1], None
            e = k * k * (3 - 2 * k)
            scale = 1 - 0.55 * e
            cam = Zoom(W / 2 - e * 0.25 * W / scale, H / 2 + e * 0.15 * H / scale, scale, W, H)
        elif self.arrive is not None:
            start, came_from = self.arrive
            span = SETTLE_S if came_from == "watch" else GROW_S
            k = min(1.0, (now - start) / span)
            if k >= 1.0:
                self.arrive = None
                return None
            e = 1 - (1 - k) ** 3
            scale = 1.35 - 0.35 * e if came_from == "watch" else 0.04 + 0.96 * e
            cam = Zoom(W / 2, H / 2, scale, W, H)
        if cam is None:
            return None
        return Then(base, cam) if base is not None else cam

    def canvas(self, now: float) -> Text:
        passage = self._passage(now)
        if passage is not None and self.mode == "map":
            self._turn(now)
            return self.topo.render(now - self.t0, spread_t=now - self.spread0, view=passage,
                                    selected=self.selected(), labels=self.labels)
        t = now - self.t0
        st = now - self.spread0
        self._turn(now)
        sel = self.selected()
        if self.zoom:
            start, direction = self.zoom
            k = min(1.0, (now - start) / ZOOM_S)
            if direction == "in":
                if k < 1:
                    return self.topo.render(t, spread_t=st, view=self._map_view(1 + 5 * k * k), selected=sel,
                                            labels=self.labels)
                if self.mode != "dive":
                    a = self.selected_agent()
                    self.dive = Dive(self.g, a, self.w, self.h) if a is not None else None
                    self.mode = "dive" if self.dive else "map"
                    self.dive_t0 = now
                self.zoom = None
            else:
                if k < 1 and self.dive is not None:
                    return self.dive.render(99.0, view=Zoom(self.dive.W / 2, self.dive.H / 2, max(0.05, 1 - k),
                                                            self.dive.W, self.dive.H))
                self.mode, self.dive, self.zoom = "map", None, None
                self.back0 = now  # the map comes back, zooming out from the node
        if self.mode == "dive" and self.dive is not None:
            k = min(1.0, (now - self.dive_t0) / 0.3)
            view = Zoom(self.dive.W / 2, self.dive.H / 2, 0.4 + 0.6 * k, self.dive.W, self.dive.H) if k < 1 else None
            return self.dive.render(now - self.dive_t0, view=view)
        scale = 1.0
        if self.back0 is not None:
            k = (now - self.back0) / ZOOM_S
            if k < 1:
                scale = 1 + 5 * (1 - k) ** 2
            else:
                self.back0 = None
        return self.topo.render(t, spread_t=st if st >= 0 else None, view=self._map_view(scale), selected=sel,
                                labels=self.labels)

    def side(self) -> Group:
        if self.mode == "dive" and self.dive is not None:
            return Group(*self.dive.panel())
        rows = self.topo.legend(99.0)
        rows.append(Text.assemble(("◇", INPUT), (" input  ", "muted"), ("●", AGENT), (" agent  ", "muted"),
                                  ("▲", "#f87171"), (" at stake  ", "muted"), ("━", CHAIN), (" strongest chain", "muted")))
        if self.topo.marks:
            rows.append(Text.assemble(("●", f"bold {FROZEN}"), (" frozen  ", "muted"), ("○", GUARDED), (" block mode", "muted")))
        sid = self.selected()
        if sid in self.g.nodes:
            node = self.g.nodes[sid]
            rows += [Text(""), Text("SELECTED", style="label"), Text(node.label, style="bold #fde047")]
            if sid in self.control:
                _k, ctl, bound, in_path = self.control[sid]
                if not in_path:
                    rows.append(Text("not wired: nothing stops it yet" + ("" if bound else ", no policy"), style="warn"))
                    rows.append(Text("x puts it under a guard", style="muted"))
                elif ctl.disabled:
                    rows.append(Text("frozen: every call is blocked", style=f"bold {FROZEN}"))
                else:
                    rows.append(Text("block mode: its rules decide every call" if ctl.mode == "block"
                                     else "wired, log mode: recorded, nothing stopped yet",
                                     style=GUARDED if ctl.mode == "block" else "muted"))
            ins = [e for e in self.g.edges if e.dst == sid]
            outs = [e for e in self.g.edges if e.src == sid]
            for title, edges, other in (("reached by", ins, "src"), ("reaches", outs, "dst")):
                if edges:
                    rows.append(Text(title, style="muted"))
                    for e in edges[:5]:
                        rows.append(Text.assemble(("  ", ""), (self.g.nodes[getattr(e, other)].label, "text")))
                    if len(edges) > 5:
                        rows.append(Text(f"  + {len(edges) - 5} more", style="muted"))
        return Group(*rows)

    def frame(self, now: float, width: int) -> Panel:
        self._sync(now)
        g = self.g
        head = Text.assemble((f"{len(g.chains)} reach chain{'s' if len(g.chains) != 1 else ''}", "bold #f0abfc"),
                             (f" · {_n(sum(1 for n in g.nodes.values() if n.kind == 'agent'), 'agent')}", "muted"),
                             ("   3D" if self.sphere and self.mode == "map" else "", "brand"))
        grid = Table.grid(padding=(0, 2))
        grid.add_column(width=self.w + 1, no_wrap=True)
        grid.add_column(width=max(24, width - self.w - 8), overflow="ellipsis")
        grid.add_row(self.canvas(now), self.side())
        chain = Text("")
        hero = self.topo.chain
        if hero is not None and self.mode == "map":
            chain = Text.assemble(("REACH CHAIN  " if g.top is not None else "STRONGEST EXPOSURE  ", "label"),
                                  ("  →  ".join(g.nodes[n].label for n in hero.nodes), "bold #f0abfc"))
            held = [g.nodes[n].label for n in hero.nodes if self.topo.marks.get(n) == "frozen"]
            if held:
                chain.append(f"   held: {', '.join(held)} frozen", style=f"bold {FROZEN}")
        if self.pending is not None:
            keys = Text.assemble((self.pending[2], "warn"), ("   [y/n]", "head"))
        elif self.message is not None and now - self.message[2] < 4.0:
            keys = Text(self.message[0], style=self.message[1])
        else:
            keys = Text("Esc back · q quit" if self.mode == "dive"
                        else "↑↓ or 1-9 select · Enter dive in · x freeze · m mode · s sphere · n names · r replay · "
                             "w watch · q quit", style="muted")
        body = Group(head, Text(""), grid, Text(""), chain, keys)
        title = Text.assemble((" CSL-Core Venom ", "brand"), ("· reach map ", "muted"))
        return Panel(body, title=title, title_align="left", box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))


class MapRoom(MapView):
    """The map as a room (see venom/rooms.py)."""

    def __init__(self, inv: Inventory, console, seed: Optional[str] = None, ws=None, args=None) -> None:
        from .. import VENOM_VERSION

        super().__init__(inv, console.width, console.height, seed=seed or VENOM_VERSION, ws=ws)
        self.args, self.console = args, console

    def frame(self, now: float, width: int, height: int = 0) -> Panel:  # type: ignore[override]
        return super().frame(now, width)


def run(console, inv: Inventory, seed: str = "map", once: bool = False, args=None) -> int:
    import sys

    if once or not sys.stdin.isatty():
        view = MapView(inv, console.width, console.height, seed=seed)
        view.spread0 -= 99.0
        console.print(view.frame(time.monotonic(), console.width))
        return 0
    from .. import rooms

    return rooms.run(console, args, "map", inv=inv, came_from="command")
