"""
The reach map: the real nodes and edges of the reach graph on a braille canvas.

Inputs sit on the left, impacts on the right, agents in between at the depth an input reaches
them. Edges are drawn as soft curves; the strongest chain lights up hop by hop with a pulse of
light travelling along it. Pure drawing: `Topo.render(...)` returns a Text for one instant.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rich.text import Text

from ..reach import Chain, ReachGraph
from .web import Canvas, _branch, _walk

EDGE = "#134e4a"
EDGE_LIT = "#14b8a6"
CHAIN = "#e879f9"
CHAIN_HOT = "bold #fdf4ff"
INPUT = "#fbbf24"
AGENT = "#5eead4"
IMPACT_STYLE = {
    "impact:root": "#f87171", "impact:spend": "#fb923c", "impact:payment": "#fb923c", "impact:cloud": "#c084fc",
    "impact:exec": "#f87171", "impact:destroy": "#f87171", "impact:publish": "#7dd3fc",
}
Point = Tuple[float, float]


class Zoom:
    """Camera: scale the map around a focus point (dive in / out)."""

    def __init__(self, fx: float, fy: float, scale: float, W: float, H: float) -> None:
        self.fx, self.fy, self.scale, self.W, self.H = fx, fy, scale, W, H

    def __call__(self, x: float, y: float) -> Tuple[float, float, float]:
        return self.W / 2 + (x - self.fx) * self.scale, self.H / 2 + (y - self.fy) * self.scale, 1.0


class Sphere:
    """Camera: the map wrapped onto a slowly turning sphere (orthographic projection). Depth < 0 is
    the far side, drawn dim behind the near side."""

    def __init__(self, angle: float, W: float, H: float, tilt: float = 0.35) -> None:
        self.angle, self.W, self.H, self.tilt = angle, W, H, tilt
        self.R = min(H, W) * 0.46

    def __call__(self, x: float, y: float) -> Tuple[float, float, float]:
        lon = (x / self.W - 0.5) * 2 * math.pi + self.angle  # the map's centre faces the viewer at angle 0
        lat = (y / self.H - 0.5) * math.pi * 0.85
        cx, cy, cz = math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)
        cy, cz = cy * math.cos(self.tilt) - cz * math.sin(self.tilt), cy * math.sin(self.tilt) + cz * math.cos(self.tilt)
        return self.W / 2 + cx * self.R, self.H / 2 + cy * self.R, cz  # braille dots are square


class Then:
    """Two cameras in a row: the first decides depth (a sphere), the second moves the image (a zoom)."""

    def __init__(self, first, second) -> None:
        self.first, self.second, self.R = first, second, getattr(first, "R", 0)

    def __call__(self, x: float, y: float) -> Tuple[float, float, float]:
        X, Y, d = self.first(x, y)
        X2, Y2, _ = self.second(X, Y)
        return X2, Y2, d


class ViewCanvas(Canvas):
    """A canvas that sends every point through a camera; the far side of a sphere stays dim."""

    def __init__(self, w: int, h: int, view) -> None:
        super().__init__(w, h)
        self.view = view

    def dot(self, x: float, y: float, style: str, prio: int = 0) -> None:
        X, Y, depth = self.view(x, y)
        if depth < 0:
            super().dot(X, Y, "#1e293b" if prio < 3 else "#334155", -1)
        else:
            super().dot(X, Y, style, prio)

    def put(self, x: float, y: float, ch: str, style: str) -> None:
        X, Y, depth = self.view(x, y)
        if depth >= -0.15:
            super().put(X, Y, ch, style if depth >= 0.15 else "#475569")


@dataclass
class Placed:
    id: str
    kind: str
    label: str
    x: float
    y: float
    number: Optional[int] = None  # agents: their number in the legend


def _depths(g: ReachGraph, agents: Sequence[str]) -> Dict[str, int]:
    """How many agent hops an input needs to reach each agent (1 = directly)."""
    depth: Dict[str, int] = {}
    queue = deque()
    for e in g.edges:
        if g.nodes[e.src].kind == "input" and e.dst in agents and e.dst not in depth:
            depth[e.dst] = 1
            queue.append(e.dst)
    while queue:
        a = queue.popleft()
        for e in g.out(a):
            if e.dst in agents and e.dst not in depth:
                depth[e.dst] = depth[a] + 1
                queue.append(e.dst)
    deepest = max(depth.values(), default=1)
    for a in agents:
        depth.setdefault(a, deepest + 1 if depth else 1)
    return depth


class Topo:
    def __init__(self, g: ReachGraph, w: int = 40, h: int = 13, max_agents: int = 9, seed: str = "reach") -> None:
        self.g = g
        self.w, self.h = w, h
        self.W, self.H = w * 2, h * 4
        rng = random.Random(seed)
        self.chain: Optional[Chain] = g.top
        connected = {e.src for e in g.edges} | {e.dst for e in g.edges}
        in_chains = [n for c in g.chains for n in c.nodes]
        agents = sorted((n for n in g.nodes.values() if n.kind == "agent" and n.id in connected),
                        key=lambda n: (-in_chains.count(n.id), -n.weight, n.label))[:max_agents]
        spine = list(self.chain.nodes) if self.chain else []
        agents = [g.nodes[n] for n in spine if g.nodes[n].kind == "agent"] + [a for a in agents if a.id not in spine]
        agents = agents[:max_agents]
        keep = {a.id for a in agents}
        inputs = [n for n in g.nodes.values() if n.kind == "input"
                  and any(e.src == n.id and e.dst in keep for e in g.edges)][:4]
        impacts = sorted((n for n in g.nodes.values() if n.kind == "impact"
                          and any(e.dst == n.id and e.src in keep for e in g.edges)), key=lambda n: -n.weight)[:5]
        self.hidden_agents = max(0, sum(1 for n in g.nodes.values() if n.kind == "agent" and n.id in connected) - len(agents))
        self.placed: Dict[str, Placed] = {}
        mid = self.H / 2
        # the spine: the strongest chain as one clear stroke through the middle, left to right
        if spine:
            xs = [3.0 + i * (self.W - 7.0) / (len(spine) - 1) for i in range(len(spine))]
            for nid, x in zip(spine, xs):
                n = g.nodes[nid]
                self.placed[nid] = Placed(nid, n.kind, n.label, x, mid + (rng.random() - 0.5) * 3 if n.kind == "agent" else mid)
        # everything else above and below it
        self._side([n for n in inputs if n.id not in self.placed], 3.0, rng)
        self._side([n for n in impacts if n.id not in self.placed], self.W - 4.0, rng)
        depth = _depths(g, [a.id for a in agents])
        deepest = max(depth.values(), default=1)
        rest = [a for a in agents if a.id not in self.placed]
        for i, a in enumerate(rest):
            x = 12 + (self.W - 26) * (depth.get(a.id, 1) - 1) / max(1, deepest - 1) if deepest > 1 else self.W / 2
            self._side([a], x + (rng.random() - 0.5) * 8, rng, slot=i)
        for n, a in enumerate(agents, 1):
            if a.id in self.placed:
                self.placed[a.id].number = n
        self._relax()
        self.edges = [e for e in g.edges if e.src in self.placed and e.dst in self.placed]
        self.artery = {id(e) for e in (self.chain.edges if self.chain else [])}
        # every edge is a vein: an organic walk from source to target, with side branches
        self.curves: Dict[int, List[Point]] = {}
        self.branches: Dict[int, List[Tuple[float, List[Point]]]] = {}
        for e in self.edges:
            a, b = self.placed[e.src], self.placed[e.dst]
            hot = id(e) in self.artery
            self.curves[id(e)] = _walk(rng, (a.x, a.y), (b.x, b.y), wobble=0.55 if hot else 0.95)
            path = self.curves[id(e)]
            brs = []
            for _ in range(rng.randint(2, 4) if hot else rng.randint(0, 2)):
                k = rng.uniform(0.2, 0.8)
                j = int(len(path) * k)
                px, py = path[j]
                nx, ny = path[min(j + 1, len(path) - 1)]
                heading = math.atan2(ny - py, nx - px) + rng.choice((-1, 1)) * rng.uniform(0.8, 1.5)
                brs.append((k, _branch(rng, (px, py), heading, rng.randint(4, 10))))
            self.branches[id(e)] = brs
        self.dust = [(rng.random() * self.W, rng.random() * self.H) for _ in range(int(self.W * self.H * 0.006))]
        self._schedule()

    # -- layout ---------------------------------------------------------------------------
    def _side(self, nodes, x: float, rng: random.Random, slot: int = 0) -> None:
        """Place nodes above and below the spine, alternating, away from the middle row."""
        mid = self.H / 2
        for i, node in enumerate(nodes):
            k = slot + i
            band = 1 + k // 2
            up = k % 2 == 0
            gap = max(6.0, self.H * 0.13)  # spread over the canvas, whatever its height
            y = mid + (-1 if up else 1) * (gap + (band - 1) * gap * 1.1 + 2) + (rng.random() - 0.5) * 3
            self.placed[node.id] = Placed(node.id, node.kind, node.label, x, min(self.H - 2, max(2, y)))

    def _relax(self, rounds: int = 30) -> None:
        """Push nodes that sit too close apart (vertical only, the spine stays put)."""
        spine = set(self.chain.nodes) if self.chain else set()
        nodes = list(self.placed.values())
        for _ in range(rounds):
            for a in nodes:
                if a.id in spine:
                    continue
                for b in nodes:
                    if a is b:
                        continue
                    dx, dy = a.x - b.x, a.y - b.y
                    if abs(dx) < 8 and abs(dy) < 6:
                        push = (6 - abs(dy)) * 0.3 * (1 if dy >= 0 else -1)
                        a.y = min(self.H - 2, max(2, a.y + push))

    def _curve(self, a: Placed, b: Placed, rng: random.Random, bend: Optional[float] = None) -> List[Point]:
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy) or 1.0
        bend = (rng.random() - 0.5) * 0.5 * dist if bend is None else bend * dist
        cx, cy = mx - dy / dist * bend, my + dx / dist * bend
        n = max(6, int(dist * 1.3))
        pts = []
        for i in range(n + 1):
            t = i / n
            x = (1 - t) ** 2 * a.x + 2 * (1 - t) * t * cx + t ** 2 * b.x
            y = (1 - t) ** 2 * a.y + 2 * (1 - t) * t * cy + t ** 2 * b.y
            pts.append((x, y))
        return pts

    # -- the spread ----------------------------------------------------------------------
    SPEED = 70.0  # dots per second along a vein

    def _schedule(self) -> None:
        """When the spread reaches each node and starts each vein: breadth first from one point, the
        entry of the strongest chain, along the real edges only."""
        self.origin = self.chain.nodes[0] if self.chain else next(
            (p.id for p in self.placed.values() if p.kind == "input"), next(iter(self.placed), None))
        self.reached: Dict[str, float] = {}
        self.vein_start: Dict[int, float] = {}
        if self.origin is None:
            return
        self.reached[self.origin] = 0.0
        frontier = [self.origin]
        while frontier:
            frontier.sort(key=lambda n: self.reached[n])
            here = frontier.pop(0)
            for e in self.edges:
                if e.src != here or id(e) in self.vein_start:
                    continue
                start = self.reached[here] + (0.0 if id(e) in self.artery else 0.12)
                self.vein_start[id(e)] = start
                arrive = start + len(self.curves[id(e)]) / self.SPEED
                if e.dst not in self.reached or arrive < self.reached[e.dst]:
                    self.reached[e.dst] = arrive
                    frontier.append(e.dst)
        self.spread_total = max(self.reached.values(), default=0.0)
        if self.spread_total > 2.8:  # a big host spreads faster, not longer
            k = 2.8 / self.spread_total
            self.reached = {n: v * k for n, v in self.reached.items()}
            self.vein_start = {e: v * k for e, v in self.vein_start.items()}
            self.speed = self.SPEED / k
            self.spread_total = 2.8
        else:
            self.speed = self.SPEED

    def chain_duration(self) -> float:
        """How long the spread runs before the map is complete."""
        return self.spread_total + 0.4

    def taken(self, nid: str, spread_t: Optional[float]) -> bool:
        return spread_t is not None and nid in self.reached and spread_t >= self.reached[nid]

    def render(self, t: float, spread_t: Optional[float] = None, complete: bool = False,
               drop: Optional[float] = None, view=None, selected: Optional[str] = None,
               pulses: Sequence[Tuple[str, Optional[str], str, float]] = (), labels: bool = False) -> Text:
        """spread_t: seconds since the drop landed on the origin (None: not yet);
        drop: 0..1 while the drop travels from the centre to the origin;
        view: a camera (Zoom, Sphere); selected: a node to ring."""
        c = ViewCanvas(self.w, self.h, view) if view is not None else Canvas(self.w, self.h)
        globe = view if isinstance(view, Sphere) else (view.first if isinstance(view, Then) and isinstance(view.first, Sphere) else None)
        if globe is not None:  # a globe behind the map: meridians, parallels and the rim
            for k in range(0, 360, 30):
                for j in range(0, 90):
                    X, Y, d = view(k / 360 * self.W, j / 90 * self.H)
                    if d >= 0:
                        Canvas.dot(c, X, Y, "#134e4a", -1)
            for j in range(1, 6):
                for k in range(0, 160):
                    X, Y, d = view(k / 160 * self.W, j / 6 * self.H)
                    if d >= 0:
                        Canvas.dot(c, X, Y, "#134e4a", -1)
            for k in range(0, 240):
                a = k / 240 * 2 * math.pi
                rx, ry = self.W / 2 + math.cos(a) * globe.R, self.H / 2 + math.sin(a) * globe.R
                if isinstance(view, Then):
                    rx, ry, _ = view.second(rx, ry)
                Canvas.dot(c, rx, ry, "#0f766e", -1)
        st = 99.0 if complete else spread_t
        for x, y in self.dust:
            c.dot(x, y, "#1e293b", -1)
        if st is None:
            for p in self.placed.values():
                c.put(p.x, p.y, "·", "#334155")
            if drop is not None and self.origin in self.placed:
                o = self.placed[self.origin]
                k = 1 - (1 - max(0.0, min(1.0, drop))) ** 2
                cx, cy = self.W / 2, self.H / 2
                x, y = cx + (o.x - cx) * k, cy + (o.y - cy) * k
                for d in range(6):  # a short bright trail behind the drop
                    kk = max(0.0, k - d * 0.04)
                    c.dot(cx + (o.x - cx) * kk, cy + (o.y - cy) * kk, "bold #fdf4ff" if d == 0 else CHAIN, 7 - d)
                c.put(x, y, "●", "bold #fdf4ff")
            return c.render()
        settled = st >= self.spread_total + 0.3
        for e in self.edges:
            start = self.vein_start.get(id(e))
            if start is None or st < start:
                continue
            path = self.curves[id(e)]
            n = min(len(path), int((st - start) * self.speed) + 1)
            art = id(e) in self.artery
            growing = n < len(path)
            beat = settled and (0.5 + 0.5 * math.sin((st - self.spread_total) * 5.0)) > 0.75
            for i, (x, y) in enumerate(path[:n]):
                tip = n - i
                if growing and tip <= 1:
                    style, prio = "bold #fdf4ff", 6
                elif growing and tip <= 6:
                    style, prio = (CHAIN if art else "#99f6e4"), 4
                elif beat:
                    style, prio = (CHAIN if art else "#14b8a6"), (3 if art else 1)
                else:
                    style, prio = ("#c026d3" if art else "#0f766e"), (3 if art else 1)
                c.dot(x, y, style, prio)
                if 0 < i < len(path) - 1 and (tip > 6 or not growing):  # grown veins thicken
                    (px, py), (qx, qy) = path[i - 1], path[i + 1]
                    nx, ny = -(qy - py), (qx - px)
                    norm = math.hypot(nx, ny) or 1.0
                    c.dot(x + nx / norm, y + ny / norm, "#a21caf" if art else "#115e59", 2 if art else 0)
                if growing and tip <= 3:  # the front glows
                    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c.dot(x + ox, y + oy, "#f5d0fe" if art else "#ccfbf1", 5)
            for k, br in self.branches[id(e)]:
                if n < len(path) * k:
                    continue
                grown = min(len(br), int((st - start - k * len(path) / self.speed) * self.speed * 0.6) + 1)
                for x, y in br[:max(0, grown)]:
                    c.dot(x, y, "#86198f" if art else "#115e59", 0)
        # once everything is taken, light keeps flowing out along the artery, and faintly elsewhere
        if settled:
            flow = st - self.spread_total - 0.3
            route = [p for e in (self.chain.edges if self.chain else []) for p in self.curves.get(id(e), [])]
            if route:
                for off in (0.0, 0.5):
                    head = int((flow * 34 + off * (len(route) + 20)) % (len(route) + 20))
                    for d in range(4):
                        j = head - d
                        if 0 <= j < len(route):
                            c.dot(*route[j], "bold #fdf4ff" if d == 0 else CHAIN, 7)
            for e in self.edges:
                if id(e) in self.artery:
                    continue
                path = self.curves[id(e)]
                j = int((flow * 18 + (sum(map(ord, e.src + e.dst)) % 23)) % (len(path) + 30))
                if j < len(path):
                    c.dot(*path[j], "#99f6e4", 5)
        # nodes: dark until the spread takes them, then a ring of light and the lit glyph
        for p in self.placed.values():
            at = self.reached.get(p.id)
            if at is None or st < at:
                c.put(p.x, p.y, "·", "#334155")
                continue
            age = st - at
            on_art = self.chain is not None and p.id in self.chain.nodes
            if p.kind == "impact" and age < 0.9 and not complete:  # taking an impact sends a shockwave
                color = IMPACT_STYLE.get(p.id, "#f87171")
                c.ring(p.x, p.y, 1.5 + age * 22, color, 5, step=age)
                if age > 0.15:
                    c.ring(p.x, p.y, 1.5 + (age - 0.15) * 22, color, 4, step=-age)
            elif age < 0.6 and not complete:
                c.ring(p.x, p.y, 1.5 + age * 12, CHAIN if on_art else "#2dd4bf", 4, step=age)
            else:  # a faint halo that breathes once taken
                halo = 2.2 + 0.5 * math.sin(t * 3.0 + p.x)
                c.ring(p.x, p.y, halo, "#86198f" if on_art else "#134e4a", 0)
            if p.kind == "input":
                c.put(p.x, p.y, "◆", f"bold {INPUT}")
            elif p.kind == "impact":
                color = IMPACT_STYLE.get(p.id, "#f87171")
                bright = age < 0.4 or (on_art and int(t * 2.5) % 2 == 0)
                c.put(p.x, p.y, "▲", f"bold {color}" if bright else color)
            else:
                pulse = on_art and int(t * 3) % 2 == 0
                c.put(p.x, p.y, "●", f"bold {CHAIN}" if on_art and (pulse or age < 0.4) else (f"bold {AGENT}" if age < 0.4 else AGENT))
                if p.number is not None:
                    c.put(p.x + 2.2, p.y, str(p.number) if p.number < 10 else "+", "bold #e2e8f0" if on_art else "#94a3b8")
        self._pulses(c, pulses)
        for p in self.placed.values():  # names on the canvas: every agent with labels on, always the selected node
            if (labels and p.kind == "agent" and self.taken(p.id, st)) or p.id == selected:
                self._label(c, view, p, "bold #fde047" if p.id == selected else "#94a3b8")
        if selected in self.placed:
            p = self.placed[selected]
            c.ring(p.x, p.y, 3.2 + 0.4 * math.sin(t * 6), "bold #fde047", 8)
        if self.origin and self.origin in self.placed and st < 0.5 and not complete:  # the drop lands
            o = self.placed[self.origin]
            c.ring(o.x, o.y, 1 + st * 10, "bold #fdf4ff", 7)
        return c.render()

    def _label(self, c, view, p: Placed, style: str) -> None:
        """A node's name beside it, placed after the camera so it never bends with the sphere."""
        X, Y, d = view(p.x, p.y) if view is not None else (p.x, p.y, 1.0)
        if d < 0.1:
            return
        text = p.label if len(p.label) <= 18 else p.label[:17] + "…"
        right = X + 4 + 2 * len(text) < self.W
        x0 = X + (4.2 if p.number is not None else 2.6) if right else X - 2.6 - 2 * len(text)
        for i, ch in enumerate(text):
            Canvas.put(c, x0 + 2 * i, Y, ch, style)

    def _pulses(self, c, pulses: Sequence[Tuple[str, Optional[str], str, float]]) -> None:
        """Live decisions: ALLOW travels as teal light from the agent toward what the call does;
        WOULD BLOCK flashes purple at the agent, BLOCK red, and the light stops short."""
        paths = {(e.src, e.dst): self.curves[id(e)] for e in self.edges}
        for agent, target, decision, age in pulses:
            p = self.placed.get(agent)
            if p is None or age < 0:
                continue
            path = paths.get((agent, target)) if target else None
            if decision == "ALLOW":
                if path:
                    head = int(len(path) * min(1.0, age / 0.9))
                    for d in range(4):
                        j = head - d
                        if 0 <= j < len(path):
                            c.dot(*path[j], "bold #ccfbf1" if d == 0 else "#5eead4", 8 - d)
                elif age < 0.6:
                    c.ring(p.x, p.y, 1.5 + age * 6, "#5eead4", 6)
            else:
                color = CHAIN if decision == "WOULD_BLOCK" else "#f87171"
                if age < 0.8:
                    c.ring(p.x, p.y, 1.5 + age * 9, color, 8, step=age)
                    c.put(p.x, p.y, "●", f"bold {color}")
                if path and age < 0.5:  # the call starts, and is stopped short
                    head = int(len(path) * 0.3 * min(1.0, age / 0.3))
                    for j in range(max(0, head - 3), head + 1):
                        if j < len(path):
                            c.dot(*path[j], color, 8)

    def legend(self, spread_t: Optional[float], complete: bool = False) -> List[Text]:
        """Right-hand legend: each node appears as the spread takes it; the artery stands out."""
        st = 99.0 if complete else spread_t
        art = set(self.chain.nodes) if self.chain else set()
        rows: List[Text] = [Text("REACH MAP", style="label")]

        def shown(p: Placed) -> bool:
            return st is not None and p.id in self.reached and st >= self.reached[p.id]

        for p in self.placed.values():
            if p.kind == "input":
                rows.append(Text.assemble(("◆ " if shown(p) else "◇ ", INPUT),
                                          (p.label, ("bold #fde68a" if p.id in art else "text") if shown(p) else "#334155")))
        for p in sorted((q for q in self.placed.values() if q.kind == "agent"), key=lambda q: q.number or 0):
            rows.append(Text.assemble((f"{p.number} ", "#94a3b8"), ("● ", AGENT if shown(p) else "#334155"),
                                      (p.label, (f"bold {CHAIN}" if p.id in art else "head") if shown(p) else "#334155")))
        if self.hidden_agents:
            rows.append(Text(f"  + {self.hidden_agents} more agents", style="muted"))
        for p in self.placed.values():
            if p.kind == "impact":
                color = IMPACT_STYLE.get(p.id, "#f87171")
                rows.append(Text.assemble(("▲ ", color if shown(p) else "#334155"),
                                          (p.label, ("bold #fecaca" if p.id in art else "text") if shown(p) else "#334155")))
        return rows
