"""
The discovery web: veins grow from a core toward the six discovery layers while they run, each
layer's node lights up when the scanner reports it, light flows back along the finished veins,
and the agents that were found attach to the outer ring in their risk colour.

Pure drawing: `Web.render(...)` takes what is known at this instant (layer progress, which
layers are done and since when, the agents) and returns a Text. The shape comes from a fixed
seed, so the same host always grows the same web.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rich.text import Text

BRAILLE = 0x2800
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

VEIN_OLD = "#115e59"
VEIN = "#14b8a6"
VEIN_HOT = "#5eead4"
TIP = "bold #f0fdfa"
PULSE = "bold #ccfbf1"
DUST = "#1e293b"
RING = "#2dd4bf"

Point = Tuple[float, float]


class Canvas:
    """Braille canvas (2 x 4 dots per cell) where a brighter style wins a shared cell, plus glyphs."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.bits = [[0] * w for _ in range(h)]
        self.style: List[List[Tuple[int, str]]] = [[(-1, "")] * w for _ in range(h)]
        self.glyph: Dict[Tuple[int, int], Tuple[str, str]] = {}

    def dot(self, x: float, y: float, style: str, prio: int = 0) -> None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < self.w * 2 and 0 <= yi < self.h * 4:
            cx, cy = xi // 2, yi // 4
            self.bits[cy][cx] |= _DOTS[yi % 4][xi % 2]
            if prio >= self.style[cy][cx][0]:
                self.style[cy][cx] = (prio, style)

    def put(self, x: float, y: float, ch: str, style: str) -> None:
        cx, cy = int(round(x)) // 2, int(round(y)) // 4
        if 0 <= cx < self.w and 0 <= cy < self.h:
            self.glyph[(cx, cy)] = (ch, style)

    def ring(self, cx: float, cy: float, r: float, style: str, prio: int = 0, step: float = 0.0) -> None:
        n = max(8, int(r * 7))
        for i in range(n):
            a = i / n * math.tau + step
            self.dot(cx + math.cos(a) * r, cy + math.sin(a) * r, style, prio)

    def render(self) -> Text:
        """One Text, with neighbouring cells of the same style merged into one run: far fewer
        style changes for the terminal to draw, which keeps fast animations smooth."""
        out = Text()
        run, run_style = [], None
        for y in range(self.h):
            for x in range(self.w):
                g = self.glyph.get((x, y))
                if g:
                    ch, st = g
                elif self.bits[y][x]:
                    ch, st = chr(BRAILLE + self.bits[y][x]), self.style[y][x][1]
                else:
                    ch, st = " ", ""
                if st != run_style and run:
                    out.append("".join(run), style=run_style or "")
                    run = []
                run_style = st
                run.append(ch)
            run.append("\n" if y < self.h - 1 else "")
        if run:
            out.append("".join(run), style=run_style or "")
        return out


def _walk(rng: random.Random, start: Point, target: Point, wobble: float = 0.9, max_steps: int = 400) -> List[Point]:
    """An organic path: steps toward the target with smooth angular noise."""
    phases = [rng.random() * math.tau for _ in range(3)]
    freqs = [0.11 + rng.random() * 0.08, 0.23 + rng.random() * 0.1, 0.05 + rng.random() * 0.04]
    x, y = start
    pts = [(x, y)]
    for i in range(max_steps):
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        if dist < 1.2:
            break
        noise = sum(math.sin(i * f + p) for f, p in zip(freqs, phases)) / 3
        damp = min(1.0, dist / 10)  # straighten near the target
        a = math.atan2(dy, dx) + noise * wobble * damp
        x, y = x + math.cos(a), y + math.sin(a)
        pts.append((x, y))
    pts.append(target)
    return pts


def _branch(rng: random.Random, origin: Point, heading: float, length: int) -> List[Point]:
    x, y = origin
    a = heading
    pts = []
    turn = (rng.random() - 0.5) * 0.25
    for _ in range(length):
        a += turn + (rng.random() - 0.5) * 0.5
        x, y = x + math.cos(a), y + math.sin(a)
        pts.append((x, y))
    return pts


@dataclass
class AgentMark:
    glyph: str
    style: str
    age: float  # seconds since it appeared (negative: not yet)
    alive: bool  # running: pulses


class Web:
    def __init__(self, layers: Sequence[str], w: int = 40, h: int = 13, seed: str = "venom") -> None:
        self.layers = list(layers)
        self.w, self.h = w, h
        rng = random.Random(seed)
        W, H = w * 2, h * 4
        self.core: Point = (W / 2, H / 2)
        rx, ry = W * 0.36, H * 0.36
        self.anchors: Dict[str, Point] = {}
        self.veins: Dict[str, List[Point]] = {}
        self.branches: Dict[str, List[Tuple[float, List[Point]]]] = {}
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            a = -math.pi / 2 + i / n * math.tau + (rng.random() - 0.5) * 0.25
            anchor = (self.core[0] + math.cos(a) * rx, self.core[1] + math.sin(a) * ry)
            self.anchors[layer] = anchor
            path = _walk(rng, self.core, anchor)
            self.veins[layer] = path
            brs = []
            for _ in range(rng.randint(3, 6)):
                k = rng.uniform(0.25, 0.9)
                j = int(len(path) * k)
                px, py = path[j]
                nx, ny = path[min(j + 1, len(path) - 1)]
                heading = math.atan2(ny - py, nx - px) + rng.choice((-1, 1)) * rng.uniform(0.7, 1.4)
                brs.append((k, _branch(rng, (px, py), heading, rng.randint(6, 16))))
            self.branches[layer] = brs
        # agent slots on an outer ring, each fed from the nearest layer node
        self.slots: List[Tuple[Point, List[Point]]] = []
        orx, ory = W * 0.47, H * 0.46
        m = 12
        for i in range(m):
            a = -math.pi / 2 + (i + 0.5) / m * math.tau
            slot = (self.core[0] + math.cos(a) * orx, self.core[1] + math.sin(a) * ory)
            near = min(self.anchors.values(), key=lambda p: math.hypot(p[0] - slot[0], p[1] - slot[1]))
            self.slots.append((slot, _walk(rng, near, slot, wobble=0.6)))
        self.dust = [(rng.random() * W, rng.random() * H) for _ in range(int(W * H * 0.008))]
        self.reach = math.hypot(W, H) / 2

    def render(self, t: float, growth: Dict[str, float], lit: Dict[str, float], agents: Sequence[AgentMark],
               complete: bool = False, flash: Optional[float] = None) -> Text:
        """growth: layer -> 0..1 vein length; lit: layer -> seconds since its node lit (absent: not yet);
        flash: 0..1 progress of the closing wave once everything has landed."""
        c = Canvas(self.w, self.h)
        for x, y in self.dust:
            c.dot(x, y, DUST, -1)
        cx0, cy0 = self.core
        if t < 0.75 and not complete:  # the core ignites: a wave runs across the canvas first
            k = t / 0.75
            r = 2 + (1 - (1 - k) ** 3) * self.reach
            c.ring(cx0, cy0, r, VEIN_HOT if k < 0.5 else VEIN_OLD, 1, step=k)
            if k < 0.6:
                c.ring(cx0, cy0, r * 0.82, VEIN_OLD, 0, step=-k)
        glow = flash is not None and flash < 1.0
        # veins and their branches
        for layer in self.layers:
            path = self.veins[layer]
            g = 1.0 if complete else max(0.0, min(1.0, growth.get(layer, 0.0)))
            n = int(len(path) * g)
            done = layer in lit or complete
            for i, (x, y) in enumerate(path[:n]):
                hot = n - i <= 5 and not done
                c.dot(x, y, VEIN_HOT if hot or glow else (VEIN if done else VEIN_OLD), 2 if hot else 1)
                if done and 0 < i < len(path) - 2:  # a finished vein thickens
                    (px, py), (qx, qy) = path[i - 1], path[i + 1]
                    nx, ny = -(qy - py), (qx - px)
                    norm = math.hypot(nx, ny) or 1.0
                    c.dot(x + nx / norm, y + ny / norm, VEIN_HOT if glow else "#0f766e", 1)
            if n and not done:
                c.dot(*path[n - 1], TIP, 4)
            for k, br in self.branches[layer]:
                if g <= k:
                    continue
                bl = int(len(br) * min(1.0, (g - k) / 0.35))
                for x, y in br[:bl]:
                    c.dot(x, y, VEIN_OLD if not done else "#0f766e", 0)
            # light flowing back to the core along a finished vein
            if done and not complete or (complete and layer in lit):
                L = len(path)
                for off in (0.0, 0.5):
                    pos = L - ((t * 26 + off * (L + 18) + sum(map(ord, layer)) % 17) % (L + 18))
                    for d in (0, 1):
                        j = int(pos) - d
                        if 0 <= j < L:
                            c.dot(*path[j], PULSE, 3)
        # layer nodes: dim seed, then a bright node with a ring of light spreading out
        for layer, (ax, ay) in self.anchors.items():
            age = lit.get(layer)
            if age is None and not complete:
                c.dot(ax, ay, VEIN_OLD, 1)
                continue
            age = 9.0 if age is None else age
            if age < 0.7:
                r = 1.5 + age * 14
                c.ring(ax, ay, r, RING if age < 0.35 else VEIN, 2, step=age)
            c.put(ax, ay, "◆", "bold #5eead4" if age > 0.25 else "bold #f0fdfa")
        # agents attach to the outer ring
        for i, mark in enumerate(agents[: len(self.slots)]):
            if mark.age < 0:
                continue
            (sx, sy), feed = self.slots[i]
            k = min(1.0, mark.age / 0.25)
            for x, y in feed[: max(1, int(len(feed) * k))]:
                c.dot(x, y, VEIN, 1)
            if k >= 1.0:
                if mark.age < 0.5:
                    c.ring(sx, sy, 1.5 + mark.age * 6, mark.style, 2)
                bright = mark.alive and int(t * 3 + i) % 2 == 0
                c.put(sx, sy, mark.glyph, f"bold {mark.style}" if bright else mark.style)
        if flash is not None and flash < 1.0:  # everything has landed: one wave out from the core
            r = 2 + flash * self.reach
            c.ring(cx0, cy0, r, TIP if flash < 0.3 else VEIN_HOT, 4, step=flash)
            c.ring(cx0, cy0, max(1.0, r - 3), VEIN, 3, step=-flash)
        # the core breathes
        cx, cy = self.core
        c.ring(cx, cy, 2.2 + math.sin(t * 4.2) * 0.7, VEIN_HOT, 3, step=t)
        c.put(cx, cy, "◉", "bold #f0fdfa" if int(t * 2) % 2 == 0 else "bold #5eead4")
        return c.render()
