"""
Verification animations for the studio: pure functions from (result, time) to a Rich
renderable, so they can be played by Textual, tested, and rendered to images.

Everything shown is the real result: the rules and pairs Z3 checked, the conflict it found and
its witness; the TLA+ engine used, the domain sizes, the number of states explored, each
invariant's verdict and the counterexample trace. Only the pacing is chosen for the eye.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .engines import TLARun, Z3Run, card_of
from ..render.words import n as _n

BRAILLE = 0x2800
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def seg(t: float, a: float, b: float) -> float:
    """Progress 0..1 of time t inside [a, b]."""
    return 0.0 if t <= a else (1.0 if t >= b else (t - a) / (b - a))


class Canvas:
    """A braille canvas: each character cell holds 2 x 4 dots."""

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.cells = [[0] * width for _ in range(height)]
        self.styles: List[List[Optional[str]]] = [[None] * width for _ in range(height)]

    def dot(self, x: float, y: float, style: str = "brand") -> None:
        xi, yi = int(x), int(y)
        if 0 <= xi < self.w * 2 and 0 <= yi < self.h * 4:
            cx, cy = xi // 2, yi // 4
            self.cells[cy][cx] |= _DOTS[yi % 4][xi % 2]
            self.styles[cy][cx] = style

    def render(self) -> Text:
        out = Text()
        for row, styles in zip(self.cells, self.styles):
            for bits, st in zip(row, styles):
                out.append(chr(BRAILLE + bits) if bits else " ", style=st or "")
            out.append("\n")
        out.rstrip()
        return out


def shimmer(text: str, t: float, base: str, glow: str = "bold #f0fdfa") -> Text:
    """Text with a light band passing over it once."""
    out = Text()
    head = t * (len(text) + 8) - 4
    for i, ch in enumerate(text):
        out.append(ch, style=glow if abs(i - head) < 2.5 else base)
    return out


# ---------------------------------------------------------------------------
# Z3
# ---------------------------------------------------------------------------

def z3_duration(run: Z3Run) -> float:
    n = len(run.pairs)
    return 1.1 + min(1.9, 0.06 * n) + (0.8 if not run.ok else 0.6)


def _short(name: str, n: int = 14) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def z3_frame(run: Z3Run, t: float, width: int = 60) -> Group:
    parts: List = []
    if run.stage in ("parse", "validate"):
        issue = run.issues[0] if run.issues else None
        k = ease(seg(t, 0, 0.5))
        head = Text.assemble(("✗ ", "high"), (("PARSE ERROR" if run.stage == "parse" else "INVALID")[: max(1, int(11 * k))], "high"))
        parts += [head, Text("")]
        if issue:
            where = f"line {issue.line}: " if issue.line else ""
            parts.append(Text(where + issue.message, style="text"))
        return Group(*parts)

    d_enc, d_reach = 0.3, 0.35
    d_pairs = min(1.9, 0.06 * len(run.pairs))
    t_reach, t_pairs = d_enc, d_enc + d_reach
    t_verdict = t_pairs + d_pairs + 0.1

    # 1. encoding: variables become solver symbols
    k = seg(t, 0, d_enc)
    chips = Text()
    items = list(run.variables.items())
    shown = min(int(math.ceil(len(items) * ease(k))), 6)
    for name, dom in items[:shown]:
        d = dom.strip()
        n = d.count('"') // 2
        sym = f": {n} value{'s' if n != 1 else ''}" if d.startswith("{") else ("∈ [" + d.replace("..", "‥") + "]" if ".." in d else ": " + d)
        chips.append(f" {name} ", style="code")
        chips.append(f"{sym}  ", style="muted")
    parts.append(Text.assemble(("ENCODE  ", "label"), (f"{_n(len(items), 'variable')} → Z3 symbols", "muted")))
    if len(items) > 6 and k >= 1:
        chips.append(f"+{len(items) - 6} more", style="muted")
    parts.append(chips if chips.plain else Text(" "))
    parts.append(Text(""))

    # 2. reachability: one probe per rule
    rules = run.rules
    kr = seg(t, t_reach, t_reach + d_reach)
    reach = Text("REACH   ", style="label")
    done = int(len(rules) * ease(kr))
    for i, r in enumerate(rules):
        if i < done:
            bad = r in run.unreachable
            reach.append("✗" if bad else "●", style="high" if bad else "ok")
        elif i == done and kr > 0:
            reach.append(SPIN[int(t * 14) % len(SPIN)], style="brand")
        else:
            reach.append("·", style="muted")
        reach.append(" ")
    if kr < 1:
        reach.append(f"  {done}/{_n(len(rules), 'rule')} probed", style="muted")
    else:
        reach.append(f"  {len(rules) - len(set(run.unreachable))}/{_n(len(rules), 'rule')} can trigger",
                     style="muted" if not run.unreachable else "warn")
    parts += [reach, Text("")]

    # 3. pairwise matrix, filled along the diagonals
    n = len(rules)
    if n >= 2:
        kp = seg(t, t_pairs, t_pairs + max(d_pairs, 0.2))
        order = sorted(run.pairs, key=lambda p: (rules.index(p[1]) - rules.index(p[0]), rules.index(p[0])))
        filled = int(len(order) * kp)
        state: Dict[Tuple[str, str], str] = {}
        conflicts = {tuple(sorted(c)) for c in run.conflicts}
        for i, p in enumerate(order):
            key = tuple(sorted(p))
            if i < filled:
                state[key] = "x" if key in conflicts else "ok"
            elif i == filled and kp < 1:
                state[key] = "probe"
        grid = Table.grid(padding=(0, 0))
        grid.add_column(no_wrap=True)
        for _ in range(n):
            grid.add_column(no_wrap=True, width=2)
        limit = min(n, 14)
        header = [Text("PAIRS   ", style="label")] + [Text(f"{(j + 1) % 10} ", style="brand" if (j + 1) % 10 == 0 else "muted") for j in range(limit)]  # 10 shows as a bright 0
        grid.add_row(*header, *([""] * (n - limit)))
        for a in range(limit):
            row: List = [Text(f"{a + 1:>6}  ", style="muted")]
            for b in range(limit):
                if b <= a:
                    row.append(Text("  "))
                    continue
                s = state.get(tuple(sorted((rules[a], rules[b]))))
                cell = {"ok": ("■ ", "ok"), "x": ("✖ ", "bold #f87171"), "probe": ("▣ ", "brand")}.get(s or "", ("· ", "bar.empty"))
                row.append(Text(*cell))
            grid.add_row(*row, *([""] * (n - limit)))
        legend = Text("        ")
        for j, r in enumerate(rules[:limit]):
            hot = any(r in c for c in run.conflicts) and kp >= 1
            legend.append(f"{j + 1} ", style="label")
            legend.append(f"{_short(r, 22)}   ", style="high" if hot else "muted")
        parts.append(grid)
        parts.append(legend)
        parts.append(Text(f"        {min(filled, len(order))}/{_n(len(order), 'pair')} checked for conflicting demands"
                          + (f"   (first {_n(limit, 'rule')} shown)" if n > limit else ""), style="muted"))
        parts.append(Text(""))

    # 4. verdict
    if t >= t_verdict:
        kv = seg(t, t_verdict, t_verdict + 0.6)
        if run.ok:
            line = f"✓ PROVEN CONSISTENT   {_n(n, 'rule')} · {_n(len(run.pairs), 'pair')} · {run.elapsed_ms} ms"
            parts.append(shimmer(line, kv, "bold #4ade80"))
            if run.unreachable:
                parts.append(Text(f"  but {', '.join(sorted(set(run.unreachable)))} can never trigger: "
                                  "it protects nothing (see the suggestions)", style="warn"))
            else:
                parts.append(Text("  every rule can trigger, and no two rules ever demand the impossible", style="muted"))
        else:
            for issue in run.issues[:2]:
                title = {"CONTRADICTION": "✗ CONTRADICTION", "UNREACHABLE": "✗ UNREACHABLE"}.get(issue.kind, f"✗ {issue.kind}")
                who = " × ".join(issue.rules) if issue.rules else ""
                parts.append(shimmer(f"{title}   {who}", kv, "bold #f87171"))
                if issue.model and kv > 0.3:
                    w = Text("  witness  ", style="label")
                    for k2, v in list(sorted(issue.model.items()))[:6]:
                        w.append(f"{k2}=", style="muted")
                        w.append(f"{v}  ", style="head")
                    parts.append(w)
                parts.append(Text("  " + issue.message, style="text"))
    return Group(*parts)


# ---------------------------------------------------------------------------
# TLA+
# ---------------------------------------------------------------------------

def tla_duration(run: TLARun) -> float:
    base = 2.6 + 0.18 * len(run.constraints)
    return base + (0.5 + 1.1 * len(_examples(run)) if _examples(run) else 0.6)


def _seed(run: TLARun) -> int:
    key = "|".join(f"{v.get('name')}:{v.get('card')}" for v in run.variables) + f"|{run.total_states}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def radar(t: float, w: int = 28, h: int = 7) -> Text:
    """While the model checker runs: a rotating sweep with fading echoes."""
    c = Canvas(w, h)
    cx, cy, R = w, h * 2, min(w, h * 2) - 1
    angle = t * 2.6
    for k in range(18):
        a = angle - k * 0.06
        r_style = "brand" if k < 4 else ("brand.dim" if k < 10 else "bar.empty")
        for s in range(0, int(R * 2), 2):
            rr = s / 2
            c.dot(cx + math.cos(a) * rr * 2 * 0.98, cy + math.sin(a) * rr, r_style)
    for ring in (0.5, 1.0):
        for d in range(0, 360, 6):
            a = math.radians(d)
            c.dot(cx + math.cos(a) * R * ring * 2 * 0.98, cy + math.sin(a) * R * ring, "bar.empty")
    return c.render()


def explorer(run: TLARun, k: float, w: int = 28, h: int = 7) -> Text:
    """The state space as a tree growing ring by ring from the initial state (k = 0..1)."""
    c = Canvas(w, h)
    cx, cy = w, h * 2
    rng = random.Random(_seed(run))
    rings = 6
    dots = min(420, max(60, int(math.log10(max(run.total_states, 10)) * 90)))
    share = run.blocked / run.checked if run.checked else 0.0
    for i in range(dots):
        ring = min(rings - 1, int(rng.random() ** 0.7 * rings))
        if (ring + 1) / rings > k + 0.02:
            continue
        a = rng.random() * math.tau
        r = (ring + 0.55 + 0.3 * rng.random()) / rings
        x, y = cx + math.cos(a) * r * (w - 1), cy + math.sin(a) * r * (h * 2 - 1)
        style = "brand" if ring >= int(k * rings) - 1 else "brand.dim"
        if k >= 1 and rng.random() < share:
            style = BLOCKED
        c.dot(x, y, style)
    c.dot(cx, cy, "head")
    return c.render()


def tla_pending(t: float, engine_note: str = "") -> Group:
    label = Text.assemble(("EXPLORE ", "label"), (SPIN[int(t * 14) % len(SPIN)] + " model checking", "brand"),
                          (f"   {t:4.1f}s", "head"), ("   " + engine_note if engine_note else "", "muted"))
    parts = [label, Text(""), radar(t)]
    if t > 3:
        parts += [Text(""), Text("a large state space takes longer; you can keep editing, the result arrives here",
                                 style="muted")]
    return Group(*parts)


BLOCKED = "#e879f9"


def _share_bar(share: float, w: int = 12) -> Text:
    full = share * w
    n = int(full)
    t = Text("█" * n, style=BLOCKED)
    if n < w:
        frac = full - n
        t.append("▏▎▍▌▋▊▉"[min(6, int(frac * 7))] if frac > 0.07 else "░", style=BLOCKED if frac > 0.07 else "bar.empty")
        t.append("░" * (w - n - 1), style="bar.empty")
    return t


def _examples(run: TLARun, limit: int = 3):
    """(rule, blocked values, trace length) for the enforced rules, the guard grid first, TLA+ traces second."""
    out = []
    fixed = {str(v.get("name")) for v in run.variables if card_of(v) == 1}  # one possible value: noise
    for c in run.constraints:
        if run.status_of(c) != "ENFORCED":
            continue
        g = run.guard.get(c.name)
        values = (g.example if g and g.example else None) or (
            {k: v for k, v in c.trace[-1].items() if not str(k).startswith("_")} if c.trace else None)
        if values:
            out.append((c.name, {k: v for k, v in values.items() if k not in fixed} or values, len(c.trace)))
        if len(out) >= limit:
            break
    return out


def _verdict(run: TLARun, k: float) -> Text:
    if run.refuses_to_load:
        return shimmer("▲ STRICT POLICY   ENABLE_FORMAL_VERIFICATION: TRUE makes the compiler refuse rules that block; "
                       "see the suggestion", k, "bold #fbbf24")
    enforced, never = len(run.enforced), len(run.never_fires)
    line = f"✓ GUARD VERIFIED   {enforced} {'rule enforces' if enforced == 1 else 'rules enforce'}"
    if never:
        line += f" · {never} never {'fires' if never == 1 else 'fire'}"
    if run.checked:
        line += f" · blocks {run.blocked:,} of {run.checked:,} checked states"
    line += f" · {run.total_states:,} states explored · {run.elapsed_ms} ms"
    return shimmer(line, k, "bold #4ade80")


def tla_frame(run: TLARun, t: float, width: int = 60) -> Group:
    parts: List = []
    badge = Text.assemble((" TLC ", "bold #0f172a on #4ade80") if run.engine == "TLC" else (" BFS ", "bold #0f172a on #fbbf24"),
                          ("  " + run.engine_note, "muted"))
    parts += [badge, Text("")]
    if run.error:
        parts.append(Text.assemble(("✗ NOT VERIFIED  ", "high"), (run.error, "text")))
        return Group(*parts)

    t_inv0 = 1.9
    compact = bool(_examples(run)) and t >= t_inv0 + 0.18 * len(run.constraints) + 0.4
    if compact:
        parts.append(Text.assemble(("SPEC    ", "label"), (f"{_n(len(run.variables), 'variable')} · state space {run.state_space} · ", "muted"),
                                   (f"{run.total_states:,} states explored", "bold #5eead4"), (f" · {run.elapsed_ms} ms", "muted")))
        parts.append(Text(""))
    # 1. spec: the variables and the size of the state space
    ks = seg(t, 0, 0.5)
    spec = Table.grid(padding=(0, 1))
    spec.add_column(no_wrap=True, width=10)
    spec.add_column(no_wrap=True)
    spec.add_column(no_wrap=True, style="muted")
    for vi in run.variables[: max(1, int(len(run.variables) * ease(ks) + 0.999)) if ks > 0 else 0][:8]:
        size = card_of(vi) or 2
        bar = "▮" * max(1, min(12, int(math.log2(max(size, 2))) * 2))
        spec.add_row(Text(_short(str(vi.get("name")), 10), style="text"), Text(bar, style="brand.dim"),
                     str(vi.get("card")).replace("|", ""))
    if not compact:
        parts += [Text.assemble(("SPEC    ", "label"), (f"{_n(len(run.variables), 'variable')} · state space {run.state_space}", "muted")), spec, Text("")]

    # 2. exploration: the tree grows, the counter climbs to the real number of states
    ke = seg(t, 0.5, 1.9)
    states = int(run.total_states * ease(ke))
    explore = Table.grid(padding=(0, 2))
    explore.add_column(no_wrap=True)
    explore.add_column(no_wrap=True)
    side = Group(Text.assemble(("EXPLORE", "label")), Text(""),
                 Text.assemble((f"{states:,}", "bold #5eead4"), (" states", "muted")),
                 Text(f"{run.elapsed_ms} ms" if ke >= 1 else "", style="muted"), Text(""),
                 Text.assemble(("● ", "brand"), ("allowed  ", "muted"), ("● ", BLOCKED), ("blocked by the guard", "muted"))
                 if ke >= 1 and run.checked else Text(""))
    explore.add_row(explorer(run, ease(ke)), side)
    if not compact:
        parts += [explore, Text("")]

    # 3. the rules as the guard runs them: what each one blocks
    t_inv = 1.9
    inv: List = []
    for i, c in enumerate(run.constraints):
        at = t_inv + i * 0.18
        if t < at:
            inv.append(Text.assemble(("· ", "muted"), (c.name, "muted")))
            continue
        if t < at + 0.15:
            inv.append(Text.assemble((SPIN[int(t * 14) % len(SPIN)] + " ", "brand"), (c.name, "head")))
            continue
        status = run.status_of(c)
        g = run.guard.get(c.name)
        kb = ease(seg(t, at + 0.15, at + 0.75))
        if status == "ENFORCED":
            share = g.share(run.checked) if g else 0.0
            inv.append(Text.assemble(("◆ ", BLOCKED), (c.name, "text"), ("  ENFORCED", f"bold {BLOCKED}")))
            inv.append(Text.assemble(("  ", ""), _share_bar(share * kb),
                                     (f"  blocks {share * 100:.0f}% of states" if g and run.checked else "  blocks reachable states", "muted")))
        elif status == "NEVER FIRES":
            inv.append(Text.assemble(("○ ", "warn"), (c.name, "text"), ("  NEVER FIRES", "warn")))
            inv.append(Text.assemble(("  ", ""), _share_bar(0), ("  no reachable state breaks it", "muted")))
        else:
            inv.append(Text.assemble(("? ", "muted"), (c.name, "text"), ("  UNKNOWN", "muted")))
    inv = Group(*inv)
    parts += [Text.assemble(("GUARD RULES", "label"), ("   what each rule blocks at runtime", "muted")), inv, Text("")]

    # 4. blocked calls, typed out and stamped; then the verdict
    t_end = t_inv + 0.18 * len(run.constraints) + 0.4
    examples = _examples(run)
    if t >= t_end and examples:
        parts.append(Text.assemble(("BLOCKED CALLS", "label"), ("   reachable states the guard stops", "muted")))
        for j, (rule, values, _steps) in enumerate(examples):
            t0 = t_end + j * 1.1
            if t < t0:
                break
            call = "  ".join(f"{k}={v}" for k, v in values.items())
            shown = call[: int(len(call) * ease(seg(t, t0, t0 + 0.6)))]
            parts.append(Text.assemble(("  ", ""), (shown, "text"), ("▌" if t < t0 + 0.6 else "", "brand")))
            if t >= t0 + 0.6:
                stamp = Text("  ")
                stamp.append_text(shimmer(" ■ BLOCKED ", seg(t, t0 + 0.6, t0 + 1.0), f"bold #0f172a on {BLOCKED}",
                                          "bold #0f172a on #fdf4ff"))
                stamp.append(f" {rule}", style=BLOCKED)
                parts.append(stamp)
        parts.append(Text(""))
    if t >= t_end + 1.1 * len(examples):
        parts.append(_verdict(run, seg(t, t_end + 1.1 * len(examples), t_end + 1.1 * len(examples) + 0.6)))
    return Group(*parts)
