"""
`cslcore watch`: live dashboard of guard decisions.

Tails .csl/venom/decisions/*.jsonl, keeps a fixed ring buffer for the live stream and
bounded counters per agent and rule, and refreshes agent states on an interval through the
host probe. Exits cleanly on Ctrl-C.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Set, Tuple

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import VENOM_VERSION
from .commands import EXIT_OK, console_for, workspace_for
from .model import Inventory
from .policy.draft import agent_key
from .render.theme import STATE_GLYPH
from .render.words import n as _n

STREAM_SIZE = 200


@dataclass
class AgentStats:
    total: int = 0
    allow: int = 0
    would_block: int = 0
    block: int = 0
    errors: int = 0
    mode: Optional[str] = None
    last: Optional[str] = None

    @property
    def flagged(self) -> int:
        return self.would_block + self.block

    @property
    def rate(self) -> Optional[float]:
        return None if not self.total else self.flagged / self.total


@dataclass
class RuleStats:
    would_block: int = 0
    block: int = 0
    agents: Set[str] = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.would_block + self.block


class WatchModel:
    """Bounded state: counters per agent and rule, a ring buffer of recent decisions."""

    def __init__(self, stream_size: int = STREAM_SIZE) -> None:
        self.agents: Dict[str, AgentStats] = {}
        self.rules: Dict[str, RuleStats] = {}
        self.tools: Dict[Tuple[str, str], AgentStats] = {}
        self.stream: Deque[dict] = deque(maxlen=stream_size)
        self.minute: Deque[float] = deque(maxlen=100_000)
        self.total = 0
        self.errors = 0
        self.first_ts: Optional[datetime] = None
        self.last_ts: Optional[datetime] = None

    def ingest(self, rec: dict) -> None:
        agent = str(rec.get("agent", "?"))
        decision = str(rec.get("decision", "ALLOW"))
        st = self.agents.setdefault(agent, AgentStats())
        st.total += 1
        st.mode = rec.get("mode") or st.mode
        st.last = rec.get("ts")
        self.total += 1
        if decision == "WOULD_BLOCK":
            st.would_block += 1
        elif decision == "BLOCK":
            st.block += 1
        else:
            st.allow += 1
        if rec.get("error"):
            st.errors += 1
            self.errors += 1
        ts_ = self.tools.setdefault((agent, str(rec.get("tool") or "?")), AgentStats())
        ts_.total += 1
        if decision == "WOULD_BLOCK":
            ts_.would_block += 1
        elif decision == "BLOCK":
            ts_.block += 1
        else:
            ts_.allow += 1
        for r in rec.get("rules") or []:
            rs = self.rules.setdefault(str(r), RuleStats())
            rs.agents.add(agent)
            if decision == "BLOCK":
                rs.block += 1
            elif decision == "WOULD_BLOCK":
                rs.would_block += 1
        try:
            ts = datetime.fromisoformat(str(rec.get("ts")))
        except ValueError:
            ts = None
        if ts is not None:
            self.first_ts = self.first_ts or ts
            self.last_ts = ts if self.last_ts is None or ts > self.last_ts else self.last_ts
            self.minute.append(ts.timestamp())
        item = {k: rec.get(k) for k in ("ts", "agent", "tool", "decision", "rules")}
        item["_seen"] = time.monotonic()  # when it arrived here, for the live map's pulses
        self.stream.append(item)

    def per_minute(self, now: datetime) -> float:
        cutoff = now.timestamp() - 60
        while self.minute and self.minute[0] < cutoff - 3600:
            self.minute.popleft()
        return float(sum(1 for t in self.minute if t >= cutoff))

    @property
    def flagged(self) -> int:
        return sum(a.flagged for a in self.agents.values())


class Tail:
    """Follows every decision log in the workspace from where it last stopped."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.offsets: Dict[str, int] = {}

    def poll(self, model: WatchModel) -> int:
        """Read what every log gained since the last poll and ingest it in time order."""
        batch = []
        for path in self.ws.decision_logs():
            lines, self.offsets[str(path)] = self.ws.read_new(path, self.offsets.get(str(path), 0))
            for line in lines:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    batch.append(rec)
        batch.sort(key=lambda r: str(r.get("ts") or ""))
        for rec in batch:
            model.ingest(rec)
        return len(batch)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def _mode_label(model: WatchModel, modes: Dict[str, str]) -> str:
    seen = {a.mode for a in model.agents.values() if a.mode} | set(modes.values())
    if not seen:
        return "no agents wired"
    return f"{seen.pop()} mode" if len(seen) == 1 else "mixed modes"


def header_text(model: WatchModel, now: datetime, started: datetime, modes: Dict[str, str]) -> Text:
    t = Text()
    t.append(_mode_label(model, modes), style="brand")
    t.append(f" · {_fmt_dur((now - started).total_seconds())} · ", style="muted")
    t.append(f"{model.per_minute(now):.0f}/min", style="head")
    if model.total:
        t.append(f" · {model.total:,} checks · {model.flagged / model.total * 100:.1f}% flagged", style="muted")
    return t


class ControlPanel:
    """
    Interactive state of the management panel. Pure logic: keys in, control changes out.

    Focus is on the agents list or on the rules list (Tab). The right pane shows the live
    stream, the tools of the open agent, the actions for the open rule, or help. Enter opens,
    Esc always goes one level back. / searches agents. Text prompts (search, reasons) take
    the keyboard until Enter or Esc.
    """

    def __init__(self, controls, model: WatchModel, inv: Optional[Inventory], ws=None) -> None:
        self.controls = controls
        self.model = model
        self.inv = inv
        self.ws = ws if ws is not None else getattr(controls, "ws", None)
        self.focus = "agents"  # agents | rules
        self.view = "live"  # live | tools | rule | help
        self.selected = 0
        self.tool_index = 0
        self.rule_index = 0
        self.rule_agent_index = 0
        self.query = ""
        self.input: Optional[Dict[str, str]] = None  # {"kind", "prompt", "buffer", "target"}
        self.pending: Optional[Tuple[str, str, str]] = None  # (action, target, question)
        self.message: Optional[Tuple[str, str]] = None  # (text, style)
        self.editor_request: Optional[str] = None  # path the run loop opens in $EDITOR
        self.studio_request: Optional[Tuple[str, str]] = None
        self.map_on = False  # the live view shows the reach map instead of the decision stream
        self.go_map = False  # f: open the full map (a room, see venom/rooms.py)
        self.pane_zoom = 1.0  # the panel's map grows into the full map, and settles when it comes back
        self.marks_at = 0.0  # when the map's frozen / block marks were last read
        self.guard_request: Optional[str] = None  # an agent to put under a guard (outside the screen)
        self.limits_request: Optional[str] = None  # an agent whose limits change (outside the screen)
        self.wire_request: Optional[str] = None  # an agent to wire (outside the screen)
        self.rescan_request = False  # after unwiring: scan again, so every view shows it
        self.approval_index = 0
        self.topo = None
        self.topo_size: Optional[Tuple[int, int]] = None

    # data ------------------------------------------------------------------------------
    def all_agents(self) -> List[str]:
        """Agents with decisions first, then every agent set up or discovered (so they can be
        managed before their first call)."""
        names = list(self.model.agents)
        extra: List[str] = list(self.controls.all())
        if self.ws is not None:
            setup = (self.ws.load_state().get("setup") or {}).get("agents") or {}
            extra += [st.get("key") for st in setup.values() if st.get("key")]
        if self.inv is not None:
            extra += [agent_key(a) for a in self.inv.agents]
        for k in extra:
            if k and k not in names:
                names.append(k)
        return names

    def in_path(self, key: str) -> bool:
        """Whether the last scan saw a guard in this agent's call path (unknown agents: assume so)."""
        if self.inv is None:
            return True
        agent = next((a for a in self.inv.agents if agent_key(a) == key), None)
        return agent is None or agent.guard.status != "none"

    def approvals(self) -> list:
        """Calls waiting for a person's approval (approvals.Approvals), oldest first."""
        if self.ws is None:
            return []
        from .approvals import Approvals
        try:
            return Approvals(self.ws).pending()
        except OSError:
            return []

    def wired_here(self, key: str) -> bool:
        """Whether `cslcore wire` (or the board) changed this agent's files, so it can be undone."""
        return self.ws is not None and key in ((self.ws.load_state().get("wiring") or {}))

    def agents(self) -> List[str]:
        names = self.all_agents()
        q = self.query.lower()
        return [n for n in names if q in n.lower()] if q else names

    def current(self) -> Optional[str]:
        names = self.agents()
        if not names:
            return None
        self.selected = max(0, min(self.selected, len(names) - 1))
        return names[self.selected]

    def tools(self, agent: str) -> List[Tuple[str, str]]:
        """(tool, risk class) for the agent: discovered tools first, then tools seen in decisions."""
        out: List[Tuple[str, str]] = []
        if self.inv is not None:
            for a in self.inv.agents:
                if agent_key(a) == agent:
                    out = [(t.name, t.risk_class) for t in a.tools if not t.name.endswith("/*")]
        seen = {t for t, _ in out}
        for (ag, tool), _st in self.model.tools.items():
            if ag == agent and tool not in seen and not tool.startswith("("):
                out.append((tool, "?"))
        ctl = self.controls.get(agent)
        for tool in sorted(ctl.disabled_tools - {t for t, _ in out}):
            out.append((tool, "?"))
        return sorted(out, key=lambda tc: (RISK_ORDER.get(tc[1], 9), tc[0].lower()))

    def rules(self) -> List[Tuple[str, RuleStats]]:
        return [(r, rs) for r, rs in ranked_rules(self.model) if not r.startswith("__")]

    def current_rule(self) -> Optional[Tuple[str, RuleStats]]:
        rules = self.rules()
        if not rules:
            return None
        self.rule_index = max(0, min(self.rule_index, len(rules) - 1))
        return rules[self.rule_index]

    def crumbs(self) -> List[str]:
        if self.view == "help":
            return ["Help"]
        if self.view == "approvals":
            return ["Approvals"]
        if self.focus == "rules" or self.view == "rule":
            out = ["Rules"]
            cur = self.current_rule()
            if self.view == "rule" and cur:
                out.append(cur[0])
            return out
        out = ["Agents" + (f" matching '{self.query}'" if self.query else "")]
        if self.view == "tools" and self.current():
            out += [self.current() or "", "Tools"]
        elif self.map_on and self.view == "live":
            out.append("Map")
        return out

    def hints(self) -> List[Tuple[str, str]]:
        if self.input is not None:
            return [("Enter", "done"), ("Esc", "cancel")]
        if self.view == "help":
            return [("Esc", "back")]
        if self.view == "approvals":
            return [("↑↓", "request"), ("y", "approve"), ("n", "deny"), ("Esc", "back")]
        if self.view == "tools":
            return [("↑↓", "tool"), ("space", "disable / enable"), ("e", "exempt tool"), ("Esc", "back")]
        if self.view == "rule":
            return [("↑↓", "agent"), ("e", "exempt agent from rule"), ("o", "open policy"), ("Esc", "back")]
        if self.focus == "rules":
            return [("↑↓", "rule"), ("Enter", "open"), ("Tab", "agents"), ("Esc", "back")]
        out = [("↑↓", "select"), ("Enter", "tools"), ("l", "limits"), ("w", "wire"), ("m", "mode"), ("M", "all"),
               ("x", "freeze"), ("e", "exempt"),
               ("/", "search"), ("Tab", "rules"), ("g", "stream" if self.map_on else "map"), ("f", "full map"), ("?", "help")]
        waiting = len(self.approvals())
        if waiting:
            out.insert(0, ("a", f"{waiting} waiting for approval"))
        out.append(("Esc", "clear search") if self.query else ("q", "quit"))
        return out

    # keys ------------------------------------------------------------------------------
    def handle(self, key: str) -> bool:
        """Apply one key. Returns False when the panel should close."""
        if self.input is not None:
            return self._handle_input(key)
        if self.pending:
            action, target, _q = self.pending
            self.pending = None
            if key in ("y", "Y"):
                self._apply(action, target)
            else:
                self.message = ("cancelled", "muted")
            return True
        self.message = None
        if key == "ctrl-c":
            return False
        if key == "?":
            self.view = "live" if self.view == "help" else "help"
            return True
        if key in ("g", "G") and self.view == "live":
            self.map_on = not self.map_on
            return True
        if key in ("f", "F") and self.view == "live":
            self.go_map = True
            return True
        if key in ("esc", "left"):
            if self.view != "live":
                self.view = "live"
            elif self.map_on and key == "esc":
                self.map_on = False
            elif self.focus == "rules":
                self.focus = "agents"
            elif self.query:
                self.query = ""
                self.selected = 0
            return True
        if key in ("q", "Q") and self.view == "live":
            return False
        if key == "tab":
            self.focus = "rules" if self.focus == "agents" else "agents"
            self.view = "live"
            return True
        if key in ("a", "A") and self.view in ("live", "approvals"):
            self.view = "live" if self.view == "approvals" else "approvals"
            self.approval_index = 0
            return True
        if self.view == "approvals":
            return self._keys_approvals(key)
        if self.view == "tools":
            return self._keys_tools(key)
        if self.view == "rule":
            return self._keys_rule(key)
        if self.focus == "rules":
            return self._keys_rules(key)
        return self._keys_agents(key)

    def _handle_input(self, key: str) -> bool:
        inp = self.input
        assert inp is not None
        if key == "esc":
            if inp["kind"] == "search":
                self.query = inp.get("before", "")
            self.input = None
            return True
        if key == "enter":
            self.input = None
            self._submit(inp)
            return True
        if key in ("backspace", "\x7f", "\x08"):
            inp["buffer"] = inp["buffer"][:-1]
        elif len(key) == 1 and key.isprintable():
            inp["buffer"] += key
        if inp["kind"] == "search":
            self.query = inp["buffer"]
            self.selected = 0
        return True

    def ask_text(self, kind: str, prompt: str, target: str = "") -> None:
        self.input = {"kind": kind, "prompt": prompt, "buffer": "", "target": target, "before": self.query}

    def _submit(self, inp: Dict[str, str]) -> None:
        kind, text, target = inp["kind"], inp["buffer"].strip(), inp["target"]
        if kind == "search":
            self.query = text
            self.selected = 0
            return
        if not text:
            self.message = ("a reason is required for an exemption", "warn")
            return
        if kind == "reason_agent":
            self.pending = ("exempt_agent", f"{target}\x00{text}",
                            f"Exempt {target} completely? it is never blocked (still recorded). Reason: {text}")
        elif kind == "reason_tool":
            agent, tool = target.split("\x00", 1)
            self.pending = ("exempt_tool", f"{agent}\x00{tool}\x00{text}",
                            f"Exempt {tool} of {agent}? its rules are removed from the policy. Reason: {text}")
        elif kind == "reason_rule":
            agent, rule = target.split("\x00", 1)
            self.pending = ("exempt_rule", f"{agent}\x00{rule}\x00{text}",
                            f"Exempt {agent} from {rule}? the policy is changed and verified. Reason: {text}")

    def _keys_agents(self, key: str) -> bool:
        agent = self.current()
        if key in ("up", "k"):
            self.selected = max(0, self.selected - 1)
        elif key in ("down", "j"):
            self.selected = min(max(0, len(self.agents()) - 1), self.selected + 1)
        elif key == "/":
            self.ask_text("search", "search agents")
        elif agent and key == "m":
            to = "block" if self.controls.get(agent).mode == "log" else "log"
            q = (f"Switch {agent} to BLOCK mode? policy violations will be blocked" if to == "block"
                 else f"Switch {agent} to LOG mode? nothing will be blocked, only recorded")
            self.pending = (f"mode_{to}", agent, q)
        elif key == "M":
            to = "block" if (self.controls.default_mode() or "log") == "log" else "log"
            n = len(self.all_agents())
            q = (f"Switch ALL {_n(n, 'agent')} to BLOCK? policy violations will be blocked everywhere" if to == "block"
                 else f"Switch ALL {_n(n, 'agent')} to LOG? nothing will be blocked anywhere, only recorded")
            self.pending = (f"all_{to}", "*", q)
        elif agent and key in ("x", "d") and not self.in_path(agent):
            self.pending = ("guard", agent, f"{agent} is not wired: freezing it would stop nothing. "
                                            "Put it under a guard now (you see each step)?")
        elif agent and key in ("x", "d"):  # x everywhere; d was the key before 0.6.6
            if self.controls.get(agent).disabled:
                self._apply("enable", agent)
            else:
                self.pending = ("disable", agent, f"Freeze {agent}? every action is blocked, in any mode, until x again")
        elif agent and key == "l":
            self.limits_request = agent
        elif agent and key == "w":
            if self.wired_here(agent):
                self.pending = ("unwire", agent, f"Unwire {agent}? its files go back as they were and nothing "
                                                 "decides its calls any more")
            else:
                self.wire_request = agent
        elif agent and key == "e":
            if self.controls.get(agent).exempt:
                self.pending = ("unexempt_agent", agent, f"End the exemption of {agent}? its policy applies again")
            else:
                self.ask_text("reason_agent", f"why is {agent} fully trusted?", agent)
        elif agent and key in ("t", "enter", "right"):
            self.view = "tools"
            self.tool_index = 0
        return True

    def _keys_approvals(self, key: str) -> bool:
        from .approvals import TTL

        items = self.approvals()
        if not items:
            self.view = "live"
            self.message = ("nothing is waiting for approval", "muted")
            return True
        self.approval_index = max(0, min(self.approval_index, len(items) - 1))
        r = items[self.approval_index]
        what = f"{r['tool']} of {r['agent']}" + (f" ({_shown(r)})" if r.get("shown") else "")
        if key in ("up", "k"):
            self.approval_index = max(0, self.approval_index - 1)
        elif key in ("down", "j"):
            self.approval_index = min(len(items) - 1, self.approval_index + 1)
        elif key in ("y", "Y", "enter"):
            self.pending = ("approve", r["id"], f"Approve {what}? the same call runs once if it comes again within "
                                                f"{TTL // 60} minutes")
        elif key in ("n", "N", "d"):
            self.pending = ("deny", r["id"], f"Deny {what}? it stays stopped")
        return True

    def _keys_tools(self, key: str) -> bool:
        agent = self.current()
        if not agent:
            return True
        tools = self.tools(agent)
        if key in ("up", "k"):
            self.tool_index = max(0, self.tool_index - 1)
        elif key in ("down", "j"):
            self.tool_index = min(max(0, len(tools) - 1), self.tool_index + 1)
        elif key in (" ", "x") and tools:
            tool = tools[min(self.tool_index, len(tools) - 1)][0]
            if tool in self.controls.get(agent).disabled_tools:
                self._apply("enable_tool", f"{agent}\x00{tool}")
            else:
                self.pending = ("disable_tool", f"{agent}\x00{tool}", f"Disable {tool} for {agent}? every call to it will be blocked")
        elif key == "e" and tools:
            tool = tools[min(self.tool_index, len(tools) - 1)][0]
            self.ask_text("reason_tool", f"why can {tool} of {agent} run without its rules?", f"{agent}\x00{tool}")
        return True

    def _keys_rules(self, key: str) -> bool:
        if key in ("up", "k"):
            self.rule_index = max(0, self.rule_index - 1)
        elif key in ("down", "j"):
            self.rule_index = min(max(0, len(self.rules()) - 1), self.rule_index + 1)
        elif key in ("enter", "right") and self.current_rule():
            self.view = "rule"
            self.rule_agent_index = 0
        return True

    def _keys_rule(self, key: str) -> bool:
        cur = self.current_rule()
        if not cur:
            self.view = "live"
            return True
        rule, rs = cur
        agents = sorted(rs.agents)
        if key in ("up", "k"):
            self.rule_agent_index = max(0, self.rule_agent_index - 1)
        elif key in ("down", "j"):
            self.rule_agent_index = min(max(0, len(agents) - 1), self.rule_agent_index + 1)
        elif key == "e" and agents:
            agent = agents[min(self.rule_agent_index, len(agents) - 1)]
            self.ask_text("reason_rule", f"why should {rule} not apply to {agent}?", f"{agent}\x00{rule}")
        elif key == "o" and agents and self.ws is not None:
            from .policy.edit import policy_path_for
            path, why = policy_path_for(self.ws, agents[min(self.rule_agent_index, len(agents) - 1)])
            if path is None:
                self.message = (why, "warn")
            else:
                # the studio edits a draft copy; it becomes active only on "go live" there
                self.studio_request = (str(path), agents[min(self.rule_agent_index, len(agents) - 1)])
        return True

    # actions ---------------------------------------------------------------------------
    def _apply(self, action: str, target: str) -> None:
        if action.startswith("all_"):
            mode = action[4:]
            self.controls.set_all(mode)
            self.message = (f"ALL agents → {mode.upper()} mode", "ok" if mode == "log" else "brand")
            return
        if action == "guard":
            self.guard_request = target
            return
        if action in ("approve", "deny"):
            from .approvals import Approvals
            from .controls import _who
            ok = Approvals(self.ws).decide(target, action == "approve", _who())
            self.message = (("approved: the call runs once when it comes again" if action == "approve"
                             else "denied: it stays stopped", "ok" if action == "approve" else "muted") if ok
                            else ("that request is no longer waiting", "warn"))
            if ok:
                self.controls._audit("*", action, target)
            return
        if action == "unwire":
            from . import wiring
            results = wiring.undo(self.ws, target)
            kept = [p for _k, p, r in results if r == "skipped"]
            self.wire_request = None
            self.message = ((f"{target} unwired: its files are as they were; nothing decides its calls now", "warn")
                            if not kept else (f"{target}: {len(kept)} file(s) changed since and were left as they are",
                                              "high"))
            self.rescan_request = True
            return
        if action.startswith("mode_"):
            self.marks_at = 0.0
            mode = action[5:]
            self.controls.set_mode(target, mode)
            self.message = (f"{target} → {mode.upper()} mode (next call, no restart)", "ok" if mode == "log" else "brand")
        elif action == "disable":
            self.marks_at = 0.0
            self.controls.set_disabled(target, True)
            self.message = (f"{target} FROZEN: every action is blocked (x unfreezes)", "high")
        elif action == "enable":
            self.marks_at = 0.0
            self.controls.set_disabled(target, False)
            self.message = (f"{target} unfrozen", "ok")
        elif action in ("disable_tool", "enable_tool"):
            agent, tool = target.split("\x00", 1)
            self.controls.set_tool(agent, tool, action == "disable_tool")
            self.message = (f"{agent} / {tool} {'disabled' if action == 'disable_tool' else 'enabled'}",
                            "high" if action == "disable_tool" else "ok")
        elif action == "exempt_agent":
            agent, reason = target.split("\x00", 1)
            self.controls.set_exempt(agent, True, reason)
            self.message = (f"{agent} EXEMPT: never blocked, still recorded", "exempt")
        elif action == "unexempt_agent":
            self.controls.set_exempt(target, False)
            self.message = (f"{target}: exemption ended, its policy applies again", "ok")
        elif action == "activate_draft":
            from datetime import datetime as _dt
            draft, active = target.split("\x00", 1)
            from pathlib import Path as _P
            stamp = _dt.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            old = self.ws.read(active) or ""
            self.ws.write_text(self.ws.venom / "history" / f"{_P(active).stem}-{stamp}.csl", old)
            self.ws.write_text(_P(active), self.ws.read(draft) or "")
            self.ws.remove(_P(draft))
            self.controls._audit(_P(active).stem, "policy edited", _P(active).name)
            self.message = (f"{_P(active).name} active; running agents reload it on their next call", "ok")
        elif action in ("exempt_tool", "exempt_rule"):
            self.message = self._edit_policy(action, *target.split("\x00"))

    def _edit_policy(self, action: str, agent: str, item: str, reason: str) -> Tuple[str, str]:
        from .controls import _who
        from .model import Exemption
        from .policy import edit

        if self.ws is None:
            return ("no workspace", "high")
        path, why = edit.policy_path_for(self.ws, agent)
        if path is None:
            return (why, "warn")
        text = self.ws.read(path) or ""
        if action == "exempt_rule":
            e = Exemption(agent=agent, scope="rule", rule=item, reason=reason, approved_by=_who(), status="approved")
            new, what = edit.exempt_agent_from_rule(text, item, agent, edit.note_for(e))
        else:
            e = Exemption(agent=agent, scope="tool", tool=item, reason=reason, approved_by=_who(), status="approved")
            rules = edit.rules_for_tool(text, item)
            if not rules:
                return (f"{item} has no rule in {path.name}: nothing to exempt", "muted")
            new, what = edit.remove_rules(text, rules, edit.note_for(e)), f"{len(rules)} rule(s) of {item} removed"
        res = edit.apply(self.ws, path, new, e)
        if res.ok:
            self.controls._audit(agent, "exempt " + ("rule" if action == "exempt_rule" else "tool"), f"{item}: {reason}")
            return (f"{what}; {res.message}", "ok")
        return (res.message, "high")


def agents_pane(model: WatchModel, inv: Optional[Inventory], states: Dict[str, str], width: int,
                panel: Optional["ControlPanel"] = None, rows: int = 14) -> Table:
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 1, 0, 0), expand=True)
    t.add_column("AGENTS", no_wrap=True, overflow="ellipsis", ratio=1)
    wide = width >= 110
    t.add_column("mode", no_wrap=True, width=5)
    if wide:
        t.add_column("checks", justify="right", no_wrap=True, width=6)
    t.add_column("block%", justify="right", no_wrap=True, width=6)
    names = panel.agents() if panel is not None else list(model.agents) + [k for k in sorted(states) if k not in model.agents]
    ctls = {n: panel.controls.get(n) for n in names} if panel is not None else {}
    current = panel.current() if panel is not None else None
    rows = max(3, rows)
    first = 0
    if panel is not None and len(names) > rows:
        first = max(0, min(panel.selected - rows // 2, len(names) - rows))
    if first > 0:
        t.add_row(Text(f"  ↑ {first} more", style="muted"), "", *([""] * (2 if wide else 1)))
        first += 1
    visible = names[first:first + rows - (1 if first else 0) - (1 if len(names) > first + rows else 0)]
    for name in visible:
        st = model.agents.get(name, AgentStats())
        state = states.get(name, "running" if st.total else "unknown")
        ctl = ctls.get(name)
        glyph = Text(STATE_GLYPH.get(state, "·") + " ", style=state if state in ("running", "stopped", "scheduled", "configured") else "muted")
        rate = st.rate
        rate_t = Text("-" if rate is None else f"{rate * 100:.0f}%",
                      style="muted" if rate is None else ("high" if rate > 0.25 else ("warn" if rate > 0.05 else "ok")))
        if panel is not None and not panel.in_path(name):
            mode_t = Text("NONE", style="muted")  # no guard in its call path: nothing is decided
        elif ctl is not None and ctl.disabled:
            mode_t = Text("OFF", style="high")
        elif ctl is not None and ctl.exempt:
            mode_t = Text("EXEMP", style="exempt")
        else:
            mode = (ctl.mode if ctl is not None else st.mode) or "-"
            mode_t = Text(mode.upper()[:5], style="warn" if mode == "log" else ("brand" if mode == "block" else "muted"))
        mark = Text("▸ " if name == current and panel is not None and panel.focus == "agents" else "  ", style="brand")
        label = mark + glyph + Text(name, style="head")
        if ctl is not None and ctl.disabled_tools:
            label.append(f" -{len(ctl.disabled_tools)}", style="high")
        row = [label, mode_t] + ([Text(f"{st.total:,}", style="text")] if wide else []) + [rate_t]
        sel = name == current and panel is not None and panel.focus == "agents"
        t.add_row(*row, style="selected" if sel else None)
    rest = len(names) - first - len(visible)
    if rest > 0:
        t.add_row(Text(f"  ↓ {rest} more", style="muted"), "", *([""] * (2 if wide else 1)))
    if not names and panel is not None and panel.query:
        t.add_row(Text(f"  no agent matches '{panel.query}'", style="muted"), "", *([""] * (2 if wide else 1)))
    return t


RISK_ORDER = {c: i for i, c in enumerate(["DESTRUCTIVE", "SPEND", "EXEC", "IDENTITY", "UNCLASSIFIED", "EXTERNAL", "WRITE", "READ", "?"])}


def _shown(r: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted((r.get("shown") or {}).items()) if k not in ("agent_id", "tool"))


def approvals_pane(panel: "ControlPanel") -> Table:
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 2, 0, 0))
    t.add_column("", no_wrap=True)
    t.add_column("AGENT", style="head", no_wrap=True)
    t.add_column("TOOL", style="text", no_wrap=True)
    t.add_column("CALL", style="muted")
    t.add_column("SINCE", style="muted", no_wrap=True)
    items = panel.approvals()
    for i, r in enumerate(items):
        mark = Text("▸", style="brand") if i == panel.approval_index else Text(" ")
        t.add_row(mark, r["agent"], r["tool"], _shown(r) or "·", str(r.get("at", ""))[11:16])
    if not items:
        t.add_row("", Text("nothing is waiting", style="muted"), "", "", "")
    return t


def tools_pane(panel: "ControlPanel", agent: str, rows: int = 12) -> Table:
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 1, 0, 0), expand=True)
    t.add_column(f"TOOLS · {agent}", no_wrap=True, overflow="ellipsis", ratio=3)
    t.add_column("class", no_wrap=True, width=12)
    t.add_column("calls", justify="right", no_wrap=True, width=6)
    t.add_column("", no_wrap=True, width=9)
    disabled = panel.controls.get(agent).disabled_tools
    tools = panel.tools(agent)
    first = max(0, min(panel.tool_index - rows // 2, len(tools) - rows))
    for i, (tool, cls) in enumerate(tools[first:first + rows], first):
        st = panel.model.tools.get((agent, tool), AgentStats())
        status = Text("DISABLED", style="high") if tool in disabled else Text("on", style="ok")
        mark = Text("▸ " if i == panel.tool_index else "  ", style="brand")
        t.add_row(mark + Text(tool, style="head"), Text(cls, style=f"risk.{cls}" if cls != "?" else "muted"),
                  Text(f"{st.total:,}", style="text"), status, style="selected" if i == panel.tool_index else None)
    if not panel.tools(agent):
        t.add_row(Text("no tools known for this agent yet", style="muted"), "", "", "")
    return t


def help_pane() -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="brand", no_wrap=True)
    t.add_column(style="text")
    for k, v in (("↑ ↓", "move"), ("Enter", "open (an agent's tools, a rule's actions)"), ("Esc", "back one level"),
                 ("Tab", "switch between agents and rules"), ("/", "search agents"),
                 ("m / M", "switch the agent / ALL agents between LOG and BLOCK"),
                 ("a", "calls waiting for a person's approval: y approves (it runs once), n denies"),
                 ("l", "the agent's limits: change them; its policy, mapping and check follow"),
                 ("w", "wire the agent (the guard into its call path), or unwire it (its files as they were)"),
                 ("x", "freeze the agent (blocks every action in any mode); x again unfreezes it"),
                 ("e", "exempt the agent (or, in its tools, one tool); a reason is required"),
                 ("space", "in tools: disable or enable one tool"),
                 ("e / o", "in a rule: exempt one agent from it / open the policy in your editor"),
                 ("g / f", "the reach map beside the decisions (g again: the stream) / the full map (w comes back)"),
                 ("q", "quit"), ("", ""), ("", "Changes reach running agents on their next tool call, without a restart."),
                 ("", "Every change is recorded in .csl/venom/audit.jsonl; policy edits keep the old version.")):
        t.add_row(k, v)
    return t


def stream_pane(model: WatchModel, rows: int, wide: bool = True) -> Table:
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 1, 0, 0), expand=True)
    t.add_column("LIVE", no_wrap=True, width=8)
    t.add_column("", no_wrap=True, overflow="ellipsis", ratio=2)
    t.add_column("", no_wrap=True, overflow="ellipsis", ratio=3)
    t.add_column("", no_wrap=True, width=11, justify="right")
    items = list(model.stream)[-max(1, rows):][::-1]
    for r in items:
        ts = str(r.get("ts") or "")
        clock = ts[11:19] if len(ts) >= 19 else ts
        d = str(r.get("decision") or "")
        style = {"ALLOW": "ok", "WOULD_BLOCK": "warn", "BLOCK": "high"}.get(d, "text")
        t.add_row(Text(clock, style="muted"), Text(str(r.get("agent") or ""), style="head"),
                  Text(str(r.get("tool") or ""), style="text"), Text(d.replace("_", " "), style=style))
        rules = r.get("rules") or []
        if rules and d != "ALLOW" and wide:
            t.add_row("", "", Text("↳ " + ", ".join(rules), style="muted"), "")
    if not items:
        t.add_row("", Text("waiting for decisions ...", style="muted"), "", "")
    return t


def ranked_rules(model: WatchModel) -> List[Tuple[str, RuleStats]]:
    return sorted(model.rules.items(), key=lambda kv: (-kv[1].total, kv[0]))


def tuning_pane(model: WatchModel, limit: int = 5, wide: bool = True, panel: Optional["ControlPanel"] = None):
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 2, 0, 0), expand=True)
    t.add_column("TUNING", justify="right", no_wrap=True, width=7)
    t.add_column("rules by would-block", no_wrap=True, overflow="ellipsis", ratio=3)
    t.add_column("", no_wrap=True, overflow="ellipsis", ratio=2)
    if wide:
        t.add_column("Tab: tune a rule" if panel is not None else "",
                     no_wrap=True, overflow="ellipsis", ratio=3, header_style="brand.dim")
    ranked = panel.rules() if panel is not None else ranked_rules(model)
    first = 0
    if panel is not None and panel.focus == "rules" and panel.rule_index >= limit:
        first = panel.rule_index - limit + 1
    for i, (rule, rs) in enumerate(ranked[first:first + limit], first):
        agents = sorted(rs.agents)
        sel = panel is not None and panel.focus == "rules" and i == panel.rule_index
        name = Text("▸ " if sel else "", style="brand") + Text(rule, style="head")
        row = [Text(f"{rs.total:,}", style="warn" if rs.would_block else "high"), name,
               Text(", ".join(agents), style="text")]
        if wide:
            hint = "Enter: exempt an agent, or o: open in the studio" if panel is not None else (f"cslcore studio --agent {agents[0]}" if agents else "")
            row.append(Text(hint if sel or panel is None else "", style="brand.dim"))
        t.add_row(*row, style="selected" if sel else None)
    if not ranked:
        t.add_row("", Text("no rule has fired yet", style="muted"), "", *([""] if wide else []))
    if wide or not ranked or panel is not None:
        return t
    return Group(t, Text("tune a rule: cslcore studio --agent <agent>", style="brand.dim"))


def rule_pane(panel: "ControlPanel") -> Table:
    cur = panel.current_rule()
    t = Table.grid(padding=(0, 1))
    t.add_column(no_wrap=True)
    t.add_column(overflow="fold")
    if not cur:
        t.add_row("", Text("no rule selected", style="muted"))
        return t
    rule, rs = cur
    t.add_row(Text("rule", style="label"), Text(rule, style="head"))
    t.add_row(Text("fired", style="label"), Text(f"{rs.would_block:,} would block · {rs.block:,} blocked", style="text"))
    t.add_row("", "")
    t.add_row(Text("agents", style="label"), Text("choose one, then x to exempt it from this rule", style="muted"))
    for i, a in enumerate(sorted(rs.agents)):
        sel = i == panel.rule_agent_index
        t.add_row(Text("▸" if sel else " ", style="brand"), Text(a, style="head" if sel else "text"),
                  style="selected" if sel else None)
    t.add_row("", "")
    t.add_row(Text("x", style="brand"), Text("exempt the agent from this rule (a reason is asked; the policy is re-verified)", style="text"))
    t.add_row(Text("o", style="brand"), Text("open the agent's policy in the studio (relax the rule, verify, go live; guards reload)", style="text"))
    return t


def footer_text(model: WatchModel, inv: Optional[Inventory]) -> Text:
    t = Text()
    t.append(f"{model.total:,} checks", style="head")
    pct = f" ({model.flagged / model.total * 100:.1f}%)" if model.total else ""
    t.append(f" · {model.flagged:,} flagged{pct}", style="warn" if model.flagged else "muted")
    t.append(f" · {model.errors} errors", style="high" if model.errors else "muted")
    if inv is not None:
        r = inv.coverage.ratio
        t.append(f" · coverage {'n/a' if r is None else f'{r * 100:.0f}%'}", style="text")
        high = sum(1 for f in inv.findings if f.severity == "high")
        t.append(f" · {high} high findings", style="high" if high else "ok")
    return t


MAP_FRAME_S = 1 / 15  # the live map draws about 15 times a second


def map_pane(panel: "ControlPanel", model: WatchModel, inv: Inventory, cols: int, rows: int) -> Text:
    """The reach map with live decisions flowing on it (see Topo._pulses)."""
    from .reach import build
    from .render.mapview import TOOL_IMPACT
    from .render.topo import Topo

    if panel.topo is None or panel.topo_size != (cols, rows):
        panel.topo = Topo(build(inv), w=cols, h=rows, max_agents=12, seed=VENOM_VERSION)
        panel.topo_size, panel.marks_at = (cols, rows), 0.0
    by_key = {agent_key(a): a for a in inv.agents}
    now = time.monotonic()
    if now - panel.marks_at > 1.0:  # frozen and block-mode agents look as they do on the full map
        from .bindings import Bindings
        bound = set(Bindings(panel.ws).all()) if panel.ws is not None else set()
        marks = {}
        for k, a in by_key.items():
            ctl = panel.controls.get(k)
            if ctl.disabled:
                marks[a.id] = "frozen"
            elif ctl.mode == "block" and k in bound:
                marks[a.id] = "block"
        panel.topo.marks, panel.marks_at = marks, now
    pulses = []
    for rec in list(model.stream)[-60:]:
        age = now - rec.get("_seen", 0)
        if age > 1.2:
            continue
        a = by_key.get(str(rec.get("agent")))
        if a is None:
            continue
        tool = next((t for t in a.tools if t.name == rec.get("tool")), None)
        target = TOOL_IMPACT.get(tool.risk_class) if tool is not None else None
        pulses.append((a.id, target, str(rec.get("decision", "ALLOW")), age))
    sel = by_key.get(panel.current() or "")
    view = None
    if panel.pane_zoom != 1.0:  # growing into the full map, or settling after it
        from .render.topo import Zoom
        view = Zoom(panel.topo.W / 2, panel.topo.H / 2, panel.pane_zoom, panel.topo.W, panel.topo.H)
    return panel.topo.render(now, complete=True, selected=sel.id if sel else None, pulses=pulses, view=view)


def render(model: WatchModel, inv: Optional[Inventory], states: Dict[str, str], modes: Dict[str, str],
           now: datetime, started: datetime, width: int, height: int, panel: Optional["ControlPanel"] = None) -> Layout:
    title = Text.assemble((" CSL-Core watch ", "brand"), (VENOM_VERSION + " ", "muted"))
    root = Layout()
    body_h = max(6, height - 3 - 9 - 3)
    root.split_column(Layout(name="header", size=3), Layout(name="body", size=body_h),
                      Layout(name="tuning", size=9), Layout(name="footer", size=3))
    head = header_text(model, now, started, modes)
    if panel is not None:
        crumbs = Text("   ")
        for i, c in enumerate(panel.crumbs()):
            if i:
                crumbs.append(" › ", style="muted")
            crumbs.append(c, style="head" if i == len(panel.crumbs()) - 1 else "text")
        head = head + crumbs
    root["header"].update(Panel(head, title=title, title_align="left",
                                box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
    agents_focus = panel is not None and panel.focus == "agents" and panel.view in ("live", "tools")
    recording = log_only(panel)
    pane = agents_pane(model, inv, states, width, panel, rows=body_h - 3 - len(recording))
    left = Panel(Group(pane, *recording) if recording else pane, box=box.ROUNDED,
                 border_style="brand.dim" if agents_focus else "muted", padding=(0, 1))
    wide = width >= 110
    right = Panel(stream_pane(model, body_h - 3, wide), box=box.ROUNDED, border_style="muted", padding=(0, 1))
    if not model.total:
        right = Panel(Group(Text("no decisions yet", style="head"), Text(""),
                            Text("Wire an agent with ", style="muted") + Text("cslcore setup", style="brand"),
                            Text("(it starts in log mode: nothing is blocked, every", style="muted"),
                            Text("decision appears here as ALLOW or WOULD BLOCK).", style="muted")),
                      box=box.ROUNDED, border_style="muted", padding=(0, 1))
    if panel is not None and panel.view == "live" and panel.map_on and inv is not None:
        cols = max(30, int(width * (3 / 5 if wide else 6 / 11)) - 8)
        right = Panel(map_pane(panel, model, inv, cols, max(8, body_h - 4)), title=Text(" reach map · live ", style="label"),
                      title_align="left", box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))
    elif panel is not None and panel.view == "tools" and panel.current():
        right = Panel(tools_pane(panel, panel.current(), body_h - 4), box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))
    elif panel is not None and panel.view == "rule":
        right = Panel(rule_pane(panel), title=Text(" rule ", style="label"), title_align="left", box=box.ROUNDED,
                      border_style="brand.dim", padding=(0, 1))
    elif panel is not None and panel.view == "approvals":
        right = Panel(approvals_pane(panel), title=Text(" waiting for approval ", style="label"), title_align="left",
                      box=box.ROUNDED, border_style="warn", padding=(0, 1))
    elif panel is not None and panel.view == "help":
        right = Panel(help_pane(), title=Text(" keys ", style="label"), box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))
    body = Layout()
    body.split_row(Layout(left, name="agents", ratio=2 if wide else 5), Layout(right, name="live", ratio=3 if wide else 6))
    root["body"].update(body)
    root["tuning"].update(Panel(tuning_pane(model, wide=wide, panel=panel), box=box.ROUNDED,
                                border_style="brand.dim" if panel is not None and panel.focus == "rules" else "muted", padding=(0, 1)))
    foot = footer_text(model, inv)
    if panel is not None:
        if panel.input is not None:
            foot = Text.assemble((panel.input["prompt"] + ": ", "label"), (panel.input["buffer"], "head"), ("▌", "brand"))
        elif panel.pending:
            foot = Text.assemble((panel.pending[2], "warn"), ("   [y/n]", "head"))
        elif panel.message:
            foot = Text(panel.message[0], style=panel.message[1])
        else:
            foot = Text()
            for k, v in panel.hints():
                foot.append(k + " ", style="brand")
                foot.append(v + "   ", style="muted")
            foot.rstrip()
    border = "warn" if panel is not None and panel.pending else ("brand" if panel is not None and panel.input is not None else "muted")
    root["footer"].update(Panel(foot, box=box.ROUNDED, border_style=border, padding=(0, 1)))
    return root


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------

def _inventory(ws) -> Optional[Inventory]:
    data = ws.latest_inventory()
    return Inventory.from_dict(data) if data else None


def _states(inv: Optional[Inventory], probe=None) -> Dict[str, str]:
    states: Dict[str, str] = {}
    if inv is None:
        return states
    running: List[str] = []
    if probe is not None:
        r = probe.run(["ps", "-eo", "args="])
        running = r.stdout.splitlines() if r is not None and r.returncode == 0 else []
    for a in inv.agents:
        state = a.state
        if running and a.entrypoint and a.kind != "assistant":
            state = "running" if any(a.entrypoint in line for line in running) else ("scheduled" if a.state == "scheduled" else "stopped")
        states[agent_key(a)] = state
    return states


def log_only(panel: Optional["ControlPanel"], limit: int = 3) -> List[Text]:
    """One line for each wired agent in log mode: it records only, and how to make it stop."""
    if panel is None or panel.ws is None:
        return []
    from .bindings import Bindings

    out = []
    for key in sorted(Bindings(panel.ws).all()):
        if panel.controls.get(key).mode == "log" and panel.in_path(key) and not panel.controls.get(key).disabled:
            out.append(Text.assemble((f"{key} records only; block it: ", "warn"),
                                     (f"cslcore mode --agent {key} block", "brand"), (" (or m)", "muted")))
    return out[:limit]


def _modes(ws) -> Dict[str, str]:
    return {k: v.get("mode") for k, v in (ws.load_state().get("modes") or {}).items() if isinstance(v, dict)}


GROW_S = 0.35  # the panel's map grows into the full map
SETTLE_S = 0.3  # and settles when the map shrinks back into it


class WatchRoom:
    """The live panel as a room (see venom/rooms.py): the same panel, polled and drawn here."""

    def __init__(self, args, console) -> None:
        from .controls import Controls
        from .probe import LocalHostProbe

        self.args, self.console = args, console
        self.ws = workspace_for(args)
        self.model = WatchModel()
        self.tail = Tail(self.ws)
        self.tail.poll(self.model)
        self.inv = _inventory(self.ws)
        self.started = datetime.now(timezone.utc)
        self.panel = ControlPanel(Controls(self.ws), self.model, self.inv, self.ws)
        self.probe = LocalHostProbe()
        self.states = _states(self.inv, self.probe)
        self.last_states = time.monotonic()
        self.modes = _modes(self.ws)
        self.refresh = max(0.2, float(getattr(args, "refresh", 1.0) or 1.0))
        self.next_poll = self.next_frame = 0.0
        self.last = None
        self.grow0: Optional[float] = None
        self.settle0: Optional[float] = None
        self.exit_to: Optional[str] = None
        self.external = None

    def enter(self, came_from: str) -> None:
        """The panel always opens on what it is for: the live decisions. Back from the full map,
        it is as it was left; its own map, if it was open, settles into place."""
        if self.panel.map_on:
            self.settle0 = time.monotonic()
        self.next_frame = 0.0

    def wait(self) -> float:
        return 0.03 if self.panel.map_on else 0.05

    def handle(self, key: str) -> bool:
        if self.grow0 is not None:
            return True
        ok = self.panel.handle(key)
        if self.panel.go_map:
            self.panel.go_map = False
            if self.panel.map_on:
                self.grow0 = time.monotonic()  # the panel's map grows into the full map
            else:
                self.exit_to = "map"
        if getattr(self.panel, "rescan_request", False):
            self.panel.rescan_request = False
            self.external = self._rescan
        if (self.panel.studio_request or self.panel.editor_request or self.panel.guard_request
                or self.panel.limits_request or self.panel.wire_request):
            self.external = self._outside
        self.next_frame = self.next_poll = 0.0
        return ok

    def frame(self, now: float, width: int, height: int):
        moving = self.grow0 is not None or self.settle0 is not None
        animated = (self.panel.map_on and self.panel.view == "live") or moving
        if now >= self.next_poll:
            self.tail.poll(self.model)
            self.modes = _modes(self.ws)
            if now - self.last_states > 15:
                self.inv = _inventory(self.ws)
                self.panel.inv = self.inv
                self.states = _states(self.inv, self.probe)
                self.last_states = now
            # the live map is an animation: decisions are read every 0.2 s so they arrive spread out
            self.next_poll = now + (0.2 if animated else self.refresh)
        if self.grow0 is not None:
            k = min(1.0, (now - self.grow0) / GROW_S)
            self.panel.pane_zoom = 1 + 0.9 * k * k
            if k >= 1.0:
                self.grow0, self.panel.pane_zoom, self.exit_to = None, 1.0, "map"
        elif self.settle0 is not None:
            k = min(1.0, (now - self.settle0) / SETTLE_S)
            self.panel.pane_zoom = 0.82 + 0.18 * (1 - (1 - k) ** 3)
            if k >= 1.0:
                self.settle0, self.panel.pane_zoom = None, 1.0
        if self.last is None or now >= self.next_frame or moving:
            self.last = render(self.model, self.inv, self.states, self.modes, datetime.now(timezone.utc), self.started,
                               width, height, self.panel)
            self.next_frame = now + (MAP_FRAME_S if animated else self.refresh)
        return self.last

    def _guard_it(self, key: str) -> None:
        from rich.prompt import Prompt

        from .wire_cmd import guard_one

        agent = next((a for a in (self.inv.agents if self.inv else []) if agent_key(a) == key), None)
        if agent is None:
            self.panel.message = (f"{key} is not in the last scan", "warn")
            return
        ok = guard_one(self.console, self.args, self.ws, agent)
        self.inv = _inventory(self.ws)
        self.panel.inv = self.inv
        self.panel.topo = None  # the map follows the new scan
        if ok:
            self.panel.controls.set_disabled(key, True)
            self.console.print(f"  [high]{key} FROZEN[/high] [muted]every action is blocked until x again[/muted]")
            self.panel.message = (f"{key} is guarded now, and FROZEN (x unfreezes)", "high")
        else:
            self.panel.message = (f"{key} is still not wired", "warn")
        Prompt.ask("  [muted]Enter: back to the live panel[/muted]", default="", show_default=False, console=self.console)

    def _rescan(self) -> None:
        from .wire_cmd import rescan, scan_probe

        _probe, root = scan_probe(self.args, self.ws)
        rescan(self.args, self.console, self.ws, root)
        self._follow_scan()

    def _follow_scan(self) -> None:
        self.inv = _inventory(self.ws)
        self.panel.inv = self.inv
        self.panel.topo = None  # the map follows the new scan
        self.states = _states(self.inv, self.probe)
        self.next_frame = 0.0

    def _limits_of(self, key: str, wire_only: bool = False) -> None:
        """l: the agent's limits, and the policy, mapping, wiring and check that follow (the board's
        loop); w: only its wiring. Running agents take the new policy on their next call."""
        from rich.prompt import Prompt

        from . import board as B
        from .wire_cmd import rescan, scan_probe, show_plan

        agent = next((a for a in (self.inv.agents if self.inv else []) if agent_key(a) == key), None)
        if agent is None:
            self.panel.message = (f"{key} is not in the last scan", "warn")
            return
        ui = B.TerminalUI(self.console)
        if not wire_only:
            ok = B.protect(ui, self.console, self.args, self.ws, agent)
            self.panel.message = ((f"{key}: new limits active; running agents use them from their next call", "ok") if ok
                                  else (f"{key}: the limits are not active (see above)", "warn"))
        else:
            from . import wiring
            from .bindings import Bindings

            if Bindings(self.ws).get(key) is None:
                self.console.print(f"  [warn]{key} has no policy yet: l sets its limits and makes one[/warn]")
                self.panel.message = (f"{key} has no policy yet: press l", "warn")
            else:
                plan = B.wire_plan(self.args, self.ws, agent)
                if plan.kind == "done":
                    self.panel.message = (f"{key}: {plan.note}", "ok")
                elif plan.kind == "manual" or not plan.changes:
                    self.console.print(Text("  " + (plan.note or "nothing to change"), style="warn"))
                    self.panel.message = (f"{key} cannot be wired automatically (see .csl/venom/wiring.md)", "warn")
                else:
                    show_plan(self.console, plan, ws=self.ws)
                    from .wire_cmd import env_ready
                    if not env_ready(self.console, self.args, self.ws, agent, plan, ui.ask):
                        self.panel.message = (f"{key} not wired: csl-core is not installed where it runs", "warn")
                    elif ui.ask(f"Wire {key}?", True):
                        wiring.apply(plan, self.ws)
                        _probe, root = scan_probe(self.args, self.ws)
                        rescan(self.args, self.console, self.ws, root)
                        self.panel.message = (f"{key} wired: its calls go through the guard", "ok")
                    else:
                        self.panel.message = ("not wired", "muted")
        self._follow_scan()
        Prompt.ask("  [muted]Enter: back to the live panel[/muted]", default="", show_default=False, console=self.console)

    def _outside(self) -> None:
        """The studio, $EDITOR, limits, wiring and putting an agent under a guard run outside the
        screen; the panel shows what came of it."""
        panel, ws = self.panel, self.ws
        if panel.guard_request:
            key, panel.guard_request = panel.guard_request, None
            self._guard_it(key)
        if panel.limits_request:
            key, panel.limits_request = panel.limits_request, None
            self._limits_of(key)
        if panel.wire_request:
            key, panel.wire_request = panel.wire_request, None
            self._limits_of(key, wire_only=True)
        if panel.studio_request:
            path, agent = panel.studio_request
            panel.studio_request = None
            from .studio.command import launch
            try:
                msg = launch(ws, self.inv, path=path, agent=agent)
            except Exception as e:  # never take the panel down with the editor
                msg = f"studio closed: {type(e).__name__}: {e}"
            panel.message = (msg or "studio closed, nothing changed", "ok" if msg.startswith("live") else "muted")
        if panel.editor_request:
            draft, active = panel.editor_request.split("\x00", 1)
            panel.editor_request = None
            from pathlib import Path as _Path
            from .policy.gate import verify_text
            ws.open_in_editor(_Path(draft))
            after = ws.read(draft) or ""
            if after == (ws.read(active) or ""):
                ws.remove(_Path(draft))
                panel.message = ("no changes", "muted")
            else:
                g = verify_text(after)
                if g.ok:
                    panel.pending = ("activate_draft", f"{draft}\x00{active}",
                                     f"{_Path(draft).name} verified ({_n(g.rules, 'rule')}). Activate it now?")
                else:
                    issue = g.issues[0].message if g.issues else g.stage
                    panel.message = (f"{_Path(draft).name} does not verify ({issue}); kept as a draft, "
                                     "the active policy is unchanged", "high")
        self.next_frame = 0.0


def run_watch(args) -> int:
    import sys

    from .controls import Controls

    console = console_for(args)
    ws = workspace_for(args)
    if getattr(args, "once", False) or not console.is_terminal or not sys.stdin.isatty():
        model = WatchModel()
        Tail(ws).poll(model)
        inv = _inventory(ws)
        started = datetime.now(timezone.utc)
        panel = ControlPanel(Controls(ws), model, inv, ws)
        console.print(render(model, inv, _states(inv), _modes(ws), started, started, console.width, 30, panel), height=30)
        return EXIT_OK
    from . import rooms

    made: Dict[str, object] = {}
    rooms.run(console, args, "watch", made=made)
    room = made.get("watch")
    seen = room.model.total if isinstance(room, WatchRoom) else 0
    console.print(f"  [muted]watch stopped · {seen:,} checks seen[/muted]")
    return EXIT_OK
