"""
`cslcore studio`: write, prove and ship a CSL policy.

    F5 / ctrl+r   verify with Z3            ctrl+s  save the draft
    F8 / ctrl+t   verify with TLA+          ctrl+l  go live
    ctrl+b        bind agents               ctrl+o  open another policy
    ctrl+n        new policy                ctrl+q  quit

The left side is a full editor. The right side shows the verification as it happens, the
suggestions (Enter applies one, then Z3 runs again), and for the bound agents: whether their
real tools fit the policy, whether their mappings stay fail-closed, and what the edit would
have changed on the decisions they really made.
"""

from __future__ import annotations

import time
from typing import List, Optional

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, OptionList, SelectionList, Static, TabbedContent, TabPane
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from .. import VENOM_VERSION
from . import anim as A
from .editor import CslEditor
from .session import StudioSession
from ..render.words import n as _n

CSS = """
Screen { background: #0b1220; }
#top { height: 1; background: #0f172a; color: #94a3b8; padding: 0 1; }
#main { height: 1fr; }
#editor { width: 3fr; border: round #134e4a; }
#editor:focus { border: round #2dd4bf; }
#side { width: 2fr; min-width: 44; }
#verify { height: 3fr; border: round #1e293b; padding: 0 1; }
#tabs { height: 2fr; border: round #1e293b; }
#suggestions { height: 1fr; }
#agentsbox { padding: 0 1; }
#status { height: 1; background: #0f172a; padding: 0 1; }
ListView > ListItem { padding: 0 1; }
ListView > ListItem.--highlight { background: #134e4a; }
ModalScreen { align: center middle; background: rgba(2, 6, 23, 0.7); }
#dialog { width: 92; max-width: 95%; height: auto; max-height: 90%; border: round #2dd4bf; background: #0f172a; padding: 1 2; }
#dialog Label { margin-bottom: 1; }
#buttons { height: 3; align-horizontal: right; margin-top: 1; }
#buttons Button { margin-left: 2; }
#search { margin-bottom: 1; }
"""


class VerifyView(Static):
    """Plays the Z3 / TLA+ animation for the latest run, then keeps its final frame."""

    def on_mount(self) -> None:
        self.kind: Optional[str] = None
        self.run = None
        self.started = 0.0
        self.result_at = 0.0
        self.note = ""
        self.timer = self.set_interval(1 / 24, self.tick, pause=True)
        self.show_idle()

    def show_idle(self) -> None:
        self.update(Text.assemble(("VERIFY\n\n", "bold #94a3b8"), ("F5", "bold #5eead4"), ("  Z3: every rule can trigger, no two rules conflict\n", "#cbd5e1"),
                                  ("F8", "bold #5eead4"), ("  TLA+: which states each rule blocks, as the guard runs it\n\n", "#cbd5e1"),
                                  ("Nothing runs on its own: you decide when to prove.", "#64748b")))

    def begin(self, kind: str, note: str = "") -> None:
        self.kind, self.run, self.note = kind, None, note
        self.started = time.monotonic()
        self.timer.resume()

    def deliver(self, run) -> None:
        self.run = run
        self.result_at = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        if self.run is None:
            if self.kind == "tla":
                self.update(A.tla_pending(now - self.started, self.note))
            else:
                self.update(Text.assemble(("Z3  ", "bold #94a3b8"), (A.SPIN[int(now * 14) % 10] + " encoding the policy", "#5eead4")))
            return
        t = now - self.result_at
        if self.kind == "z3":
            self.update(A.z3_frame(self.run, t))
            done = t > A.z3_duration(self.run)
        else:
            self.update(A.tla_frame(self.run, t))
            done = t > A.tla_duration(self.run)
        if done:
            self.timer.pause()

    def skip(self) -> None:
        if self.run is not None:
            self.result_at -= 100
            self.tick()


class PolicyPicker(ModalScreen[Optional[str]]):
    BINDINGS = [Binding("escape", "dismiss(None)", "close")]

    def __init__(self, items) -> None:
        super().__init__()
        self.items = items

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(Text("Open a policy", style="bold #5eead4"))
            options = [Option(Text.assemble(("＋ new policy", "bold #5eead4")), id="__new__")]
            for it in self.items:
                style = {"live": "#4ade80", "draft": "#fbbf24", "adopted": "#7dd3fc"}.get(it["status"], "#cbd5e1")
                options.append(Option(Text.assemble((f"{it['name']:<28}", "bold #e2e8f0"), (f"{it['status']:<9}", style),
                                                    (it["agents"], "#64748b")), id=it["path"]))
            yield OptionList(*options, id="picker")

    @on(OptionList.OptionSelected)
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class AgentBinder(ModalScreen[Optional[List[str]]]):
    BINDINGS = [Binding("escape", "dismiss(None)", "cancel")]

    def __init__(self, agents: List[str], selected: List[str], kinds: dict) -> None:
        super().__init__()
        self.agents, self.selected, self.kinds = agents, set(selected), kinds

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(Text.assemble(("Bind agents to this policy", "bold #5eead4"),
                                      ("   space toggles · type to filter · patterns like pay-* work", "#64748b")))
            yield Input(placeholder="filter agents", id="search")
            yield SelectionList(*self._items(""), id="agents")
            with Horizontal(id="buttons"):
                yield Button("Bind selected", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def _items(self, query: str):
        import fnmatch
        q = query.strip()
        out = []
        for a in self.agents:
            if q and not (fnmatch.fnmatch(a, q) if any(c in q for c in "*?[") else q.lower() in a.lower()):
                continue
            out.append(Selection(Text.assemble((a, "bold #e2e8f0"), (f"  {self.kinds.get(a, '')}", "#64748b")), a, a in self.selected))
        return out

    @on(Input.Changed, "#search")
    def filter(self, event: Input.Changed) -> None:
        sl = self.query_one("#agents", SelectionList)
        sl.clear_options()
        sl.add_options(self._items(event.value))

    @on(Input.Submitted, "#search")
    def select_matches(self, event: Input.Submitted) -> None:
        """Enter in the filter selects every agent it shows (bulk binding by pattern)."""
        sl = self.query_one("#agents", SelectionList)
        sl.select_all()
        self.selected |= set(sl.selected)

    @on(SelectionList.SelectedChanged)
    def changed(self, event) -> None:
        sl = self.query_one("#agents", SelectionList)
        shown = {s.value for s in self._items(self.query_one("#search", Input).value)}
        self.selected = (self.selected - shown) | set(sl.selected)

    @on(Button.Pressed, "#ok")
    def ok(self) -> None:
        self.dismiss(sorted(self.selected))

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class GoLiveDialog(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "cancel"), Binding("y", "go", "go live")]

    def __init__(self, summary: Text) -> None:
        super().__init__()
        self.summary = summary

    def on_mount(self) -> None:
        self.query_one("#go", Button).focus()  # Enter confirms, Esc cancels

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(Text("Go live", style="bold #5eead4"))
            yield VerticalScroll(Static(self.summary))
            with Horizontal(id="buttons"):
                yield Button("Go live (y / Enter)", variant="success", id="go")
                yield Button("Cancel", id="cancel")

    def action_go(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#go")
    def go(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(False)


class StudioApp(App):
    CSS = CSS
    TITLE = "CSL-Core Studio"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("f5,ctrl+r", "z3", "Z3", priority=True),
        Binding("f8,ctrl+t", "tla", "TLA+", priority=True),
        Binding("ctrl+s", "save", "save", priority=True),
        Binding("ctrl+l", "live", "go live", priority=True),
        Binding("ctrl+b", "bind", "bind agents", priority=True),
        Binding("ctrl+o", "open", "open", priority=True),
        Binding("ctrl+n", "new", "new", priority=True),
        Binding("escape", "skip", "skip animation", show=False),
        Binding("ctrl+q", "quit_studio", "quit", priority=True),
    ]

    def __init__(self, session: StudioSession, use_real_tlc: bool = True) -> None:
        super().__init__()
        self.session = session
        self.use_real_tlc = use_real_tlc
        self.result_message = ""

    # -- layout -------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Horizontal(id="main"):
            yield CslEditor(self.session.text, id="editor")
            with Vertical(id="side"):
                with VerticalScroll(id="verify"):
                    yield VerifyView(id="verifyview")
                with TabbedContent(id="tabs"):
                    with TabPane("Suggestions", id="tab-suggest"):
                        yield ListView(id="suggestions")
                    with TabPane("Agents", id="tab-agents"):
                        yield VerticalScroll(Static(id="agentsbox"))
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        from ..render.theme import THEME
        self.console.push_theme(THEME)  # the animations use the same style names as the rest of the CLI
        self.refresh_top()
        self.refresh_side()
        self.status("F5 Z3 · F8 TLA+ · ctrl+l go live" if self.session.text else "ctrl+o open a policy, or ctrl+n start one")
        self.query_one("#editor", CslEditor).focus()

    @property
    def editor(self) -> CslEditor:
        return self.query_one("#editor", CslEditor)

    def status(self, message: str, style: str = "#94a3b8") -> None:
        self.query_one("#status", Static).update(Text(message, style=style))

    def refresh_top(self) -> None:
        s = self.session
        s.text = self.editor.text if self.is_mounted and self.query("#editor") else s.text
        state = s.state + (" · unsaved" if s.modified else "")
        style = {"live": "#4ade80", "draft": "#fbbf24"}.get(s.state, "#fbbf24")
        agents = ", ".join(s.agents[:3]) + (f" +{len(s.agents) - 3}" if len(s.agents) > 3 else "") if s.agents else "no agent bound"
        self.query_one("#top", Static).update(Text.assemble(
            (f" CSL-Core Studio {VENOM_VERSION} ", "bold #5eead4"), ("· ", "#475569"), (s.name, "bold #e2e8f0"),
            ("  ", ""), (state, style), ("   bound: ", "#64748b"), (agents, "#cbd5e1")))

    def refresh_side(self) -> None:
        text = self.editor.text
        lv = self.query_one("#suggestions", ListView)
        lv.clear()
        self.suggestions = self.session.suggestions(text)
        if not self.suggestions:
            lv.append(ListItem(Label(Text("No suggestions. Run Z3 (F5) or TLA+ (F8) to get some.", style="#64748b"))))
        for s in self.suggestions:
            badge = {"z3": ("Z3 ", "#5eead4"), "tla": ("TLA", "#c4b5fd"), "agent": ("AGT", "#7dd3fc")}[s.source]
            conf = {"HIGH": "#4ade80", "MEDIUM": "#fbbf24", "LOW": "#94a3b8"}.get(s.confidence, "#94a3b8")
            lv.append(ListItem(Label(Text.assemble((f" {badge[0]} ", f"bold #0f172a on {badge[1]}"), ("  ", ""),
                                                   (s.title, "bold #e2e8f0"), ("  " + s.confidence.lower(), conf),
                                                   ("\n" + s.explanation, "#94a3b8"),
                                                   ("\n↵ apply" if s.applicable else "", "#2dd4bf")))))
        self.query_one("#agentsbox", Static).update(self.agents_panel(text))

    def agents_panel(self, text: str) -> Text:
        s = self.session
        out = Text()
        if not s.agents:
            out.append("No agent bound to this policy.\n\n", style="#cbd5e1")
            out.append("ctrl+b", style="bold #5eead4")
            out.append(" binds agents: their real tool names are checked against the policy, their mappings are "
                       "tested for fail-open cases, and their recorded decisions are replayed against your edit.", style="#94a3b8")
            return out
        for f in s.fit(text):
            out.append(f"{f.agent}\n", style="bold #e2e8f0")
            out.append(f"  tools covered   {f.covered}/{f.tools}\n", style="#cbd5e1")
            if f.renames:
                out.append(f"  named wrong     {', '.join(f'{a}→{b}' for a, b in list(f.renames.items())[:3])}\n", style="#fbbf24")
            if f.uncovered_risky:
                out.append(f"  no rule         {', '.join(f'{t} ({c})' for t, c in f.uncovered_risky[:4])}\n", style="#fb923c")
            ok = f.fail_open == 0
            out.append(f"  mapping test    {_n(f.cases, 'case')} · {f.fail_open} fail-open" + (f" · {f.mapping_note}" if f.mapping_note else "") + "\n",
                       style="#4ade80" if ok else "#f87171")
        r = s.replay(text)
        out.append("\nREPLAY ", style="bold #94a3b8")
        if r.error:
            out.append(f"not replayed: the text does not compile ({r.error}); press F5 for details\n", style="#fbbf24")
        elif r.total == 0:
            out.append("no recorded decisions yet for these agents\n", style="#64748b")
        else:
            out.append(f"{r.replayed:,} of {r.total:,} recorded calls replayed against this text\n", style="#cbd5e1")
            out.append(f"  {r.newly_blocked} would now be blocked", style="#f87171" if r.newly_blocked else "#64748b")
            out.append(" · ")
            out.append(f"{r.newly_allowed} would now be allowed\n", style="#4ade80" if r.newly_allowed else "#64748b")
            for agent, tool, change, rules in r.examples[:4]:
                out.append(f"  {agent} {tool}: {change}" + (f" ({', '.join(rules[:2])})" if rules else "") + "\n", style="#94a3b8")
            if r.skipped:
                out.append(f"  {_n(r.skipped, 'call')} could not be replayed (unmapped input, exemptions, kill switch)\n", style="#64748b")
        return out

    # -- events -------------------------------------------------------------------------
    @on(CslEditor.Changed)
    def edited(self) -> None:
        self.session.text = self.editor.text
        self.refresh_top()

    @on(ListView.Selected, "#suggestions")
    def apply_suggestion(self, event: ListView.Selected) -> None:
        idx = self.query_one("#suggestions", ListView).index
        if idx is None or idx >= len(getattr(self, "suggestions", [])):
            return
        s = self.suggestions[idx]
        if not s.applicable:
            self.status("this suggestion explains what to do; it has no automatic fix", "#fbbf24")
            return
        new = s.patch(self.editor.text)
        if not new or new == self.editor.text:
            self.status("the suggestion no longer applies to the current text", "#fbbf24")
            return
        self.editor.replace(new, (0, 0), self.editor.document.end)  # undo (ctrl+z) restores the previous text
        self.status(f"applied: {s.title} · verifying with Z3", "#4ade80")
        self.action_z3()

    # -- actions ------------------------------------------------------------------------
    def action_skip(self) -> None:
        self.query_one("#verifyview", VerifyView).skip()

    def action_save(self) -> None:
        p = self.session.save(self.editor.text)
        self.refresh_top()
        self.refresh_side()
        self.status(f"saved {self.session.ws.rel(p)} (a draft: nothing is live until ctrl+l)", "#4ade80")

    def action_z3(self) -> None:
        v = self.query_one("#verifyview", VerifyView)
        v.begin("z3")
        self.status("Z3: checking that every rule can trigger and no two rules conflict ...")
        self._run_z3(self.editor.text)

    @work(thread=True, exclusive=True, group="verify")
    def _run_z3(self, text: str) -> None:
        run = self.session.verify_z3(text)
        self.call_from_thread(self._z3_done, run)

    def _z3_done(self, run) -> None:
        self.query_one("#verifyview", VerifyView).deliver(run)
        lines = [i.line for i in run.issues if i.line]
        from .suggest import rule_line
        for i in run.issues + run.warnings:
            for r in i.rules:
                ln = rule_line(self.editor.text, r)
                if ln:
                    lines.append(ln)
        self.editor.mark_problems(lines)
        self.refresh_side()
        if run.ok and run.unreachable:
            self.status(f"Z3: consistent, but {', '.join(sorted(set(run.unreachable)))} can never trigger", "#fbbf24")
        elif run.ok:
            self.status(f"Z3: proven consistent · {_n(len(run.rules), 'rule')} · {_n(len(run.pairs), 'pair')} · {run.elapsed_ms} ms", "#4ade80")
        else:
            self.status(f"Z3: {len(run.issues)} problem(s); see the suggestions", "#f87171")

    def action_tla(self) -> None:
        v = self.query_one("#verifyview", VerifyView)
        v.begin("tla", "real TLC when available")
        self.status("TLA+: exploring every reachable state ...")
        self._run_tla(self.editor.text)

    @work(thread=True, exclusive=True, group="verify")
    def _run_tla(self, text: str) -> None:
        run = self.session.verify_tla(text, use_real_tlc=self.use_real_tlc)
        self.call_from_thread(self._tla_done, run)

    def _tla_done(self, run) -> None:
        self.query_one("#verifyview", VerifyView).deliver(run)
        self.editor.mark_problems([])  # a rule the guard enforces is not a problem
        self.refresh_side()
        if run.error:
            self.status(f"TLA+: not verified · {run.error}", "#f87171")
        elif run.refuses_to_load:
            self.status("TLA+: ENABLE_FORMAL_VERIFICATION: TRUE makes the compiler refuse rules that block; "
                        "the first suggestion sets it to FALSE (the studio keeps checking TLA+ for you)", "#fbbf24")
        else:
            msg = f"TLA+: guard verified · {_n(len(run.enforced), 'rule')} enforce"
            if run.checked:
                msg += f" · {run.blocked:,} of {run.checked:,} states blocked"
            if run.never_fires:
                msg += f" · never fires: {', '.join(run.never_fires[:3])}"
            self.status(msg + f" · {run.total_states:,} states · {run.engine}", "#fbbf24" if run.never_fires else "#4ade80")

    def action_bind(self) -> None:
        kinds = {}
        if self.session.inv is not None:
            from ..policy.draft import agent_key
            kinds = {agent_key(a): f"{a.kind} · {_n(len(a.tools), 'tool')}" for a in self.session.inv.agents}

        def done(result: Optional[List[str]]) -> None:
            if result is None:
                return
            self.session.agents = result
            self.refresh_top()
            self.refresh_side()
            self.query_one("#tabs", TabbedContent).active = "tab-agents"
            self.status(f"{_n(len(result), 'agent')} selected; they are bound when the policy goes live (ctrl+l)", "#5eead4")

        if not self.session.all_agents():
            self.status("no discovered agents in this workspace yet: run cslcore venom first", "#fbbf24")
            return
        self.push_screen(AgentBinder(self.session.all_agents(), self.session.agents, kinds), done)

    def action_open(self) -> None:
        def done(path: Optional[str]) -> None:
            if path is None:
                return
            if path == "__new__":
                self.session.new()
            else:
                self.session.open(path)
            self.load_session()

        if self.session.modified:
            self.status("unsaved changes were kept as a draft", "#fbbf24")
            self.session.save(self.editor.text)
        self.push_screen(PolicyPicker(self.session.policies()), done)

    def action_new(self) -> None:
        if self.session.modified:
            self.session.save(self.editor.text)
        agent = self.session.agents[0] if self.session.agents and not self.session.draft else None
        self.session.new(agent)
        self.load_session()

    def load_session(self) -> None:
        self.editor.load_text(self.session.text)
        self.editor.mark_problems([])
        self.query_one("#verifyview", VerifyView).show_idle()
        self.refresh_top()
        self.refresh_side()
        self.status(f"{self.session.name}: {self.session.state}")

    def action_live(self) -> None:
        text = self.editor.text
        z = self.session.z3_current(text)
        if z is None:
            self.status("Z3 first: running it now; press ctrl+l again when it is green", "#fbbf24")
            self.action_z3()
            return
        if not z.ok:
            self.status("not live: Z3 found problems (see the suggestions)", "#f87171")
            return
        self.push_screen(GoLiveDialog(self.live_summary(text)), self._live_confirmed)

    def live_summary(self, text: str) -> Text:
        import difflib
        s = self.session
        out = Text()
        out.append("Z3      ", style="bold #94a3b8")
        out.append("proven consistent\n", style="#4ade80")
        t = s.tla_current(text)
        out.append("TLA+    ", style="bold #94a3b8")
        if t is None:
            out.append("not run on this text (F8); optional\n", style="#fbbf24")
        else:
            if t.error:
                out.append(f"not verified ({t.error})\n", style="#f87171")
            elif t.refuses_to_load:
                out.append("the compiler refuses this policy (ENABLE_FORMAL_VERIFICATION: TRUE); see suggestions\n", style="#f87171")
            else:
                out.append(f"guard verified · {_n(len(t.enforced), 'rule')} enforce" + (f" · {len(t.never_fires)} never fire" if t.never_fires else "")
                           + (f" · blocks {t.blocked:,} of {t.checked:,} states" if t.checked else "") + "\n",
                           style="#fbbf24" if t.never_fires else "#4ade80")
        out.append("file    ", style="bold #94a3b8")
        out.append(f"{s.ws.rel(s.active)}" + (" (replaces the live version, kept in history)" if s.active and s.active.exists() else " (new)") + "\n",
                   style="#cbd5e1")
        if s.external:
            out.append("        ", style="")
            out.append(f"started from your file {s.external}; that file is not changed\n", style="#fbbf24")
        out.append("agents  ", style="bold #94a3b8")
        out.append((", ".join(s.agents) if s.agents else "none selected (ctrl+b)") + "\n", style="#cbd5e1")
        for f in s.fit(text):
            out.append(f"        {f.agent}: mapping {_n(f.cases, 'case')}, {f.fail_open} fail-open\n", style="#4ade80" if not f.fail_open else "#f87171")
        old = s.ws.read(s.active) if s.active and s.active.exists() else ""
        diff = list(difflib.unified_diff((old or "").splitlines(), text.splitlines(), "live", "new", lineterm="", n=1))
        if diff:
            out.append("\n")
            for l in diff[:40]:
                st = "#4ade80" if l.startswith("+") and not l.startswith("+++") else ("#f87171" if l.startswith("-") and not l.startswith("---") else "#64748b")
                out.append(l + "\n", style=st)
        return out

    def _live_confirmed(self, go: bool) -> None:
        if not go:
            self.status("nothing changed")
            return
        res = self.session.go_live(self.editor.text)
        if res.ok and self.editor.text != self.session.text:
            self.editor.load_text(self.session.text)
        self.refresh_top()
        self.refresh_side()
        msg = res.message + (f" · not bound: {'; '.join(res.refused)}" if res.refused else "")
        self.status(msg, "#4ade80" if res.ok and not res.refused else ("#fbbf24" if res.ok else "#f87171"))
        self.result_message = msg

    def action_quit_studio(self) -> None:
        if self.session.modified:
            self.session.save(self.editor.text)
            self.result_message = f"unsaved edits kept as a draft: {self.session.ws.rel(self.session.draft)}"
        self.exit(self.result_message)
