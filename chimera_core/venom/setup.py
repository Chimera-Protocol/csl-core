"""
`cslcore setup`: the guided, resumable flow.

    1 Scope       2 Discover    3 Inventory   4 Findings    5 Exemptions
    6 Policies    7 Verify      8 Map         9 Mode & wiring   10 Activate

One step per screen: a short summary, one recommended action, Enter to continue (q stops;
progress is kept). Progress lives in .csl/venom/state.json. Re-running resumes at the first
incomplete step; after a complete run it starts a new cycle and reports what changed.

At a terminal, step 6 is the protection board (chimera_core.venom.board): the riskiest agents
first; for one agent its limits, the policy made from them, the wiring change and the check in one
loop; standard protection for the rest with one key. Steps 7 to 10 then only handle what the
board did not (drafts from the studio, an editor or an assistant). `--strategy` keeps the step by
step flow. After a complete run, `cslcore setup` opens on a home screen with the board on `b`.

Scripted runs (`--yes`) accept defaults but never approve exemptions and never activate a
policy; activation in a scripted run needs the explicit `--activate` flag.

0.5.1 installations: policies already wired into agents are adopted in place (never
copied or modified), their own mappers can be tested, and the wiring step shows the one-line
`observe()` change that adds decision logs, log mode and kill switches.
"""

from __future__ import annotations

import fnmatch
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Group
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import VENOM_VERSION
from . import exemptions as ex
from .commands import EXIT_OK, console_for, remember_root, run_scan, save_report, workspace_for
from .policy import limits as L
from .model import Agent, Inventory, PolicyRef
from .policy import draft as D
from .policy.gate import verify_text
from .render.screen import scan_screen
from .render.words import n as _n

STEPS = ["scope", "discover", "inventory", "findings", "exemptions", "policies", "verify", "map", "wire", "activate"]
TITLES = {
    "scope": "Scope", "discover": "Discover", "inventory": "Inventory", "findings": "Findings", "exemptions": "Exemptions",
    "policies": "Policies", "verify": "Verify", "map": "Map", "wire": "Mode & wiring", "activate": "Activate",
}
EXIT_INCOMPLETE = 5
PAUSE_AFTER = {"inventory", "findings", "policies", "map", "wire"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StopFlow(Exception):
    """The operator chose to stop; progress is saved."""


def short(path: str) -> str:
    """A path as people want to read it: relative to the current folder, else with ~ for home."""
    import os
    p = str(path)
    cwd = os.getcwd().rstrip("/") + "/"
    if p.startswith(cwd):
        return p[len(cwd):]
    home = str(Path.home()).rstrip("/") + "/"
    return "~/" + p[len(home):] if p.startswith(home) else p


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def grid(*rows: Tuple[str, Any], label_width: int = 10) -> Table:
    """Label / value rows that wrap cleanly on narrow terminals."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="label", no_wrap=True, width=label_width)
    t.add_column(overflow="fold")
    for label, value in rows:
        t.add_row(label, value if not isinstance(value, str) else Text(value, style="text"))
    return t


class Flow:
    def __init__(self, args) -> None:
        self.args = args
        self.console = console_for(args)
        self.ws = workspace_for(args)
        self.yes = bool(getattr(args, "yes", False))
        self.activate_flag = bool(getattr(args, "activate", False))
        self.state: Dict[str, Any] = self.ws.load_state()
        self.setup: Dict[str, Any] = self.state.setdefault("setup", {})
        self.inv: Optional[Inventory] = None
        self.previous: Optional[Dict[str, Any]] = None
        self.interactive = sys.stdin.isatty() and not self.yes

    # -- bookkeeping ----------------------------------------------------------------
    def save(self) -> None:
        """Write only the flow's own key; modes, controls and test results belong to others."""
        fresh = self.ws.load_state()
        fresh["setup"] = self.setup
        self.ws.save_state(fresh)
        self.state = fresh

    def set_state(self, key: str, value: Any) -> None:
        fresh = self.ws.load_state()
        fresh[key] = value
        fresh["setup"] = self.setup
        self.ws.save_state(fresh)
        self.state = fresh

    def done(self, step: str) -> None:
        completed = self.setup.setdefault("completed", [])
        if step not in completed:
            completed.append(step)
        self.setup["updated"] = _now()
        self.save()

    def agents_state(self) -> Dict[str, Dict[str, Any]]:
        return self.setup.setdefault("agents", {})

    # -- interaction ----------------------------------------------------------------------
    def header(self, step: str) -> None:
        i = STEPS.index(step) + 1
        bar = Text()
        for j, s in enumerate(STEPS, 1):
            style = "ok" if s in self.setup.get("completed", []) else ("brand" if j == i else "muted")
            bar.append("●" if j <= i else "○", style=style)
        self.console.print()
        self.console.print(Text.assemble(("  ", ""), (f"{i:>2}/10 ", "muted"), (TITLES[step].upper().ljust(14), "brand"), bar))
        self.console.print()

    def pause(self, reach_map: bool = False) -> None:
        if not self.interactive:
            return
        from rich.prompt import Prompt
        while True:
            hint = Text.assemble(("  ", ""), ("Enter", "brand"), (" continue · ", "muted"))
            if reach_map:
                hint.append_text(Text.assemble(("m", "brand"), (" reach map · ", "muted")))
            hint.append_text(Text.assemble(("q", "brand"), (" stop here (cslcore setup resumes)", "muted")))
            ans = Prompt.ask(hint, default="", show_default=False, console=self.console).strip().lower()
            if ans == "q":
                raise StopFlow()
            if ans == "m" and reach_map:
                self.open_room("map")
                continue
            return

    def open_room(self, name: str) -> None:
        """The reach map or the live panel, from inside setup; quitting it comes back here."""
        from .rooms import interactive, run
        if interactive(self.console):
            run(self.console, self.args, name, inv=self.load_inventory() if name == "map" else None, came_from="setup")

    def ask(self, question: str, default: bool) -> bool:
        from .policy.workbench import confirm
        if self.yes:
            return default
        return confirm(self.console, escape(question), False, default=default)

    def choose(self, question: str, choices: List[str], default: str) -> str:
        if not self.interactive:
            return default
        from rich.prompt import Prompt
        return Prompt.ask(f"  {escape(question)}", choices=choices, default=default, console=self.console)

    def text_input(self, question: str, default: str = "") -> str:
        if not self.interactive:
            return default
        from rich.prompt import Prompt
        return Prompt.ask(f"  {escape(question)}", default=default, show_default=bool(default), console=self.console)

    # -- inventory ------------------------------------------------------------------------
    def load_inventory(self) -> Inventory:
        if self.inv is None:
            data = self.ws.latest_inventory()
            if data is None:
                self.discover()
            else:
                self.inv = Inventory.from_dict(data)
        return self.inv  # type: ignore[return-value]

    def targets(self) -> List[Agent]:
        inv = self.load_inventory()
        agents = [a for a in inv.agents if D.needs_policy(a)]
        only = getattr(self.args, "agent", None)
        if only:
            agents = [a for a in agents if only in (a.id, a.display_name, D.agent_key(a))]
        return agents

    def existing_install(self) -> List[Tuple[Agent, PolicyRef]]:
        """Agents whose code already enforces a policy (a 0.5.1 integration)."""
        from .analysis.coverage import link_policies
        inv = self.load_inventory()
        out = []
        for a in inv.agents:
            if a.guard.status == "none" or a.kind == "assistant":
                continue
            linked = [p for p in link_policies(a, inv.policies) if p.status != "draft"]
            if linked:
                out.append((a, linked[0]))
        return out

    def policy_text(self, path: str) -> Optional[str]:
        """Workspace files directly, host files through the probe (fixture-aware)."""
        p = Path(path)
        text = self.ws.read(p if p.is_absolute() else self.ws.root / p)
        if text is None:
            from .probe import probe_for
            probe, _ = probe_for(getattr(self.args, "root", None))
            text = probe.read_text(path)
        return text

    # -- steps ----------------------------------------------------------------------------
    def scope(self) -> bool:
        root = getattr(self.args, "root", None)
        where = "this machine" if not root else f"the folder {root}"
        body = grid(
            ("reads", Text.assemble(("code (parsed, never run), assistant and MCP configs, cron / systemd / launchd, "
                                     "the process list, run history and existing .csl policies", "text"))),
            ("never", "changes what it reads, runs discovered code, or reads credential values, prompts or transcripts"),
            ("writes", f"only {self.ws.root}/.csl/venom and {self.ws.root}/policies, and only after you confirm"),
        )
        self.console.print(Panel(Group(Text.assemble(("Venom looks at ", "text"), (where, "head"), (".", "text")), Text(""), body),
                                 title=Text(f" CSL-Core setup {VENOM_VERSION} ", style="brand"), title_align="left",
                                 box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
        if self.ws.plan_only:
            self.console.print("  [warn]--plan-only[/warn]: nothing will be written")
        return self.ask("Start?", True)

    def discover(self) -> bool:
        self.previous = self.ws.latest_inventory()
        res = run_scan(self.args, self.console)
        self.inv = res.inventory
        if not self.ws.plan_only:
            save_report(self.ws, self.inv)
            remember_root(self.ws, self.args)
        return True

    def inventory(self) -> bool:
        inv = self.load_inventory()
        self.console.print(scan_screen(inv, VENOM_VERSION, self.console.width, report_hint=".csl/venom/reports/latest.md",
                                       show_findings=False))
        existing = self.existing_install()
        if existing:
            names = ", ".join(sorted({a.display_name for a, _ in existing}))
            self.console.print()
            self.console.print(Panel(Group(
                Text.assemble(("Existing CSL-Core setup found: ", "head"),
                              (f"{plural(len({p.path for _, p in existing}), 'policy').replace('policys', 'policies')} wired into "
                               f"{plural(len(existing), 'agent')} ({names}).", "text")),
                Text("Nothing in your code changes. In step 6 you keep these policies as they are (adopt them); "
                     "in step 8 you can test your own mappers for fail-open cases.", style="muted")),
                box=box.ROUNDED, border_style="ok", padding=(0, 1)))
        self._changes()
        return True

    def _changes(self) -> None:
        prev = self.previous
        if not prev or not self.inv:
            return
        old = {a["id"]: {t["name"] for t in a.get("tools", [])} for a in prev.get("agents", [])}
        lines: List[Text] = []
        for a in self.inv.agents:
            if a.id not in old:
                lines.append(Text.assemble(("  + new agent ", "ok"), (a.display_name, "head")))
                continue
            new_tools = sorted({t.name for t in a.tools} - old[a.id])
            if new_tools:
                lines.append(Text.assemble(("  + new tools ", "ok"), (", ".join(new_tools), "head"), (f" on {a.display_name}", "muted")))
                if a.guard.status != "none":
                    lines.append(Text(f"    offer: cslcore policy extend {a.display_name} --agent {a.display_name}", style="brand"))
                    self.setup.setdefault("offers", []).append({"extend": a.display_name, "tools": new_tools})
        for gone in sorted(set(old) - {a.id for a in self.inv.agents}):
            lines.append(Text.assemble(("  - gone ", "muted"), (gone, "text")))
        prev_drift = {(d["policy"], d["variable"], d.get("value")) for d in prev.get("drift", [])}
        for d in self.inv.drift:
            if (d.policy, d.variable, d.value) not in prev_drift and d.kind != "unsupplied_variable":
                lines.append(Text.assemble(("  ! new drift ", "warn"), (f"{d.policy}.{d.variable} {d.value or ''}", "text")))
        self.console.print()
        self.console.print(Text("  CHANGES SINCE LAST SCAN", style="label"))
        for l in lines or [Text("  none", style="muted")]:
            self.console.print(l)

    def findings(self) -> bool:
        from .analysis.findings import EXPLAIN
        from .render.theme import SEV_GLYPH

        inv = self.load_inventory()
        if not inv.findings:
            self.console.print("  [ok]No findings.[/ok]")
            return True
        counts = {s: sum(1 for f in inv.findings if f.severity == s) for s in ("high", "medium", "low", "info")}
        self.console.print(Text.assemble(("  ", ""), *[(f"{SEV_GLYPH[s]} {n} {s}   ", s) for s, n in counts.items() if n]))
        self.console.print()
        shown = [f for f in inv.findings if f.severity == "high"][:5] or inv.findings[:3]
        for f in shown:
            why, fix = EXPLAIN.get(f.id, ("", f.recommendation or ""))
            t = Table.grid(padding=(0, 1))
            t.add_column(width=1, no_wrap=True)
            t.add_column(width=4, no_wrap=True)
            t.add_column(overflow="fold")
            t.add_row(Text(SEV_GLYPH[f.severity], style=f.severity), Text(f.id, style="label"), Text(f.summary, style="head"))
            if why:
                t.add_row("", Text("why", style="label"), Text(why, style="text"))
            t.add_row("", Text("do", style="label"), Text(fix or f.recommendation or "", style="text"))
            self.console.print(Padding(t, (0, 0, 1, 2)))
        rest = len(inv.findings) - len(shown)
        if rest:
            by_id: Dict[str, int] = {}
            for f in inv.findings:
                if f not in shown:
                    by_id[f.id] = by_id.get(f.id, 0) + 1
            summary = ", ".join(f"{fid} ×{n}" for fid, n in sorted(by_id.items()))
            self.console.print(Text(f"  {rest} more ({summary}): .csl/venom/reports/latest.md", style="muted"))
        return True

    def exemptions(self) -> bool:
        items = self.ws.load_exemptions()
        proposed = [e for e in items if e.status == "proposed"]
        approved = [e for e in items if e.status == "approved"]
        if approved:
            self.console.print(f"  {len(approved)} approved exemptions in effect")
        if not proposed:
            self.console.print("  [muted]Nothing to approve. Fully trusted agents or tools can be exempted any time with "
                               "cslcore exempt add, or from cslcore watch.[/muted]")
            return True
        for e in proposed:
            self.console.print(Text.assemble(("  proposed  ", "warn"), (e.agent + (f" / {e.tool}" if e.tool else ""), "head"), (f"  {e.reason or ''}", "muted")))
            if not self.interactive:
                self.console.print("  [muted]left as proposed (approval is interactive only: cslcore exempt approve <n> --approved-by NAME)[/muted]")
                continue
            if self.ask("Approve?", False):
                e.approved_by = self.text_input("approved by")
                e.status = "approved"
                try:
                    ex.validate(e)
                except ex.ExemptionError as err:
                    self.console.print(f"  [high]{err}[/high]")
                    e.status = "proposed"
        if not self.ws.plan_only:
            self.ws.save_exemptions(items)
        return True

    # -- policies -------------------------------------------------------------------------
    def _strategy(self, n: int) -> str:
        given = getattr(self.args, "strategy", None)
        if given:
            return given
        if not self.interactive:
            return "templates"
        if n <= 2:
            return "choose"
        self.console.print(Padding(grid(
            ("r", "recommended: keep the policies your agents already use, templates for the rest"),
            ("c", "choose per agent (existing policy, template, write it yourself, or with your assistant)"),
            ("t", "templates for every agent"), label_width=3), (0, 0, 0, 4)))
        return {"r": "recommended", "c": "choose", "t": "templates"}[self.choose(f"{_n(n, 'agent')} need{'s' if n == 1 else ''} a policy. How?", ["r", "c", "t"], "r")]

    def policies(self) -> bool:
        agents = self.targets()
        states = self.agents_state()
        if not agents:
            self.console.print("  [ok]every agent is covered or exempt[/ok]")
            return True
        if self.uses_board():
            return self.board(agents)
        strategy = self._strategy(len(agents))
        for a in agents:
            self._policy_for(a, strategy)
        self.save()
        waiting = [s for s in states.values() if s.get("assistant") and not s.get("draft")]
        if waiting:
            self.console.print(f"  [warn]{len(waiting)} agent(s) waiting for an assistant draft[/warn]: "
                               "the flow continues with the others; re-run cslcore setup once the draft is saved")
        return True

    def uses_board(self) -> bool:
        """At a terminal, without --strategy or --agent: the protection board."""
        return self.interactive and not getattr(self.args, "strategy", None) and not getattr(self.args, "agent", None)

    def board(self, agents: List[Agent]) -> bool:
        """The protection board: the riskiest agents first, one loop per agent (limits, policy,
        wiring, check), standard protection for the rest with one key. Opens where it was left."""
        from . import board as B
        from .bindings import Bindings

        def other_ways(a: Agent) -> None:
            self._policy_for(a, "choose")
            self.save()

        B.run(self, self.console, self.args, self.ws, agents, self.agents_state(), other_ways=other_ways)
        bindings = Bindings(self.ws).all()
        for a in agents:  # what the board made, for the steps after it
            st = self.agents_state().setdefault(a.id, {"key": D.agent_key(a)})
            b = bindings.get(st["key"])
            if b is not None and b.mapping and not st.get("draft") and not st.get("adopted"):
                st.update(policy=b.policy, mapping=b.mapping, protected=True)
            elif st.get("board_skip"):
                st["skipped"] = True  # the later steps leave it alone too
        self.save()
        return True

    def _policy_for(self, a: Agent, strategy: str) -> None:
        """One agent's policy, the way the strategy says: an existing one, templates from its limits,
        the studio, an editor, an assistant, or skipped."""
        from .policy.match import candidates

        inv = self.load_inventory()
        states = self.agents_state()
        key = D.agent_key(a)
        st = states.setdefault(a.id, {"key": key})
        draft_path = self.ws.drafts / f"{key}.csl"
        active_path = self.ws.policies / f"{key}.csl"
        if st.get("adopted") and st.get("policy"):
            self._line("✓", a, f"keeps {short(st['policy'])} (adopted)")
            return
        if active_path.exists() and not self.ws.read(draft_path):
            st["policy"] = self.ws.rel(active_path)
            self._line("✓", a, f"active: {self.ws.rel(active_path)}")
            return
        if draft_path.exists():
            st["draft"] = self.ws.rel(draft_path)
            self._line("✓", a, f"draft: {self.ws.rel(draft_path)}")
            return
        cands = candidates(a, inv.policies)
        wired = [c for c in cands if c[1] >= 1.0]
        if strategy == "recommended":
            how = "1" if wired else "t"
        elif strategy == "templates":
            how = "t"
        else:
            how = self._menu(a, cands)
        if how in ("1", "2", "3") and int(how) <= len(cands):
            ref, score, _why = cands[int(how) - 1]
            if score >= 1.0:
                self._adopt(a, st, ref)
            else:
                self._copy_existing(a, st, ref, draft_path)
            return
        if how == "s":
            st["skipped"] = True
            self._line("·", a, "skipped")
            return
        if how == "a":
            self._assistant(a, st, draft_path)
            return
        lim = self.limits_for(a, ask=how == "t")  # in the studio or an editor the operator writes them
        text, _notes = L.policy_text(a, lim)
        if self.ws.plan_only:
            self._line("·", a, f"--plan-only: would write {self.ws.rel(draft_path)}")
            return
        self.ws.write_text(draft_path, text)
        d = type("Drafted", (), {"rules": [ln for ln in text.splitlines() if "STATE_CONSTRAINT" in ln]})()
        st["draft"] = self.ws.rel(draft_path)
        if how == "w":
            self._studio(a, st, draft_path, active_path)
        elif how == "e":
            self.console.print(f"    opening {self.ws.rel(draft_path)} in {self.ws.editor()} (a template to start from)")
            self.ws.open_in_editor(draft_path)
            self._line("✓", a, f"written in your editor: {self.ws.rel(draft_path)}")
        else:
            self._line("✓", a, f"{plural(len(d.rules), 'rule')} from templates: {self.ws.rel(draft_path)}")

    def limits_for(self, a: Agent, ask: bool = True) -> "L.Limits":
        """The agent's limits: what was set before (or the defaults), the command-line flags, and, at a
        terminal, the operator's own numbers. Saved in the workspace; the policy is made from them."""
        from .wire_cmd import scan_probe

        key = D.agent_key(a)
        probe, _root = scan_probe(self.args, self.ws)
        base = a.project or (probe.home() if a.kind == "assistant" else None)
        scope = [probe.real_path(base)] if base else []
        lim = L.defaults(a, scope, L.load(self.ws, key))
        try:
            applied = L.apply_flags(lim, key, getattr(self.args, "limit", None) or [], getattr(self.args, "add_tool", None) or [],
                                    getattr(self.args, "profile", None), names=(a.display_name, a.id))
        except L.LimitError as e:
            self.console.print(f"  [high]{e}[/high]")
            raise StopFlow()
        lim = L.defaults(a, scope, lim)  # tools added on the command line get their kind
        for item in applied:
            self.console.print(Text.assemble(("    ", ""), (a.display_name, "head"), (f"  {item}", "muted")))
        if self.interactive and ask:
            self._ask_limits(a, lim)
        if not self.ws.plan_only:
            L.save(self.ws, lim)
        return lim

    def _ask_limits(self, a: Agent, lim: "L.Limits") -> None:
        """At a terminal: what each tool may do, in the operator's own numbers (board.ask_limits)."""
        from .board import ask_limits

        self.console.print()
        ask_limits(self, self.console, a, lim)

    def _line(self, mark: str, a: Agent, text: str) -> None:
        style = {"✓": "ok", "·": "muted", "✗": "high"}.get(mark, "text")
        self.console.print(Text.assemble(("  ", ""), (mark + " ", style), (a.display_name, "head"), ("  " + text, "muted")))

    def _menu(self, a: Agent, cands) -> str:
        risky = ", ".join(sorted({t.risk_class for t in a.tools if t.risk_class != "READ"})) or "read only"
        self.console.print()
        self.console.print(Text.assemble(("  ", ""), (a.display_name, "head"), (f"   {_n(len(a.tools), 'tool')} · {risky}", "muted")))
        rows = []
        for i, (ref, score, why) in enumerate(cands, 1):
            name = ref.policy_id or ref.domain or Path(ref.path).name
            label = "keep (already in use)" if score >= 1.0 else f"start from it ({score * 100:.0f}% fit)"
            rows.append((str(i), Text.assemble((name, "head"), (f"  {label} · {why}", "muted"))))
        rows += [("t", "draft from templates (limits, approvals, allowlists per risk class)"),
                 ("w", "write it in the studio (edit, Z3 and TLA+ checks, suggestions, go live)"),
                 ("e", "write it in your own editor (a template to start from)"),
                 ("a", "draft it with your AI assistant"), ("s", "skip for now")]
        self.console.print(Padding(grid(*rows, label_width=2), (0, 0, 0, 4)))
        default = "1" if cands and cands[0][1] >= 1.0 else "t"
        return self.choose("choice", [str(i) for i in range(1, len(cands) + 1)] + ["t", "w", "e", "a", "s"], default)

    def _studio(self, a: Agent, st: Dict[str, Any], draft_path: Path, active_path: Path) -> None:
        """Open the studio on the agent's template draft; setup continues when it closes."""
        from .studio.command import launch

        if not self.interactive:
            self._line("·", a, f"template draft: {self.ws.rel(draft_path)} (open it with cslcore studio --agent {D.agent_key(a)})")
            return
        try:
            msg = launch(self.ws, self.inv, path=str(draft_path), agent=D.agent_key(a))
        except Exception as e:  # the studio never ends the setup
            msg = ""
            self.console.print(f"    [warn]the studio closed with an error ({type(e).__name__}: {e}); the draft is kept[/warn]")
        if active_path.exists() and not draft_path.exists():
            st.pop("draft", None)
            st["policy"] = self.ws.rel(active_path)
            self._line("✓", a, f"live from the studio: {self.ws.rel(active_path)}")
        elif draft_path.exists():
            self._line("✓", a, f"draft from the studio: {self.ws.rel(draft_path)}" + (f" · {msg}" if msg else ""))
        else:
            st.pop("draft", None)
            self._line("·", a, "nothing saved in the studio")

    def _adopt(self, a: Agent, st: Dict[str, Any], ref: PolicyRef) -> None:
        st["policy"] = ref.path
        st["adopted"] = True
        if not self.ws.plan_only:
            adopted = dict(self.ws.load_state().get("adopted") or {})
            adopted[a.id] = ref.path
            self.set_state("adopted", adopted)
        self._line("✓", a, f"keeps {short(ref.path)} as it is (adopted, not copied)")

    def _copy_existing(self, a: Agent, st: Dict[str, Any], ref: PolicyRef, draft_path: Path) -> None:
        from .analysis.coverage import suggest, tool_variable
        text = self.policy_text(ref.path) or ""
        tv = tool_variable(ref, a.tools)
        names = [t.name for t in a.tools]
        renames = {v: suggest(v, names) for v in ref.vocabulary.get(tv, [])} if tv else {}
        text = D.fix_text(text, {k: v for k, v in renames.items() if v and v != k}, {})
        if self.ws.plan_only:
            return
        self.ws.write_text(draft_path, text)
        st["draft"] = self.ws.rel(draft_path)
        fixed = sum(1 for k, v in renames.items() if v and v != k)
        self._line("✓", a, f"draft from {Path(ref.path).name}" + (f", {fixed} tool names aligned" if fixed else ""))

    def _assistant(self, a: Agent, st: Dict[str, Any], draft_path: Path) -> None:
        key = D.agent_key(a)
        claude = self.ws.find_program("claude")
        if claude and self.interactive and self.ask("Claude Code is installed. Let it draft the policy now (uses your Claude account)?", True):
            import importlib.util
            import json as _json
            if importlib.util.find_spec("mcp") is None:
                self.console.print("    [warn]the MCP extra is missing: pip install \"csl-core[mcp]\"[/warn]")
            else:
                cfg = {"mcpServers": {"csl-core": {"command": sys.executable, "args": ["-m", "chimera_core.mcp.server"],
                                                   "env": {"CSL_VENOM_WORKSPACE": str(self.ws.root)}}}}
                prompt = (f"Use the csl-core MCP tools to draft a CSL policy for the agent {key}: call venom_policy_context "
                          f"and venom_agent for {key}, write the policy with its real tool names, check it with verify_policy, "
                          f"and save it with venom_save_draft(agent_id=\"{key}\"). Do not do anything else.")
                tools = ",".join(f"mcp__csl-core__{t}" for t in ("venom_policy_context", "venom_agent", "verify_policy", "venom_save_draft"))
                with self.console.status(f"  Claude Code is drafting a policy for {a.display_name} ...", spinner="dots"):
                    rc = self.ws.run_program([claude, "-p", prompt, "--mcp-config", _json.dumps(cfg), "--allowedTools", tools])
                if draft_path.exists():
                    st["draft"] = self.ws.rel(draft_path)
                    self._line("✓", a, f"drafted by Claude Code: {self.ws.rel(draft_path)} (verified next)")
                    return
                self.console.print(f"    [warn]no draft came back (exit {rc})[/warn]")
        st["assistant"] = True
        self.console.print(Panel(Group(
            Text("In Claude Code, Claude Desktop or Cursor with the csl-core MCP server, run:", style="text"),
            Text(f"  /mcp__csl-core__venom_draft {key}", style="brand"),
            Text(""),
            Text("No csl-core server yet? Claude Code: claude mcp add csl-core -- csl-core-mcp", style="muted"),
            Text(f"The assistant verifies the draft and saves it to {self.ws.rel(draft_path)}; nothing is activated.", style="muted")),
            title=Text(f" draft with your assistant · {a.display_name} ", style="brand"), title_align="left",
            box=box.ROUNDED, border_style="brand.dim", padding=(0, 1)))
        if not self.interactive:
            return
        while True:
            ans = self.text_input("press Enter once the draft is saved, or s to continue without it")
            if draft_path.exists():
                st["draft"] = self.ws.rel(draft_path)
                st.pop("assistant", None)
                self._line("✓", a, f"draft received: {self.ws.rel(draft_path)}")
                return
            if ans.strip().lower() == "s":
                return
            self.console.print(f"    [warn]no draft at {self.ws.rel(draft_path)} yet[/warn]")

    # -- verify / map ---------------------------------------------------------------------
    def verify(self) -> bool:
        from .policy.workbench import gate_panel
        ok = True
        for aid, st in self.agents_state().items():
            path = st.get("draft") or (st.get("policy") if st.get("adopted") else None)
            if not path:
                continue
            text = self.policy_text(path)
            if text is None:
                continue
            g = verify_text(text)
            st["verified"] = g.ok
            label = short(path) if not st.get("adopted") or st.get("draft") else f"{short(path)} (adopted)"
            if g.ok:
                self.console.print(Text.assemble(("  ✓ ", "ok"), (label, "head"), (f"  {plural(g.rules, 'rule')} · Z3: no contradictions", "muted")))
                continue
            self.console.print(gate_panel(g, path))
            if st.get("adopted") and not st.get("draft"):
                self.console.print("  [warn]this policy is already in use by your code; fix it there and re-run setup[/warn]")
                continue
            ok = False
            if self.interactive and self.ask("Open it in the studio to fix it (Z3 and TLA+ with suggestions)?", True):
                self._review_in_studio(aid, st)
                return self.verify()
        self.save()
        if not ok:
            self.console.print("  [high]fix the drafts above, then re-run cslcore setup[/high]")
            return ok
        if self.interactive and self._offer_review():
            return self.verify()
        return ok

    def _offer_review(self) -> bool:
        """Drafts can be reviewed in the studio before they are mapped and activated."""
        drafts = [(aid, st) for aid, st in self.agents_state().items() if st.get("draft")]
        if not drafts:
            return False
        self.console.print()
        rows = [(str(i), Text.assemble((st.get("key") or aid, "head"), (f"  {short(st['draft'])}", "muted")))
                for i, (aid, st) in enumerate(drafts, 1)]
        self.console.print(Padding(grid(*rows, label_width=2), (0, 0, 0, 4)))
        pick = self.text_input("Review a draft in the studio first? number, or Enter to continue", "").strip()
        if not pick.isdigit() or not 1 <= int(pick) <= len(drafts):
            return False
        aid, st = drafts[int(pick) - 1]
        self._review_in_studio(aid, st)
        return True

    def _review_in_studio(self, aid: str, st: Dict[str, Any]) -> None:
        a = next((x for x in self.load_inventory().agents if x.id == aid), None)
        if a is None or not st.get("draft"):
            return
        key = D.agent_key(a)
        self._studio(a, st, self.ws.root / st["draft"], self.ws.policies / f"{key}.csl")
        self.save()

    def map(self) -> bool:
        from .layers.governance import read_policy
        from .mapping import codegen, harness
        from .mapping.assistant import compile_guard, mapping_path, parse_mapping_arg, record, render_results, test_mapping
        from .mapping.spec import build_spec

        inv = self.load_inventory()
        by_id = {a.id: a for a in inv.agents}
        ok = True
        self.console.print(Text("  Every mapping is tested with case variants, unknown values, missing keys, wrong types "
                                "and range edges, and every derived check (path in scope, command allowlisted, "
                                "destination allowed) with bypass tricks. None of them may end in ALLOW.", style="muted"))
        self.console.print()
        for aid, st in self.agents_state().items():
            rel = st.get("draft") or st.get("policy")
            a = by_id.get(aid)
            if not rel or a is None or st.get("verified") is False or st.get("protected"):
                continue  # the board bound its policy and tested its mapping already
            text = self.policy_text(rel) or ""
            final_rel = rel if st.get("adopted") and not st.get("draft") else f"policies/{st['key']}.csl"
            ref = read_policy(rel if Path(rel).is_absolute() else str(self.ws.root / rel), text, "draft")
            spec = build_spec(a, ref, L.load(self.ws, D.agent_key(a)))
            if st.get("adopted") and self.interactive:
                own = self.text_input(f"{a.display_name}: test your own mapper? (path.py:function, 'openclaw', or Enter to generate one)")
                if own.strip():
                    path, func = parse_mapping_arg(own.strip())
                    allowed = self._mapper_checks(a, spec, st)
                    st["own_mapper"] = own.strip()
                    try:
                        res = test_mapping(self.console, self.ws, spec, text, final_rel, path, func, allowed=allowed)
                    except SystemExit as e:  # an own mapper that cannot be loaded never ends the setup
                        self.console.print(f"    [warn]could not test {own.strip()}: {e}[/warn]")
                        self.console.print(f"    [muted]try it later: cslcore map --agent {a.display_name} --test --mapping {own.strip()}[/muted]")
                    else:
                        st["fail_open"] = len(res.fail_open)
                        if res.fail_open:
                            self.console.print("  [warn]your mapper lets some inputs through that should block (listed above); "
                                               "the generated mapping below closes them, and MAPPING.md shows how to fix "
                                               "yours[/warn]")
            path = mapping_path(self.ws, a)
            code = codegen.generate(spec, final_rel, str(path.parent))
            import types
            mod = types.ModuleType("_venom_setup_mapping")
            mod.__file__ = str(path)  # the generated code reads its scope roots from its own place
            exec(compile(code, "<generated mapping>", "exec"), mod.__dict__)  # generator output
            res = harness.run(spec, mod.map_call, compile_guard(text), mod)
            st["mapping"] = self.ws.rel(path)
            st.setdefault("fail_open", len(res.fail_open))
            if not self.ws.plan_only:
                record(self.ws, spec, res, self.ws.rel(path))
            mark = ("✓ ", "ok") if not res.fail_open else ("✗ ", "high")
            self.console.print(Text.assemble(("  ", ""), mark, (a.display_name, "head"),
                                             (f"  {_n(len(res.cases), 'case')} · {len(res.fail_open)} fail-open", "muted"),
                                             (f"  → {self.ws.rel(path)}", "muted")))
            if res.fail_open:
                render_results(self.console, res, spec, "generated mapping", final_rel)
                ok = False
                continue
            if not self.ws.plan_only:
                self.ws.write_text(path, code)
        self.save()
        return ok

    def _mapper_checks(self, a: Agent, spec, st: Dict[str, Any]) -> Dict[str, List[str]]:
        """Which policy variables the operator's mapper computes as checks, and one value it accepts for each
        kind, so the bypass tricks can be built (kept in the setup state for later runs)."""
        from .mapping.spec import apply_classify

        saved = st.get("mapper_checks") or {}
        items = list(saved.get("classify") or [])
        flags = [v.name for v in spec.variables if v.kind == "flag" and v.source != "constant"]
        if not items and (spec.classify or flags):
            hint = ", ".join(f"{v}={k}" for v, (k, _p) in spec.classify.items()) or f"{flags[0]}=scope:file_path"
            raw = self.text_input(f"{a.display_name}: which variables does your mapper compute as checks? "
                                  f"VAR=scope|command|destination[:param], comma separated (e.g. {hint}; Enter for none)")
            items = [x.strip() for x in raw.split(",") if x.strip()]
        for problem in apply_classify(spec, items):
            self.console.print(f"    [warn]{problem}[/warn]")
        allowed: Dict[str, List[str]] = {k: list(v) for k, v in (saved.get("allowed") or {}).items()}
        prompts = {"scope": "a folder it accepts as in scope (e.g. /srv/app)",
                   "command": "a command it accepts (e.g. git status)",
                   "destination": "a destination it accepts (e.g. https://api.example.com/)"}
        for kind in dict.fromkeys(k for k, _p in spec.classify.values()):
            if not allowed.get(kind):
                value = self.text_input(f"  bypass tests for {kind}: {prompts[kind]}, Enter to skip").strip()
                if value:
                    allowed[kind] = [value]
        st["mapper_checks"] = {"classify": items, "allowed": allowed}
        return allowed

    # -- mode and wiring ----------------------------------------------------------------------
    def _choose_modes(self, agents: List[Tuple[str, Agent, Dict[str, Any]]]) -> None:
        from .controls import Controls

        controls = Controls(self.ws)
        current = controls.default_mode()
        given = getattr(self.args, "mode", None)
        if given:
            default = given
        elif current:
            default = current
        else:
            # nothing chosen yet: agents already active keep their mode; the ones activated now for the
            # first time start in the mode picked here (block unless chosen otherwise)
            if self.interactive:
                self.console.print(Padding(grid(
                    ("block", "what the limits do not allow is stopped; the check after activation shows what runs"),
                    ("log", "nothing is stopped yet; every decision is recorded as ALLOW or WOULD BLOCK"),
                    label_width=6), (0, 0, 0, 4)))
                self.new_mode = self.choose("Mode for agents activated now for the first time", ["block", "log"], "block")
            else:
                self.new_mode = "block"
            enforcing = [k for k, a, s in agents if s.get("adopted") and k not in controls.all()]
            if enforcing and not self.ws.plan_only:
                controls.set_many(enforcing, "block")
            self.console.print(Text.assemble(("  mode  ", "label"),
                                             (self.new_mode.upper(), "brand" if self.new_mode == "block" else "warn"),
                                             (" for agents activated now for the first time; agents already active keep "
                                              "their mode", "text")))
            self.console.print(Text("        change any time: cslcore watch (m, M) or cslcore mode --agent NAME log|block",
                                    style="muted"))
            return
        if not self.ws.plan_only and default != current:
            controls.set_default(default)
        other = "block" if default == "log" else "log"
        keys = [k for k, _a, _s in agents]
        picked: List[str] = []
        if self.interactive and keys:
            pattern = self.text_input(f"Agents to start in {other.upper()} instead (names or patterns like pay-*, Enter for none)")
            for pat in [p.strip() for p in pattern.replace(",", " ").split() if p.strip()]:
                picked += [k for k in keys if fnmatch.fnmatch(k, pat) and k not in picked]
        # agents whose own code already enforces (0.5.1) stay in block unless chosen otherwise
        enforcing = [k for k, a, s in agents if s.get("adopted") and k not in picked and default == "log"]
        if not self.ws.plan_only:
            if picked:
                controls.set_many(picked, other)
            if enforcing:
                controls.set_many(enforcing, "block")
        self.console.print(Text.assemble(("  mode  ", "label"), (default.upper(), "warn" if default == "log" else "brand"),
                                         (" for all agents", "text")))
        if picked:
            self.console.print(Text.assemble(("        ", ""), (other.upper(), "brand" if other == "block" else "warn"),
                                             (f" for {', '.join(picked)}", "text")))
        if enforcing:
            self.console.print(Text.assemble(("        ", ""), ("BLOCK", "brand"),
                                             (f" kept for {', '.join(enforcing)} (already enforcing in your code)", "text")))
        self.console.print(Text("        change any time: cslcore watch (m, M) or cslcore mode --all log", style="muted"))

    def wire(self) -> bool:
        from .wire import observe_snippet, snippet_for

        inv = self.load_inventory()
        by_id = {a.id: a for a in inv.agents}
        agents = [(st["key"], by_id[aid], st) for aid, st in self.agents_state().items()
                  if aid in by_id and (st.get("mapping") or st.get("adopted")) and not st.get("protected")]
        if not agents:
            done = sum(1 for st in self.agents_state().values() if st.get("protected"))
            self.console.print(f"  [muted]{'every agent was wired on the protection board' if done else 'no agent to wire yet'}"
                               "[/muted]")
            return True
        self._choose_modes(agents)
        self.console.print()
        out: List[str] = []
        snippets = []
        from . import wiring
        from .wire_cmd import scan_probe

        probe, _root = scan_probe(self.args, self.ws)
        self.console.print(Text("  WIRING   after activation, the guard goes into each agent's call path: each change is "
                                "shown as a diff and made only when you confirm (--wire with --yes)", style="label"))
        by_hand = []
        for key, a, st in agents:
            if st.get("adopted"):
                sn = observe_snippet(a, key, short(str(self.ws.root)))
            else:
                sn = snippet_for(a, key, f"policies/{key}.csl", st["mapping"], short(str(self.ws.root)))
            snippets.append(sn)
            out.append(f"## {a.display_name}\n\n{sn.title}\n\n```{sn.language}\n{sn.code}\n```\n\n" + " ".join(sn.notes) + "\n")
            plan = wiring.plan_for(a, key, self.ws, probe)
            if st.get("adopted"):
                how, style = "its own guard, in its own code (optional: observe() adds decision logs and modes)", "muted"
            elif plan.kind == "done":
                how, style = "wired already", "ok"
            elif plan.kind == "hook":
                how, style = "automatic: a Claude Code hook decides every tool call", "text"
            elif plan.kind == "code" and plan.changes:
                how, style = f"automatic: a decorator on {wiring_count(plan)}", "text"
            else:
                how, style = "by hand: " + (plan.note or sn.title), "warn"
                by_hand.append(sn)
            self.console.print(Text.assemble(("    ", ""), (a.display_name, "head"), (f"  {how}", style)))
        if len(by_hand) == 1:
            self.console.print(Padding(Syntax(by_hand[0].code, by_hand[0].language, theme="ansi_dark",
                                              background_color="default", word_wrap=True), (1, 0, 0, 4)))
        if out and not self.ws.plan_only:
            self.ws.write_text(self.ws.venom / "wiring.md", "# Integration snippets\n\n" + "\n".join(out))
            self.console.print(Text.assemble(("    the exact code for each: ", "muted"), (self.ws.rel(self.ws.venom / "wiring.md"), "brand")))
        self.save()
        return True

    def wire_now(self) -> None:
        """After activation: make the wiring change in each agent whose policy is now active, so the
        guard is really in its call path (a hook, or a decorator on each tool function). Shown as a
        diff and confirmed; --yes needs --wire. Never before a policy is active: a guard without
        its policy refuses every call."""
        from . import wiring
        from .bindings import Bindings
        from .wire_cmd import rescan, scan_probe, show_plan

        if self.ws.plan_only:
            return
        apply_all = bool(getattr(self.args, "wire", False))
        if not apply_all and not self.interactive:
            if getattr(self.args, "yes", False):
                self.console.print("  [muted]the wiring change is not made with --yes alone; pass --wire, or run "
                                   "cslcore wire[/muted]")
            return
        inv = self.load_inventory()
        bound = set(Bindings(self.ws).all())
        by_id = {a.id: a for a in inv.agents}
        probe, root = scan_probe(self.args, self.ws)
        plans = []
        agent_of = {}
        for aid, st in self.agents_state().items():
            a = by_id.get(aid)
            if a is not None and st.get("key") in bound:
                plans.append(wiring.plan_for(a, st["key"], self.ws, probe))
                agent_of[st["key"]] = a
        todo = [p for p in plans if p.changes and p.kind in ("hook", "code")]
        manual = [p for p in plans if p.kind == "manual"]
        if not todo and not manual:
            return
        self.console.print()
        self.console.print(Text("  WIRE     the change that puts each guard in its agent's call path", style="label"))
        applied = 0
        if todo and (apply_all or self.ask(f"Make it now in {plural(len(todo), 'agent')}? Each diff is shown first", True)):
            from .wire_cmd import env_ready

            chosen = []
            for p in todo:
                show_plan(self.console, p, ws=self.ws)
                if not env_ready(self.console, self.args, self.ws, agent_of[p.key], p,
                                 self.ask if self.interactive else None):
                    continue
                if apply_all or self.ask(f"Wire {p.agent}?", True):
                    chosen.append((agent_of[p.key], p))
            applied = wiring.apply_many(chosen, self.ws, probe, on_error=lambda p, e: self.console.print(
                f"  [high]{p.agent} not wired: {e}[/high]"))
        for p in manual:
            self.console.print(Text.assemble(("  ", ""), (p.agent, "head"), (f"  {p.note}", "warn")))
        if applied:
            rescan(self.args, self.console, self.ws, root)
            self.inv = None
            self.console.print(Text.assemble(("  ✓ wired ", "ok"), (plural(applied, "agent"), "head"),
                                             ("  undo any time: cslcore wire --undo", "muted")))

    def activate(self) -> bool:
        from .policy.workbench import diff_text
        pending = [(aid, st) for aid, st in self.agents_state().items() if st.get("draft") and st.get("verified")]
        if not pending:
            self.console.print("  [muted]nothing to activate[/muted]")
            return True
        each = False
        if self.interactive and not self.activate_flag and len(pending) > 1:
            for _aid, st in pending:
                self.console.print(Text.assemble(("    ", ""), (st["draft"], "head"), ("  verified", "ok")))
            pick = self.choose(f"Activate all {len(pending)} verified drafts? y all · n none · c one by one", ["y", "n", "c"], "y")
            if pick == "n":
                self.console.print("  [muted]left as drafts: cslcore policy activate <name> when ready[/muted]")
                return True
            each = pick == "c"
            self.activate_flag = pick == "y"
        for aid, st in pending:
            src = self.ws.root / st["draft"]
            dst = self.ws.policies / f"{st['key']}.csl"
            text = self.ws.read(src)
            if text is None:
                continue
            if not verify_text(text).ok:
                self.console.print(f"  [high]{st['draft']} no longer verifies; not activated[/high]")
                continue
            old = self.ws.read(dst)
            if old and old != text:
                d = diff_text(old, text, self.ws.rel(dst), st["draft"])
                if d:
                    self.console.print(d)
            if self.ws.plan_only:
                self.console.print(f"  [muted]--plan-only: would activate {self.ws.rel(dst)}[/muted]")
                continue
            if self.activate_flag:
                go = True
            elif self.interactive or each:
                go = self.ask(f"Activate {self.ws.rel(dst)}?", True)  # verified, its diff shown: as everywhere else
            else:
                self.console.print(f"  [muted]{st['draft']} left as a draft (--yes never activates; pass --activate)[/muted]")
                go = False
            if go:
                from .controls import mode_on_activation
                first = old is None
                self.ws.write_text(dst, text)
                if getattr(self.args, "mode", None):  # chosen for all agents: the workspace default says it
                    from .controls import Controls
                    mode = Controls(self.ws).get(st["key"]).mode
                else:  # a first activation: the mode picked in step 9 (block unless chosen otherwise)
                    mode = mode_on_activation(self.ws, st["key"], first, getattr(self, "new_mode", None) if first else None)
                self.ws.remove(src)
                st.pop("draft", None)
                st["policy"] = self.ws.rel(dst)
                if st.get("mapping"):
                    from .bindings import Bindings
                    Bindings(self.ws).bind(st["key"], dst, st["mapping"])
                self.console.print(Text.assemble(("  ✓ active ", "ok"), (self.ws.rel(dst), "head"),
                                                 (f"  {mode} mode" + (" (new: block unless chosen otherwise)"
                                                                      if first and mode == "block" else ""), "muted")))
        self.save()
        return not any(st.get("draft") for _, st in pending)

    # -- home ---------------------------------------------------------------------------
    def home_panel(self) -> Panel:
        from .controls import Controls
        from .watch import Tail, WatchModel

        inv = self.load_inventory()
        controls = Controls(self.ws)
        agents = self.agents_state()
        with_policy = sum(1 for st in agents.values() if st.get("policy"))
        ctl = {st.get("key"): controls.get(st.get("key")) for st in agents.values() if st.get("key")}
        exempt = sum(1 for c in ctl.values() if c.exempt)
        disabled = sum(1 for c in ctl.values() if c.disabled)
        default = controls.default_mode() or "log"
        other = sorted(k for k, c in ctl.items() if c.mode != default)
        model = WatchModel()
        Tail(self.ws).poll(model)
        day = sum(1 for t in model.minute if t >= (datetime.now(timezone.utc).timestamp() - 86400))
        high = sum(1 for f in inv.findings if f.severity == "high")
        rows = [
            ("agents", f"{plural(len(inv.agents), 'agent')} discovered · {with_policy} with a policy"
             + (f" · {exempt} exempt" if exempt else "") + (f" · {disabled} disabled" if disabled else "")),
            ("modes", Text.assemble((default.upper(), "warn" if default == "log" else "brand"), (" by default", "text"),
                                    (f" · {('BLOCK' if default == 'log' else 'LOG')}: {', '.join(other)}" if other else "", "text"))),
            ("decisions", f"{model.total:,} recorded · {day:,} in the last 24 hours · "
             f"{(model.flagged / model.total * 100) if model.total else 0:.0f}% would block or blocked" if model.total
             else "none yet: wire an agent (see .csl/venom/wiring.md)"),
            ("last scan", f"{inv.host.scanned_at.replace('T', ' ')[:16]} · " + (f"{high} high findings" if high else "no high findings")),
        ]
        if model.total and model.flagged / model.total > 0.2:
            nxt = "many calls would be blocked: tune the top rules in the panel (w, then Tab)"
        elif model.total and default == "log":
            nxt = "decisions look settled? switch agents to block in the panel (m or M)"
        elif not model.total:
            nxt = "wire your agents (one change each, .csl/venom/wiring.md), then watch them in the panel"
        else:
            nxt = "scan again to pick up new agents and tools (s)"
        rows.append(("next", Text(nxt, style="brand")))
        menu = grid(("b", "the protection board: limits, policy, wiring and check per agent"),
                    ("s", "scan again and review what changed"), ("w", "open the live management panel"),
                    ("m", "the reach map"), ("p", "policies"), ("o", "modes and freezes"), ("q", "quit"), label_width=2)
        return Panel(Group(grid(*rows, label_width=10), Text(""), menu),
                     title=Text(f" CSL-Core {VENOM_VERSION} · {self.ws.root.name} ", style="brand"), title_align="left",
                     box=box.ROUNDED, border_style="brand.dim", padding=(0, 1))

    def home(self) -> Optional[int]:
        """Returns an exit code to stop, or None to start a new setup cycle."""
        self.console.print(self.home_panel())
        pick = self.choose("choice", ["b", "s", "w", "m", "p", "o", "q"], "b")
        if pick == "s":
            return None
        if pick == "b":
            agents = self.targets()
            if agents:
                self.board(agents)
            else:
                self.console.print("  [ok]every agent is covered or exempt[/ok]")
            return EXIT_OK
        if pick == "w":
            from .watch import run_watch
            return run_watch(self.args)
        if pick == "p":
            from .policy.workbench import act_list
            return act_list(self.console, self.ws, self.load_inventory())
        if pick == "m":
            self.open_room("map")
            return EXIT_OK
        if pick == "o":
            from .controls import Controls
            from .exempt_cmd import _control_table
            _control_table(self.console, Controls(self.ws))
            self.console.print("  [muted]cslcore mode --all log|block · cslcore mode --agent ID log|block|--disable[/muted]")
            return EXIT_OK
        return EXIT_OK

    # -- driver -------------------------------------------------------------------------
    def run(self) -> int:
        completed = self.setup.get("completed", [])
        if len(completed) == len(STEPS) and self.interactive and not getattr(self.args, "restart", False):
            code = self.home()
            if code is not None:
                return code
        if getattr(self.args, "restart", False) or len(completed) == len(STEPS):
            if completed:
                self.console.print("  [muted]starting a new cycle[/muted]" if not getattr(self.args, "restart", False) else "  [muted]restarting[/muted]")
            self.setup["completed"] = []
            self.setup["cycle"] = int(self.setup.get("cycle", 0)) + 1
            completed = []
        start = next((i for i, s in enumerate(STEPS) if s not in completed), 0)
        if start > 0:
            self.console.print(Text.assemble(("  resuming at step ", "muted"), (f"{start + 1} {TITLES[STEPS[start]]}", "brand")))
        stop_after = getattr(self.args, "stop_after", None)
        try:
            for step in STEPS[start:]:
                self.header(step)
                ok = getattr(self, step)()
                if not ok:
                    self.console.print(f"\n  [warn]stopped at {TITLES[step]}[/warn]: re-run [brand]cslcore setup[/brand] to continue")
                    return EXIT_INCOMPLETE
                if not self.ws.plan_only:
                    self.done(step)
                if stop_after and step == stop_after:
                    return EXIT_OK
                if step in PAUSE_AFTER:
                    self.console.print()
                    self.pause(reach_map=step in ("inventory", "findings"))
        except StopFlow:
            self.console.print("  [muted]stopped; progress is saved. Run cslcore setup to continue.[/muted]")
            return EXIT_INCOMPLETE
        self.wire_now()
        self.check_all()
        self.console.print()
        self.console.print(self.summary())
        from .rooms import ask_next, interactive
        if self.interactive and interactive(self.console):
            pick = ask_next(self.console, {"w": "watch it live", "m": "reach map", "q": "quit"})
            if pick in ("w", "m"):
                self.open_room("watch" if pick == "w" else "map")
        return EXIT_OK

    def check_all(self) -> None:
        """Sample calls for every agent with a policy made from its limits, decided by that policy:
        what runs and what stops, against what the operator set."""
        from . import check

        agents = {a.id: a for a in self.load_inventory().agents}
        shown = False
        for aid, st in sorted(self.agents_state().items(), key=lambda kv: kv[1].get("key", "")):
            a = agents.get(aid)
            if a is None or not st.get("policy"):
                continue
            report = check.run(self.ws, a)
            if not report.cases:
                continue
            if not self.ws.plan_only:
                from .board import store_check
                store_check(self.ws, st.get("key", D.agent_key(a)), report)
            if not shown:
                self.console.print()
                self.console.print(Text("  Check: sample calls decided by each active policy", style="brand"))
                shown = True
            self.console.print()
            from .board import approval_note
            check.show(self.console, report, st.get("key", a.display_name), compact=report.ok,
                       approval=approval_note(self.args, self.ws, a))

    def summary(self) -> Panel:
        """Each agent as it really is: protected only when its guard is in its call path, in block
        mode; in log mode its calls are recorded and nothing is stopped."""
        from . import board as B

        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="head", no_wrap=True)
        rows.add_column(no_wrap=True)
        rows.add_column(style="muted", overflow="fold")
        waiting = protected = recording = 0
        inv = self.load_inventory()
        agents = {a.id: a for a in inv.agents}
        for aid, st in sorted(self.agents_state().items(), key=lambda kv: kv[1].get("key", "")):
            key = st.get("key", aid)
            a = agents.get(aid)
            if st.get("policy") and a is not None:
                r = B.row_for(self.ws, a, self.agents_state(), self.args, inv.policies)
                where = short(st["policy"]) + (" (adopted)" if st.get("adopted") else "")
                if r.frozen:
                    rows.add_row(key, Text("frozen", style="high"), f"{where} · every call is blocked until unfrozen")
                elif r.policy == "adopted" or (r.own and not r.policy):
                    protected += 1
                    rows.add_row(key, Text("its own guard", style="ok"), f"{where} · enforced by its own code")
                elif r.wired in ("wired", "partly") and r.mode == "block":
                    protected += r.wired == "wired"
                    check = {"ok": " · check ✓", "failed": " · check ✗: cslcore limits --agent " + key + " --check"}
                    rows.add_row(key, Text("protected" if r.wired == "wired" else "partly", style="ok" if r.wired == "wired"
                                           else "warn"),
                                 f"{where} · block mode: stops what its limits do not allow{check.get(r.check, '')}"
                                 + (f" · {r.wiring_note}" if r.wired == "partly" else ""))
                elif r.wired in ("wired", "partly"):
                    recording += 1
                    rows.add_row(key, Text("recording only", style="warn"),
                                 f"{where} · log mode: its calls are recorded, nothing is stopped · "
                                 f"to stop: cslcore mode --agent {key} block")
                elif r.wired == "manual":
                    rows.add_row(key, Text("wire by hand", style="warn"),
                                 f"{where} · {r.wiring_note or 'see .csl/venom/wiring.md'}")
                else:
                    rows.add_row(key, Text("not wired", style="warn"),
                                 f"{where} · nothing stops it yet: cslcore wire --agent {key}")
            elif st.get("draft"):
                rows.add_row(key, Text("draft", style="warn"), f"{st['draft']} · activate: cslcore policy activate {key} (or ctrl+l in cslcore studio)")
            elif st.get("assistant"):
                waiting += 1
                rows.add_row(key, Text("waiting", style="warn"), "for your assistant's draft")
            elif st.get("skipped"):
                rows.add_row(key, Text("skipped", style="muted"), "")
        total = len(self.agents_state())
        head = f" setup complete · {protected} of {total} protected" + (f" · {recording} recording only" if recording else "") + " "
        good = not waiting and protected == total
        title = Text(head if not waiting else head.rstrip() + ", agents waiting ", style="ok" if good else "warn")
        undo = Group(
            Text.assemble(("Undo or loosen, any time (running agents follow on their next call):", "muted")),
            Text.assemble(("  ", ""), ("cslcore mode --agent NAME log", "brand"),
                          ("            record only, stop nothing (", "muted"), ("--all log", "brand"), (" for every agent)", "muted")),
            Text.assemble(("  ", ""), ("cslcore limits --agent NAME --set TOOL=FREE..MAX", "brand"),
                          ("   raise a limit; ", "muted"), ("--decide TOOL=allow", "brand"), (" lets a tool run", "muted")),
            Text.assemble(("  ", ""), ("cslcore wire --undo --agent NAME", "brand"),
                          ("         take the guard out: its files go back as they were", "muted")),
            Text.assemble(("  ", ""), ("cslcore mode --agent NAME --disable", "brand"),
                          ("      the other way: stop every call at once", "muted")))
        body = Group(rows, Text(""), undo, Text(""),
                     Text.assemble(("Next: ", "muted"), ("cslcore setup", "brand"),
                                   ("    the protection board (b): limits, wiring and checks per agent", "muted")),
                     Text.assemble(("      ", ""), ("cslcore watch", "brand"),
                                   ("    live decisions; l limits, w wiring, m mode, x freeze", "muted")),
                     Text.assemble(("      ", ""), ("cslcore studio", "brand"),
                                   ("   edit a policy, check it with Z3 and TLA+, go live", "muted")))
        return Panel(body, title=title, title_align="left", box=box.ROUNDED, border_style="ok" if good else "warn", padding=(0, 1))


def wiring_count(plan) -> str:
    n = len(plan.wrapped)
    return f"{n} tool function{'s' if n != 1 else ''}"


def cmd_setup(args) -> int:
    return Flow(args).run()
