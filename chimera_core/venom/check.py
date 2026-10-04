"""
The policy check: does the active policy, through its mapping, do what the agent's limits say?

For every tool, sample calls are made from the limits themselves (an amount at the free limit, one
in the approval band with and without approval, one over the maximum; a call of each category a
command, a query or a path can fall in) and decided by the real guard: the bound policy and its
generated mapping, in block mode, whatever mode the agent runs in. The expected outcome comes from
the limits, not from the policy, so the check is independent of what it checks.

Categories (what a command, a query or a path is) are given as category values: the mapping's own
classifiers return the value of the case instead of reading an argument. So no sample command text
is needed, and a mapping that never classifies a tool's calls shows up as a failed row.

Nothing runs and nothing is recorded: only the mapping and the guard are called.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .model import Agent, Tool
from .policy import limits as L

RUNS, STOPPED = "runs", "stopped"
COMMAND_WORDS = {"REMOTE_EXEC": "a command that runs downloaded code", "DESTRUCTIVE": "a command that destroys data",
                 "PRIVILEGE": "a command that takes admin rights", "SECRETS": "a command that reads secrets",
                 "EXFIL": "a command that sends data out", "PERSISTENCE": "a command that changes startup files",
                 "UNREADABLE": "a command that cannot be read"}
OK_CATEGORY = {"shell": {"command_class": "OK", "command_allowlisted": "YES"}, "sql": {"sql_class": "READ"},
               "write": {"path_class": "IN_SCOPE"}, "read": {"path_class": "IN_SCOPE"},
               "send": {"destination_allowlisted": "YES"}}
CLASSIFIERS = {"command_class": "args_command_class", "sql_class": "args_sql_class", "path_class": "args_path_class",
               "command_allowlisted": "command_allowed", "destination_allowlisted": "destination_allowed"}


@dataclass
class Case:
    tool: str
    what: str
    expected: str
    args: Dict[str, Any] = field(default_factory=dict)
    category: Dict[str, str] = field(default_factory=dict)  # derived variable -> its value for this case
    approval: bool = False
    got: str = ""

    @property
    def ok(self) -> bool:
        return self.expected == self.got


@dataclass
class Report:
    agent: str
    cases: List[Case] = field(default_factory=list)
    note: str = ""  # why there is nothing to check

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(c.ok for c in self.cases)

    @property
    def failed(self) -> List[Case]:
        return [c for c in self.cases if not c.ok]

    def counts(self) -> Tuple[int, int, int]:
        """(expected to run, expected to stop, not as expected)."""
        runs = sum(1 for c in self.cases if c.expected == RUNS)
        return runs, len(self.cases) - runs, len(self.failed)


# ---------------------------------------------------------------------------
# the cases, from the limits alone
# ---------------------------------------------------------------------------

def _sample(name: str, ptype: str, scope: List[str]) -> Any:
    """A harmless value for a parameter the case does not speak about."""
    n, t = name.lower(), (ptype or "").lower().split("[")[0]
    if t in ("int", "integer", "float", "number"):
        return 0
    if t in ("bool", "boolean"):
        return False
    if t in ("list", "array", "tuple", "set", "sequence"):
        return []
    if L.PATH_PARAMS.search(n):
        return (scope[0].rstrip("/") + "/notes.txt") if scope else "notes.txt"
    if "url" in n or "webhook" in n:
        return "https://example.com"
    return "sample"


def _number(tool: Tool, param: str, value: int) -> Any:
    """A number for a parameter; a list parameter gets that many items (its length is what counts)."""
    p = next((x for x in tool.params if x.name == param), None)
    if p is not None and (p.type or "").lower().split("[")[0] in ("list", "array", "tuple", "set", "sequence"):
        return ["item"] * value
    return value


def _band(tool: Tool, param: str, lo: int, hi: int, label: str, base: Dict[str, Any]) -> List[Case]:
    t = tool.name
    out = [Case(t, f"{label} {lo:,}", RUNS, {**base, param: _number(tool, param, lo)})]
    if lo < hi:
        out.append(Case(t, f"{label} {lo + 1:,}", STOPPED, {**base, param: _number(tool, param, lo + 1)}))
        out.append(Case(t, f"{label} {hi:,}", RUNS, {**base, param: _number(tool, param, hi)}, approval=True))
    out.append(Case(t, f"{label} {hi + 1:,}", STOPPED, {**base, param: _number(tool, param, hi + 1)}, approval=True))
    return out


def cases_for(tool: Tool, tl: L.ToolLimit, lim: L.Limits) -> List[Case]:
    t = tool.name
    base = {p.name: _sample(p.name, p.type or "", lim.scope) for p in tool.params}
    numbers = {p: v for p, v in (tl.numbers or {}).items() if not (tl.kind == "spend" and p == tl.amount_param)}
    for p, (lo, _hi) in numbers.items():
        base[p] = _number(tool, p, int(lo))  # an ordinary call stays inside every limit
    if tl.kind == "spend" and tl.amount_param:
        base[tl.amount_param] = 0
    if tl.decide == "block":
        return [Case(t, "any call", STOPPED, dict(base), approval=True)]
    if tl.decide == "allow":
        return [Case(t, "any call", RUNS, dict(base))]
    always = tl.decide == "approval"
    cases: List[Case] = []
    kind = tl.kind
    approve = (lambda what, expected, **kw: Case(t, what, expected, dict(base), **kw))
    if kind == "spend":
        if tl.amount_param:
            hi = int(tl.never_above if tl.never_above is not None else L.DEFAULT_NEVER_ABOVE)
            lo = int(tl.allow_up_to if tl.allow_up_to is not None else min(hi, L.DEFAULT_ALLOW_UP_TO))
            band = _band(tool, tl.amount_param, lo, hi, "amount", base)
            if always:  # every call needs approval: the free amount too
                band = [Case(t, f"amount {lo:,}", STOPPED, {**base, tl.amount_param: lo}),
                        Case(t, f"amount {lo:,}", RUNS, {**base, tl.amount_param: lo}, approval=True)] + band[1:]
            cases += band
        else:
            cases += [approve("a call", STOPPED), approve("a call", RUNS, approval=True)]
    elif kind == "shell":
        if lim.profile == "strict":
            cases += [approve("a listed command", RUNS, category={"command_allowlisted": "YES"}, approval=always),
                      approve("a command not on the list", STOPPED, category={"command_allowlisted": "NO"}, approval=True)]
        else:
            from ..actions import COMMAND_CLASSES

            cases.append(approve("an ordinary command", RUNS, category={"command_class": "OK"}, approval=always))
            cases += [approve(COMMAND_WORDS.get(c, c.lower()), STOPPED, category={"command_class": c},
                              approval=True) for c in COMMAND_CLASSES if c != "OK"]
        if always:
            cases.append(approve("an ordinary command", STOPPED))
    elif kind == "sql":
        cases += [approve("a reading query", RUNS, category={"sql_class": "READ"}, approval=always),
                  approve("a writing query", STOPPED, category={"sql_class": "WRITE"}),
                  approve("a writing query", RUNS, category={"sql_class": "WRITE"}, approval=True),
                  approve("a destructive query", STOPPED, category={"sql_class": "DESTRUCTIVE"}, approval=True),
                  approve("an unreadable query", STOPPED, category={"sql_class": "UNREADABLE"}, approval=True)]
        if always:
            cases.append(approve("a reading query", STOPPED, category={"sql_class": "READ"}))
    elif kind == "write":
        cases += [approve("a file inside its folder", RUNS, category={"path_class": "IN_SCOPE"}, approval=always),
                  approve("a file outside its folder", STOPPED, category={"path_class": "OUTSIDE"}, approval=True),
                  approve("a credential or startup file", STOPPED, category={"path_class": "SENSITIVE"}, approval=True),
                  approve("a path that cannot be read", STOPPED, category={"path_class": "UNREADABLE"}, approval=True)]
        if always:
            cases.append(approve("a file inside its folder", STOPPED, category={"path_class": "IN_SCOPE"}))
    elif kind == "read":
        if L._has(tool, L.PATH_PARAMS):
            cases += [approve("a file inside its folder", RUNS, category={"path_class": "IN_SCOPE"}, approval=always),
                      approve("a file elsewhere", RUNS, category={"path_class": "OUTSIDE"}, approval=always),
                      approve("a credential or key", STOPPED, category={"path_class": "SENSITIVE"}, approval=True)]
        elif not tool.params:  # its arguments are not known: anything that names a credential stops
            cases += [approve("an ordinary call", RUNS, category={"path_class": "OUTSIDE"}, approval=always),
                      approve("a call that names a credential or key", STOPPED, category={"path_class": "SENSITIVE"},
                              approval=True)]
        else:
            cases.append(approve("an ordinary call", RUNS, approval=always))
        if always:
            cases.append(approve("an ordinary call", STOPPED, category={"path_class": "IN_SCOPE"}))
    elif kind == "send":
        if lim.profile == "strict" or lim.destinations:
            cases += [approve("to a listed destination", RUNS, category={"destination_allowlisted": "YES"}, approval=always),
                      approve("to another destination", STOPPED, category={"destination_allowlisted": "NO"},
                              approval=True)]
        else:
            cases.append(approve("to anyone", RUNS, approval=always))
        if always:
            cases.append(approve("an ordinary call", STOPPED, category={"destination_allowlisted": "YES"}))
    elif kind in ("publish", "destroy", "identity") or (kind == "other" and (lim.profile == "strict" or always)):
        cases += [approve("a call", STOPPED), approve("a call", RUNS, approval=True)]
    else:
        cases.append(approve("an ordinary call", RUNS))
    for p, (lo, hi) in sorted(numbers.items()):
        for c in _band(tool, p, int(lo), int(hi), p, base):
            if always and not c.approval:
                if c.expected == STOPPED:
                    continue  # every call needs approval: the band below covers it
                c.approval = True
            cases.append(c)
    ordinary = OK_CATEGORY.get(kind, {})  # what the case does not speak about is ordinary
    for c in cases:
        c.category = {**ordinary, **c.category}
    return cases


# ---------------------------------------------------------------------------
# deciding them with the real policy and mapping
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _categories(module, values: Dict[str, str]) -> Iterator[None]:
    """The mapping's classifiers return the case's category values instead of reading arguments."""
    saved = {}
    for var, fn in CLASSIFIERS.items():
        if var in values and hasattr(module, fn):
            saved[fn] = getattr(module, fn)
            setattr(module, fn, (lambda value: lambda *a, **k: value)(values[var]))
    try:
        yield
    finally:
        for fn, f in saved.items():
            setattr(module, fn, f)


def decide(guard, module, case: Case) -> str:
    from ..mapping import MappingError

    try:
        with _categories(module, case.category):
            ctx = module.map_call(case.tool, dict(case.args), {"approval": "YES"} if case.approval else {})
    except MappingError:
        return STOPPED
    except Exception:
        return STOPPED  # the guard fails closed on a mapping that breaks
    try:
        return RUNS if guard.verify(ctx).allowed else STOPPED
    except Exception:
        return STOPPED


def run(ws, agent: Agent, lim: Optional[L.Limits] = None) -> Report:
    from ..runtime import ChimeraGuard, RuntimeConfig
    from .bindings import Bindings
    from .observe import _compile_quiet
    from .policy.draft import agent_key

    key = agent_key(agent)
    report = Report(key)
    lim = lim or L.load(ws, key)
    if lim is None:
        report.note = "no limits set"
        return report
    b = Bindings(ws).get(key)
    if b is None or not b.mapping:
        report.note = "no active policy"
        return report
    text = ws.read(Bindings(ws).abs(b.policy)) or ""
    if "made from its limits" not in text:
        report.note = "the policy is written by hand; check it in cslcore studio"
        return report
    try:
        guard = ChimeraGuard(_compile_quiet(text), RuntimeConfig(raise_on_block=False))
        with contextlib.redirect_stdout(io.StringIO()):
            module = ws.load_module(Bindings(ws).abs(b.mapping))
    except Exception as e:
        report.note = f"the policy or its mapping does not load ({type(e).__name__})"
        return report
    for tool in sorted(L.tools_of(agent, lim), key=lambda x: x.name.lower()):
        tl = lim.tools.get(tool.name) or L.ToolLimit(L.kind_of(tool))
        for case in cases_for(tool, tl, lim):
            case.got = decide(guard, module, case)
            report.cases.append(case)
    return report


# ---------------------------------------------------------------------------
# on screen
# ---------------------------------------------------------------------------

def table(report: Report, show_all: bool = False) -> Table:
    """Tool by tool: what runs and what stops; a row that is not as the limits say is marked."""
    t = Table(box=None, show_header=True, header_style="label", pad_edge=False, padding=(0, 2, 0, 0))
    t.add_column("TOOL", style="head", no_wrap=True)
    t.add_column("RUNS", style="ok")
    t.add_column("STOPPED", style="text")
    by_tool: Dict[str, List[Case]] = {}
    for c in report.cases:
        by_tool.setdefault(c.tool, []).append(c)
    for tool, cases in by_tool.items():
        def cell(expected: str) -> Text:
            out = Text()
            for c in (x for x in cases if x.expected == expected):
                if out:
                    out.append("\n")
                what = c.what + (", with approval" if expected == RUNS and c.approval else
                                 ", without approval" if expected == STOPPED and not c.approval else "")
                if c.ok:
                    out.append(what, style="ok" if expected == RUNS else "text")
                else:
                    out.append(f"✗ {what}: {c.got}", style="high")
            return out
        t.add_row(tool, cell(RUNS), cell(STOPPED))
    return t


def show(console, report: Report, name: str = "", compact: bool = False) -> None:
    """The check for one agent: a line, and the table (compact: the table only when a call is wrong)."""
    title = name or report.agent
    if not report.cases:
        console.print(Text.assemble(("  check ", "muted"), (title, "head"), (f"   {report.note}", "muted")))
        return
    runs, stops, bad = report.counts()
    head = (Text("  ✓ ", style="ok") if not bad else Text("  ✗ ", style="high"))
    head.append_text(Text.assemble((title, "head"),
                                   (f"   {runs} sample calls run, {stops} stop, as its limits say" if not bad
                                    else f"   {bad} of {len(report.cases)} sample calls not as its limits say", "muted" if not bad else "high")))
    console.print(head)
    if compact and not bad:
        return
    console.print(Padding(table(report), (0, 0, 0, 4)))
    console.print(Text("    decided by the active policy and mapping in block mode; nothing ran\n"
                       "    a stopped call stays stopped with an approval, unless it says \"without approval\"", style="muted"))
