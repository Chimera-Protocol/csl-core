"""
Z3 and TLA+ runs for the studio, returned as plain data the animations and the suggestion
panel can use. The engines themselves are called as they are (frozen for 0.6); nothing here
changes how they decide.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Z3Issue:
    kind: str  # CONTRADICTION | UNREACHABLE | UNSUPPORTED | INTERNAL_ERROR | PARSE_ERROR | VALIDATION_ERROR
    message: str
    rules: List[str] = field(default_factory=list)
    model: Optional[Dict[str, Any]] = None
    line: Optional[int] = None


@dataclass
class Z3Run:
    ok: bool
    rules: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    issues: List[Z3Issue] = field(default_factory=list)
    warnings: List[Z3Issue] = field(default_factory=list)
    unreachable: List[str] = field(default_factory=list)
    conflicts: List[Tuple[str, str]] = field(default_factory=list)
    elapsed_ms: int = 0
    stage: str = "ok"  # parse | validate | verify | ok


@dataclass
class TLAConstraint:
    name: str
    status: str  # HOLDS | VIOLATED | UNKNOWN
    states: int
    time_ms: int
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TLASuggestion:
    constraint: str
    title: str
    explanation: str
    confidence: str
    before: Optional[str]
    after: Optional[str]


@dataclass
class RuleGuard:
    """What one rule does at runtime, measured with the compiled guard over a grid of states."""
    rule: str
    blocked: int = 0  # states of the grid this rule blocks
    example: Optional[Dict[str, Any]] = None  # one blocked call

    def share(self, checked: int) -> float:
        return self.blocked / checked if checked else 0.0


@dataclass
class TLARun:
    ok: bool
    engine: str  # TLC | BFS
    engine_note: str = ""
    variables: List[Dict[str, str]] = field(default_factory=list)  # name, domain, card
    state_space: str = "?"
    constraints: List[TLAConstraint] = field(default_factory=list)
    suggestions: List[TLASuggestion] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: int = 0
    guard: Dict[str, RuleGuard] = field(default_factory=dict)  # per rule, see guard_view
    strict: bool = False  # ENABLE_FORMAL_VERIFICATION: TRUE, the compiler then needs every rule to hold
    checked: int = 0  # grid states pushed through the guard
    blocked: int = 0  # of those, blocked by at least one rule
    grid_cut: bool = False  # the grid stopped at its time budget (a sample, not every state)

    @property
    def total_states(self) -> int:
        return max((c.states for c in self.constraints), default=0)

    def status_of(self, c: "TLAConstraint") -> str:
        """Guard reading of a TLA+ result: a rule that reachable states break is a rule that blocks.

        ENFORCED     the rule blocks some states (the checker found one, or the guard grid did)
        NEVER FIRES  neither the checker nor the guard grid found a state it blocks
        UNKNOWN      the checker could not decide and the grid did not run

        The grid runs the compiled guard itself over every enum value and every threshold edge,
        so it settles a rule the fallback model checker (which explores a bounded sample) missed."""
        g = self.guard.get(c.name)
        if c.status == "VIOLATED" or (g is not None and g.blocked):
            return "ENFORCED"
        if c.status == "HOLDS" or (g is not None and self.checked):
            return "NEVER FIRES"
        return "UNKNOWN"

    @property
    def enforced(self) -> List[str]:
        return [c.name for c in self.constraints if self.status_of(c) == "ENFORCED"]

    @property
    def never_fires(self) -> List[str]:
        return [c.name for c in self.constraints if self.status_of(c) == "NEVER FIRES"]

    @property
    def refuses_to_load(self) -> bool:
        """The compiler rejects a strict policy whose rules reachable states break."""
        return self.strict and bool(self.enforced)



def _line_of(err: Exception) -> Optional[int]:
    for attr in ("line", "lineno"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    loc = getattr(err, "location", None)
    if isinstance(loc, (tuple, list)) and loc and isinstance(loc[0], int):
        return loc[0]
    return None


def run_z3(text: str) -> Z3Run:
    from ...language.parser import parse_csl
    from ...language.validator import CSLValidator

    t0 = time.perf_counter()
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            ast = parse_csl(text)
    except Exception as e:
        return Z3Run(False, issues=[Z3Issue("PARSE_ERROR", str(e).splitlines()[0][:300] if str(e) else type(e).__name__, line=_line_of(e))],
                     stage="parse", elapsed_ms=int((time.perf_counter() - t0) * 1000))
    rules = [c.name for c in ast.constraints or []]
    variables = {d.name: d.domain for d in (ast.domain.variable_declarations if ast.domain else [])}
    run = Z3Run(True, rules=rules, variables=variables, pairs=list(itertools.combinations(rules, 2)))
    try:
        with contextlib.redirect_stdout(sink):
            CSLValidator().validate(ast)
    except Exception as e:
        run.ok, run.stage = False, "validate"
        run.issues.append(Z3Issue("VALIDATION_ERROR", str(e).splitlines()[0][:300] if str(e) else type(e).__name__, line=_line_of(e)))
        run.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return run
    try:
        from ...engines.z3_engine.verifier import LogicVerifier

        with contextlib.redirect_stdout(sink):
            _ok, raw = LogicVerifier().verify(ast)
    except Exception as e:
        run.ok, run.stage = False, "verify"
        run.issues.append(Z3Issue("INTERNAL_ERROR", f"{type(e).__name__}: {e}"[:300]))
        run.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return run
    for it in raw or []:
        kind = it.get("kind", "ERROR")
        if kind == "COVERAGE":
            continue
        rules_hit = list(it.get("rules") or [])
        if kind == "UNREACHABLE" and "mutually exclusive" in str(it.get("message", "")):
            continue  # a note: rules on different tools share an action variable, which is normal
        if kind == "UNREACHABLE":  # a warning: the policy compiles, but the rule can never trigger
            run.unreachable += rules_hit
            run.warnings.append(Z3Issue(kind, str(it.get("message", "")), rules_hit, it.get("model")))
            continue
        if it.get("severity") not in (None, "error"):
            continue
        run.issues.append(Z3Issue(kind, str(it.get("message", "")), rules_hit, it.get("model")))
        if kind == "CONTRADICTION" and len(rules_hit) >= 2:
            run.conflicts.append((rules_hit[0], rules_hit[1]))
    run.ok = not run.issues
    run.stage = "ok" if run.ok else "verify"
    run.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return run


def run_tla(text: str, use_real_tlc: bool = True, timeout: int = 60, max_states: int = 5000) -> TLARun:
    from ...language.parser import parse_csl
    from ...language.validator import CSLValidator
    from ...engines.tla_engine.tla_spec_builder import TLASpecBuilder
    from ...engines.tla_engine.tlc_runner import TLCRunner
    from ...engines.tla_engine.verifier import TLAVerifier, _normalize_cex

    t0 = time.perf_counter()
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            ast = parse_csl(text)
            CSLValidator().validate(ast)
    except Exception as e:
        return TLARun(False, "BFS", error=str(e).splitlines()[0][:300] if str(e) else type(e).__name__)
    constraints = ast.constraints or []
    if not constraints:
        return TLARun(False, "BFS", error="the policy has no rules to check")
    v = TLAVerifier(max_states=max_states, animate=False, use_real_tlc=use_real_tlc, tlc_timeout=timeout,
                    tlc_auto_download=False)
    runner = TLCRunner(auto_download=False)
    real = use_real_tlc and runner.is_available()
    note = "real TLC (java -jar tla2tools.jar)" if real else (
        "Python model checker (TLC not found; set TLA2TOOLS_JAR or install Java for real TLC)" if use_real_tlc
        else "Python model checker")
    try:
        with contextlib.redirect_stdout(sink):
            spec = TLASpecBuilder().build(ast)
            var_info = v._spec_var_info(spec)
            if real:
                results, raw = v._run_tlc(spec, constraints, "TLC")
                if raw is not None and not raw.success and not raw.violations:
                    # TLC could not check this spec (for example decimals: "TLC can't handle real
                    # numbers"); the built-in checker does, and the screen says why it was used
                    reason = _tlc_reason(getattr(raw, "tlc_output", "") or "") or "TLC did not complete"
                    real = False
                    note = f"built-in model checker ({reason})"
                    results = v._run_mock(ast, constraints, "MOCK")
            else:
                results = v._run_mock(ast, constraints, "MOCK")
    except Exception as e:
        return TLARun(False, "TLC" if real else "BFS", note, error=f"{type(e).__name__}: {e}"[:300],
                      elapsed_ms=int((time.perf_counter() - t0) * 1000))
    run = TLARun(True, "TLC" if real else "BFS", note, var_info, _space(var_info))
    for r in results:
        trace = _normalize_cex(r.counterexample or []) if r.counterexample else []
        run.constraints.append(TLAConstraint(r.name, r.status, int(r.states_checked or 0), int(r.time_ms or 0), trace))
    # The TLA+ suggestion engine aims at making every rule hold over the whole domain (narrowing
    # domains). For a guard a broken rule is the guard doing its job, so it is not used here.
    run.strict = bool(STRICT.search(re.sub(r"//[^\n]*", "", text)))
    try:
        guard_view(text, ast, run)
    except Exception:
        pass
    run.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return run


GRID_LIMIT = 20000
GRID_FULL = 4000
GRID_BUDGET_S = 0.5  # the grid never holds the screen up for longer than this
STRICT = re.compile(r"ENABLE_FORMAL_VERIFICATION\s*:\s*TRUE")


def _grid_values(domain: str, numbers: List[int]) -> Optional[List[Any]]:
    d = str(domain).strip()
    if d.startswith("{"):
        return re.findall(r'"([^"]*)"', d) or None
    m = re.fullmatch(r"(-?\d+)\s*\.\.\s*(-?\d+)", d)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if hi - lo <= 24:
        return list(range(lo, hi + 1))
    vals = {lo, hi, (lo + hi) // 2}
    for n in numbers:  # the thresholds the rules use, and their neighbours
        vals.update(v for v in (n - 1, n, n + 1) if lo <= v <= hi)
    return sorted(vals)


def guard_view(text: str, ast, run: TLARun) -> None:
    """Push a grid of states through the compiled guard and record, per rule, how many it blocks.

    The grid takes every enum value and, for numeric ranges, the bounds plus every threshold the
    rules mention (and its neighbours), so each rule's edges are inside it. Above GRID_LIMIT states
    a fixed-seed sample is used."""
    import random

    from ...runtime import ChimeraGuard, RuntimeConfig
    from ..observe import _compile_quiet

    body = text[text.find("STATE_CONSTRAINT"):] if "STATE_CONSTRAINT" in text else text
    numbers = sorted({int(n) for n in re.findall(r"(?<![\w.])-?\d+(?![\w.])", re.sub(r"//[^\n]*", "", body))})
    names, axes = [], []
    for d in ast.domain.variable_declarations if ast.domain else []:
        vals = _grid_values(d.domain, numbers)
        if vals is None:
            return
        names.append(d.name)
        axes.append(vals)
    if not names:
        return
    # measured as the guard runs; the compile-time TLA+ requirement (see TLARun.strict) is reported apart
    loose = STRICT.sub("ENABLE_FORMAL_VERIFICATION: FALSE", text)
    guard = ChimeraGuard(_compile_quiet(loose), RuntimeConfig(raise_on_block=False))
    total = 1
    for a in axes:
        total *= len(a)
    if total <= GRID_FULL:  # small enough to check every state within the budget
        states = itertools.product(*axes)
    else:  # a fixed-seed random sample, so stopping at the budget does not bias it
        rng = random.Random(7)
        states = (tuple(rng.choice(a) for a in axes) for _ in range(GRID_LIMIT))
    run.guard = {c.name: RuleGuard(c.name) for c in run.constraints}
    deadline = time.perf_counter() + GRID_BUDGET_S
    for i, combo in enumerate(states):
        if i % 128 == 0:
            time.sleep(0.0005)  # a real pause: sleep(0) hands the lock straight back to this thread
            if time.perf_counter() > deadline:
                run.grid_cut = True
                break
        r = guard.verify(dict(zip(names, combo)))
        run.checked += 1
        if r.allowed:
            continue
        run.blocked += 1
        for rule in r.violated_rule_ids:
            g = run.guard.setdefault(rule, RuleGuard(rule))
            g.blocked += 1
            if g.example is None:
                g.example = dict(zip(names, combo))


def _tlc_reason(output: str) -> Optional[str]:
    """TLC's own one-line reason from its tool-mode output (the message body of an error block)."""
    lines = output.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"@!@!@STARTMSG (\d+):1 @!@!@", line)
        if m:
            body = [l.strip() for l in lines[i + 1:i + 4] if l.strip() and not l.startswith("@!@!@")]
            if body:
                return body[0].rstrip(".")[:80]
    return None


def card_of(vi: Dict[str, str]) -> Optional[int]:
    """The (abstracted) domain size the checker explores: the first number of the card label."""
    m = re.search(r"\d[\d,]*", str(vi.get("card", "")))
    return int(m.group(0).replace(",", "")) if m else None


def _space(var_info: List[Dict[str, str]]) -> str:
    total = 1
    for vi in var_info:
        n = card_of(vi)
        if n is None:
            return "∞"
        total *= n
    return f"{total:,}"
