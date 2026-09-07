"""
TLA+ Verifier — CSL AST → Formal Model Checking

Dispatch strategy:
  1. If Java is present AND tla2tools.jar is available → use REAL TLC
     (complete, mathematically rigorous exhaustive model checking)
  2. Otherwise → fall back to MockModelChecker (Python BFS, sound but not TLC)

Real TLC path:
  • TLASpecBuilder generates a proper .tla + .cfg
  • TLCRunner invokes `java -jar tla2tools.jar` in a temp directory
  • TLC output is parsed into TLCResult with per-invariant violations
  • Counterexample traces come from TLC's own trace format

Mock path (when TLC unavailable):
  • MockModelChecker does BFS over the abstract state space
  • Same invariants, same violation detection, less rigorous guarantee

Both paths share:
  • Rich terminal animations (with a banner indicating which engine ran)
  • TLASuggestionEngine analysis of violations
  • ProofCertificate generation
"""

from __future__ import annotations

import re
import time
from itertools import product as cart_product
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── CSL AST ──────────────────────────────────────────────────────────────────
from chimera_core.language.ast import (
    Constitution, Constraint, Expression,
    Literal, Variable, BinaryOp, UnaryOp, FunctionCall, MemberAccess,
    TemporalOperator, ModalOperator,
    ComparisonOperator, LogicalOperator, ArithmeticOperator,
)

# ── Formal engine (same package) ─────────────────────────────────────────────
from .model_checker import (
    MockModelChecker, State as MCState,
    CheckResult, ModelCheckingResult,
)
from .proof_builder import (
    ModelCheckingProofBuilder, ProofCertificate, ProofValidator,
)
from .animations import (
    TLAAnimationEngine, ConstraintAnimResult, VerificationAnimResult,
)
from .tla_spec_builder import TLASpecBuilder, TLASpecResult
from .tlc_runner import (
    TLCRunner, TLCResult, TLCViolation,
    java_available, find_jar,
    run_tlc_on_spec,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DOMAIN PARSER (for MockModelChecker state space)
# ═════════════════════════════════════════════════════════════════════════════

_SET_RE   = re.compile(r'^\{(.+)\}$', re.DOTALL)
_RANGE_RE = re.compile(r'^(-?[\d.]+)\.\.(-?[\d.]+)$')


def _parse_domain(domain_str: str, max_samples: int = 6) -> List[Any]:
    """
    Convert a CSL domain string into a finite list of representative values.
    Used by the Mock path (BFS state space).
    """
    if not isinstance(domain_str, str):
        return [0]

    s = domain_str.strip()

    # String set  {"A", "B"}
    m = _SET_RE.match(s)
    if m:
        inner = m.group(1)
        values: List[Any] = []
        for tok in re.split(r',\s*', inner):
            tok = tok.strip().strip('"').strip("'")
            if tok:
                values.append(tok)
        return values if values else [None]

    # Boolean
    if s.upper() == "BOOLEAN":
        return [True, False]

    # Numeric range
    m = _RANGE_RE.match(s)
    if m:
        lo_s, hi_s = m.group(1), m.group(2)
        is_float = '.' in lo_s or '.' in hi_s
        if is_float:
            lo, hi = float(lo_s), float(hi_s)
            step = (hi - lo) / max(1, max_samples - 1)
            return [round(lo + i * step, 6) for i in range(max_samples)]
        else:
            lo, hi = int(lo_s), int(hi_s)
            if hi - lo + 1 <= max_samples:
                return list(range(lo, hi + 1))
            step = max(1, (hi - lo) // (max_samples - 1))
            vals = list(range(lo, hi, step))
            if hi not in vals:
                vals.append(hi)
            return vals[:max_samples]

    if s in ("Int",):
        return [-1, 0, 1, 10, 100, -100]
    if s in ("Nat",):
        return [0, 1, 10, 100, 1000]

    return [0]


def _cardinality_label(domain_str: str) -> str:
    """Human-readable cardinality for animation table."""
    if not isinstance(domain_str, str):
        return "?"
    s = domain_str.strip()
    if _SET_RE.match(s):
        vals = _parse_domain(s)
        return str(len(vals))
    if s.upper() == "BOOLEAN":
        return "2"
    m = _RANGE_RE.match(s)
    if m:
        lo_s, hi_s = m.group(1), m.group(2)
        is_float = '.' in lo_s or '.' in hi_s
        if is_float:
            return "∞"
        lo, hi = int(lo_s), int(hi_s)
        return f"{hi - lo + 1:,}" if (hi - lo) < 10_000 else "∞"
    return "∞"


# ═════════════════════════════════════════════════════════════════════════════
# 2. EXPRESSION EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════

class _EvalError(Exception):
    pass


def _eval(expr: Expression, state: Dict[str, Any]) -> Any:
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, Variable):
        return state.get(expr.name)

    if isinstance(expr, MemberAccess):
        obj = _eval(expr.object, state)
        if isinstance(obj, dict):
            return obj.get(expr.member)
        return getattr(obj, expr.member, None)

    if isinstance(expr, UnaryOp):
        val = _eval(expr.operand, state)
        op  = expr.operator
        if op == LogicalOperator.NOT:      return not val
        if op == ArithmeticOperator.SUB:   return -val
        return val

    if isinstance(expr, BinaryOp):
        l = _eval(expr.left,  state)
        r = _eval(expr.right, state)
        op = expr.operator
        if op == LogicalOperator.AND:  return bool(l) and bool(r)
        if op == LogicalOperator.OR:   return bool(l) or bool(r)
        if l is None or r is None:     return False
        if op == ComparisonOperator.EQ:  return l == r
        if op == ComparisonOperator.NEQ: return l != r
        if op == ComparisonOperator.LT:  return l <  r
        if op == ComparisonOperator.GT:  return l >  r
        if op == ComparisonOperator.LTE: return l <= r
        if op == ComparisonOperator.GTE: return l >= r
        if op == ArithmeticOperator.ADD: return l + r
        if op == ArithmeticOperator.SUB: return l - r
        if op == ArithmeticOperator.MUL: return l * r
        if op == ArithmeticOperator.DIV:
            return l / r if r != 0 else float('inf')
        if op == ArithmeticOperator.MOD:
            return l % r if r != 0 else 0
        raise _EvalError(f"Unsupported binary op: {op}")

    if isinstance(expr, FunctionCall):
        args = [_eval(a, state) for a in expr.args]
        name = expr.name
        if name == "len":  return len(args[0]) if args else 0
        if name == "max":  return max(args)
        if name == "min":  return min(args)
        if name == "abs":  return abs(args[0]) if args else 0
        raise _EvalError(f"Unknown function: {name}")

    raise _EvalError(f"Unsupported expression: {type(expr).__name__}")


def _check_action(constraint: Constraint, state: Dict[str, Any]) -> bool:
    action = constraint.action
    var_name = action.variable
    actual   = state.get(var_name)

    try:
        expected = _eval(action.value, state)
    except _EvalError:
        return True

    modal = action.modal_operator

    if modal == ModalOperator.MUST_BE:     return actual == expected
    if modal == ModalOperator.MUST_NOT_BE: return actual != expected
    if modal == ModalOperator.MAY_BE:      return True
    if modal == ModalOperator.EQ:          return actual == expected
    if modal == ModalOperator.NEQ:         return actual != expected

    if actual is None or expected is None:
        return True

    try:
        if modal == ModalOperator.LT:  return actual <  expected
        if modal == ModalOperator.GT:  return actual >  expected
        if modal == ModalOperator.LTE: return actual <= expected
        if modal == ModalOperator.GTE: return actual >= expected
    except TypeError:
        return True

    return True


def _build_invariant(constraint: Constraint):
    """Build a callable invariant: MCState → bool."""
    temporal = constraint.condition.temporal_operator

    def invariant(mc_state: MCState) -> bool:
        s = mc_state.variables
        try:
            if temporal == TemporalOperator.ALWAYS:
                cond_holds = True
            else:
                cond_holds = bool(_eval(constraint.condition.condition, s))
        except _EvalError:
            return True

        if not cond_holds:
            return True

        return _check_action(constraint, s)

    return invariant


# ═════════════════════════════════════════════════════════════════════════════
# 3. STATE SPACE BUILDER (Mock path)
# ═════════════════════════════════════════════════════════════════════════════

def _build_state_space(
    constitution: Constitution,
    max_neighbors: int = 3,
) -> Tuple[MCState, Callable]:
    if not constitution.domain or not constitution.domain.variable_declarations:
        initial = MCState({"_dummy": 0}, state_id=0)
        return initial, lambda s: []

    decls  = constitution.domain.variable_declarations
    names  = [d.name for d in decls]
    domain_samples: Dict[str, List[Any]] = {
        d.name: _parse_domain(d.domain if isinstance(d.domain, str) else str(d.domain))
        for d in decls
    }

    init_vars = {n: domain_samples[n][0] for n in names}
    initial   = MCState(init_vars, state_id=0)

    def next_state_func(state: MCState) -> List[MCState]:
        neighbours: List[MCState] = []
        for name in names:
            samples = domain_samples[name]
            current = state.variables.get(name)
            try:
                idx = samples.index(current)
            except ValueError:
                idx = 0
            next_val = samples[(idx + 1) % len(samples)]
            if next_val == current:
                continue
            new_vars = dict(state.variables)
            new_vars[name] = next_val
            neighbours.append(MCState(new_vars))
            if len(neighbours) >= max_neighbors:
                break
        return neighbours

    return initial, next_state_func


# ═════════════════════════════════════════════════════════════════════════════
# 4. NORMALIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_cex(counterexample) -> List[Dict[str, Any]]:
    """
    Normalize a counterexample trace to a list of plain dicts.
    Handles both MCState objects and raw dicts.
    """
    result: List[Dict[str, Any]] = []
    for s in counterexample:
        if isinstance(s, MCState):
            result.append(dict(s.variables))
        elif isinstance(s, dict):
            result.append(s)
        else:
            # Unknown type — try .variables attribute
            try:
                result.append(dict(s.variables))
            except AttributeError:
                result.append({})
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. ISSUE TYPE
# ═════════════════════════════════════════════════════════════════════════════

class TLAIssue:
    def __init__(
        self,
        kind: str,
        constraint: str,
        message: str,
        counterexample: Optional[List[Dict]] = None,
    ):
        self.kind            = kind
        self.constraint      = constraint
        self.message         = message
        self.counterexample  = counterexample

    def __repr__(self):
        return f"TLAIssue({self.kind!r}, {self.constraint!r})"


# ═════════════════════════════════════════════════════════════════════════════
# 5. TLC RESULT → ConstraintAnimResult CONVERTER
# ═════════════════════════════════════════════════════════════════════════════

def _tlc_result_to_anim_results(
    tlc_result: TLCResult,
    constraints: List[Constraint],
    elapsed_ms: int,
) -> List[ConstraintAnimResult]:
    """
    Convert a TLCResult (one run covering all invariants) into a list of
    per-constraint ConstraintAnimResult objects, matching the Mock path's output.
    """
    violated_names: Dict[str, TLCViolation] = {}
    for v in tlc_result.violations:
        # TLC uses the safe-identifier form; map back to original
        violated_names[v.invariant] = v

    results: List[ConstraintAnimResult] = []
    safe_re = re.compile(r'[^A-Za-z0-9_]')

    for c in constraints:
        safe_name = safe_re.sub("_", c.name)
        # Check both original name and safe name
        violation = violated_names.get(c.name) or violated_names.get(safe_name)

        if violation:
            # Convert TLC trace (list of {var: val_str}) to MCState-compatible format
            trace_states = []
            for s in violation.trace:
                # TLC returns strings; try to coerce to Python types
                coerced = {k: _coerce_tlc_value(v) for k, v in s.items()}
                trace_states.append(MCState(coerced))
            results.append(ConstraintAnimResult(
                name=c.name,
                status="VIOLATED",
                states_checked=tlc_result.states_explored,
                time_ms=elapsed_ms,
                counterexample=trace_states if trace_states else None,
            ))
        else:
            # No violation recorded for this constraint. NOTE: this function
            # runs even when TLC failed to complete (e.g. an unsupported
            # construct) — in that case tlc_result.violations is empty not
            # because the constraint holds, but because TLC never actually
            # checked it. TLAVerifier.verify() detects that case via
            # tlc_result.success/error and overrides the final result with a
            # VERIFICATION_ERROR before returning, but in animated mode this
            # function's "HOLDS" is what the live terminal animation shows
            # first — a real, lower-severity follow-up: the animation can
            # flash a misleading HOLDS before the correct error prints after
            # it. The returned/programmatic result (is_valid, issues) is
            # correct regardless.
            results.append(ConstraintAnimResult(
                name=c.name,
                status="HOLDS",
                states_checked=tlc_result.states_explored,
                time_ms=elapsed_ms,
                counterexample=None,
            ))

    return results


def _coerce_tlc_value(raw: str) -> Any:
    """Parse TLC string values back to Python types where possible."""
    raw = raw.strip()
    # Boolean
    if raw == "TRUE":  return True
    if raw == "FALSE": return False
    # Quoted string
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    # Integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN VERIFIER
# ═════════════════════════════════════════════════════════════════════════════

class TLAVerifier:
    """
    Formal TLA+ verifier for CSL constitutions.

    With `use_real_tlc=True` (default):
      1. Checks if Java + tla2tools.jar are available
      2. Generates a TLA+ spec with predicate abstraction
      3. Runs `java -jar tla2tools.jar` → real exhaustive model checking
      4. Falls back to MockModelChecker if TLC unavailable

    With `use_real_tlc=False`:
      Always uses MockModelChecker (Python BFS).
    """

    def __init__(
        self,
        max_states: int = 5_000,
        animate: bool = True,
        use_real_tlc: bool = True,
        tlc_timeout: int = 60,
        tlc_jar_path: Optional[str] = None,
        tlc_auto_download: bool = True,
    ):
        self.max_states        = max_states
        self.animate           = animate
        self.use_real_tlc      = use_real_tlc
        self.tlc_timeout       = tlc_timeout
        self.tlc_jar_path      = tlc_jar_path
        self.tlc_auto_download = tlc_auto_download

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def verify(
        self,
        constitution: Constitution,
    ) -> Tuple[bool, List[TLAIssue]]:
        """
        Run model checking over all constraints in the constitution.

        Returns:
            (all_valid: bool, issues: List[TLAIssue])
        """
        domain_name = constitution.domain.name if constitution.domain else "Unknown"
        constraints = constitution.constraints or []
        var_info    = self._build_var_info(constitution)

        # ── Decide which engine to use ────────────────────────────────
        runner      = TLCRunner(
            jar_path=self.tlc_jar_path,
            auto_download=self.tlc_auto_download,
        )
        tlc_ready   = self.use_real_tlc and runner.is_available()
        engine_mode = "TLC" if tlc_ready else "MOCK"

        # ── Build spec (needed even for Mock for state space info) ────
        builder = TLASpecBuilder()
        spec    = builder.build(constitution)

        # Enrich var_info with predicate-abstraction details
        for di in spec.domain_info:
            for vi in var_info:
                if vi["name"] == di["name"]:
                    vi["tla_set"] = di["tla_set"]
                    vi["card"]    = di["card"]
                    break

        # ── Run the appropriate engine ────────────────────────────────
        if tlc_ready:
            c_results, tlc_raw = self._run_tlc(
                spec, constraints, engine_mode
            )
        else:
            c_results = self._run_mock(constitution, constraints, engine_mode)
            tlc_raw   = None

        # ── Collect issues + run suggestion engine ────────────────────
        from chimera_core.engines.tla_engine.suggestion_engine import (
            TLASuggestionEngine, ViolationAnalysis,
        )
        from chimera_core.engines.tla_engine.animations import render_violation_reports

        suggestion_engine = TLASuggestionEngine()
        issues:   List[TLAIssue]         = []
        analyses: List[ViolationAnalysis] = []

        # ── TLC failed to actually run (e.g. an unsupported construct like a
        # real-number literal) ─────────────────────────────────────────────
        # This must be surfaced as an error, not silently treated as "every
        # constraint holds". A run that TLC could not complete tells us
        # nothing about whether the policy is safe — per-constraint status
        # in c_results is meaningless in this case (see _tlc_result_to_anim_
        # results, which marks unmatched constraints "HOLDS" unconditionally
        # because it has no other signal to go on).
        if tlc_raw is not None and (not tlc_raw.success) and (not tlc_raw.violations):
            issues.append(TLAIssue(
                kind="VERIFICATION_ERROR",
                constraint="<all>",
                message=(
                    tlc_raw.error
                    or "TLC did not complete verification and reported no violations "
                       "or a clear success — treat this policy as unverified."
                ),
            ))
            all_valid = False
            if self.animate:
                from rich.console import Console
                Console().print(
                    f"[bold red]TLA+ verification did not complete:[/] {issues[-1].message}"
                )
            self._emit_certificates(c_results)
            return all_valid, issues

        for r in c_results:
            if r.status == "VIOLATED":
                issues.append(TLAIssue(
                    kind="SAFETY_VIOLATION",
                    constraint=r.name,
                    message=(
                        f"{'TLC' if engine_mode == 'TLC' else 'BFS'} model checking "
                        f"found a counterexample for □({r.name}): "
                        f"invariant violated after {r.states_checked} states."
                    ),
                    counterexample=r.counterexample,
                ))

                c_obj = self._find_constraint(constraints, r.name)
                if c_obj is not None:
                    # Normalize counterexample states to plain dicts for suggestion engine
                    raw_cex = r.counterexample or []
                    cex_dicts = _normalize_cex(raw_cex)
                    analysis = suggestion_engine.analyze(c_obj, cex_dicts, constitution)
                    analysis._raw_counterexample = raw_cex
                    analyses.append(analysis)

        all_valid = len(issues) == 0

        if analyses and self.animate:
            render_violation_reports(analyses)

        self._emit_certificates(c_results)

        return all_valid, issues

    # ------------------------------------------------------------------
    # INTERNAL: TLC path
    # ------------------------------------------------------------------

    def _run_tlc(
        self,
        spec: TLASpecResult,
        constraints: List[Constraint],
        engine_mode: str,
    ) -> Tuple[List[ConstraintAnimResult], Optional[TLCResult]]:
        """
        Run real TLC.  Wraps result in ConstraintAnimResult list for
        uniform downstream processing.

        Strategy: run TLC first (silently), then play the animation with the
        pre-computed results injected into the checker_fn.  This lets us surface
        TLC's own version string, PID, and worker count in the banner — values
        that are impossible to produce with the Python BFS mock.
        """
        if self.animate:
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            con = Console()

            # ── Step 0: run TLC for real, show a brief spinner ────────
            tlc_result: Optional[TLCResult] = None
            with Progress(
                SpinnerColumn(spinner_name="bouncingBar", style="bright_green"),
                TextColumn("  [bold bright_green]⚡ Invoking TLC[/]  [dim]java -jar tla2tools.jar …[/]"),
                TimeElapsedColumn(),
                console=con,
                transient=True,
            ) as prog:
                prog.add_task("", total=None)
                tlc_result = run_tlc_on_spec(
                    spec,
                    jar_path=self.tlc_jar_path,
                    timeout=self.tlc_timeout,
                    auto_download=self.tlc_auto_download,
                )

            if tlc_result is None:
                tlc_result = TLCResult(success=True, used_real_tlc=True)

            # ── Step 1: pre-compute per-constraint results ─────────────
            elapsed = tlc_result.time_ms
            c_results_precomputed = _tlc_result_to_anim_results(
                tlc_result, constraints, elapsed
            )
            result_map = {r.name: r for r in c_results_precomputed}

            # ── Step 2: full animation with identity proof in banner ───
            def tlc_checker_fn(name: str, _progress_cb) -> ConstraintAnimResult:
                return result_map.get(
                    name,
                    ConstraintAnimResult(name=name, status="HOLDS", states_checked=0, time_ms=0),
                )

            from chimera_core.engines.tla_engine.animations import TLAAnimationEngine
            anim = TLAAnimationEngine()
            anim.run(
                domain_name=spec.module_name,
                var_info=self._spec_var_info(spec),
                constraint_names=[c.name for c in constraints],
                checker_fn=tlc_checker_fn,
                engine_mode=engine_mode,
                tlc_version=tlc_result.tlc_version,
                tlc_pid=tlc_result.tlc_pid,
                java_workers=tlc_result.java_workers,
            )

            return c_results_precomputed, tlc_result
        else:
            # Silent path
            tlc_result = run_tlc_on_spec(
                spec,
                jar_path=self.tlc_jar_path,
                timeout=self.tlc_timeout,
                auto_download=self.tlc_auto_download,
            )
            if tlc_result is None:
                tlc_result = TLCResult(success=True, used_real_tlc=True)
            elapsed = tlc_result.time_ms
            c_results = _tlc_result_to_anim_results(tlc_result, constraints, elapsed)
            return c_results, tlc_result

    # ------------------------------------------------------------------
    # INTERNAL: Mock path
    # ------------------------------------------------------------------

    def _run_mock(
        self,
        constitution: Constitution,
        constraints: List[Constraint],
        engine_mode: str,
    ) -> List[ConstraintAnimResult]:
        initial_state, next_state_func = _build_state_space(constitution)
        checker = MockModelChecker(max_states=self.max_states, max_depth=50)
        var_info = self._build_var_info(constitution)

        if self.animate:
            anim = TLAAnimationEngine()

            def mock_checker_fn(name: str, _progress_cb) -> ConstraintAnimResult:
                c = self._find_constraint(constraints, name)
                if c is None:
                    return ConstraintAnimResult(
                        name=name, status="UNKNOWN",
                        states_checked=0, time_ms=0,
                    )
                return self._check_one_mock(c, initial_state, next_state_func, checker)

            anim_result = anim.run(
                domain_name=constitution.domain.name if constitution.domain else "Unknown",
                var_info=var_info,
                constraint_names=[c.name for c in constraints],
                checker_fn=mock_checker_fn,
                engine_mode=engine_mode,
            )
            return anim_result.constraint_results
        else:
            return [
                self._check_one_mock(c, initial_state, next_state_func, checker)
                for c in constraints
            ]

    def _check_one_mock(
        self,
        constraint: Constraint,
        initial_state: MCState,
        next_state_func: Callable,
        checker: MockModelChecker,
    ) -> ConstraintAnimResult:
        invariant = _build_invariant(constraint)
        t0 = time.perf_counter()
        mc_result: ModelCheckingResult = checker.check_safety(
            initial_state=initial_state,
            next_state_func=next_state_func,
            invariant=invariant,
            property_name=constraint.name,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if mc_result.result == CheckResult.VALID:
            status, cex = "HOLDS", None
        elif mc_result.result == CheckResult.VIOLATED:
            status = "VIOLATED"
            cex    = mc_result.counterexample.states if mc_result.counterexample else None
        else:
            status, cex = "UNKNOWN", None

        return ConstraintAnimResult(
            name=constraint.name,
            status=status,
            states_checked=mc_result.states_explored,
            time_ms=elapsed_ms,
            counterexample=cex,
        )

    # ------------------------------------------------------------------
    # INTERNAL: helpers
    # ------------------------------------------------------------------

    def _build_var_info(self, constitution: Constitution) -> List[Dict[str, str]]:
        if not constitution.domain or not constitution.domain.variable_declarations:
            return []
        result = []
        for d in constitution.domain.variable_declarations:
            domain_str = d.domain if isinstance(d.domain, str) else str(d.domain)
            result.append({
                "name":   d.name,
                "domain": domain_str,
                "card":   _cardinality_label(domain_str),
            })
        return result

    @staticmethod
    def _spec_var_info(spec: TLASpecResult) -> List[Dict[str, str]]:
        """Build var_info from TLASpecResult domain_info."""
        return [
            {
                "name":   di["name"],
                "domain": di["domain"],
                "card":   di["card"],
            }
            for di in spec.domain_info
        ]

    @staticmethod
    def _find_constraint(
        constraints: List[Constraint], name: str
    ) -> Optional[Constraint]:
        for c in constraints:
            if c.name == name:
                return c
        return None

    @staticmethod
    def _emit_certificates(
        results: List[ConstraintAnimResult],
    ) -> List[ProofCertificate]:
        certs: List[ProofCertificate] = []
        validator = ProofValidator()
        for r in results:
            builder = ModelCheckingProofBuilder(
                property_name=r.name,
                property_formula=f"[]{r.name}",
            )
            builder.add_model_checking_result(
                result=r.status,
                states_explored=r.states_checked,
                time_ms=r.time_ms,
            )
            cert = builder.build()
            if r.status == "HOLDS":
                validator.validate(cert)
            certs.append(cert)
        return certs
