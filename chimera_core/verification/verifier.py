"""
CSL Core - Logic Verifier (Z3 Engine)

Enterprise-grade compile-time consistency checking for CSL-Core.

What this verifier guarantees (CSL-Core scope):
1) Rule reachability: each WHEN/ALWAYS rule condition is satisfiable under declared domains.
2) Per-rule internal consistency: condition ∧ action_constraint is satisfiable.
3) Pairwise overlap action conflicts: if two rules can co-trigger, their action constraints must be jointly satisfiable.
4) Policy-wide action consistency (NEW / "big move"):
   For each action variable, we attempt to force all rules that constrain it to trigger together and use Z3 UNSAT CORE
   extraction to find *some* minimal conflicting subset. We then confirm that the subset's conditions overlap (SAT)
   to distinguish real contradictions from mutually-exclusive combinations.

Design principles:
- Deterministic, fail-closed verification (unsupported => error).
- Actionable debug: return overlap models, and UNSAT cores for conflicts.
- Coverage reporting: which rules were analyzed vs skipped/unsupported.

Notes:
- CSL-Core only statically analyzes WHEN/ALWAYS (simple temporal operators).
- Temporal/modal/advanced AST features may exist as enterprise teaser, but CSL-Core verifier fails-closed if it cannot
  encode them into Z3.
"""

from __future__ import annotations
import re
import z3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Set, Union, Iterable

from ..language.ast import (
    Constitution, VariableDeclaration, Constraint, Expression,
    BinaryOp, UnaryOp, Literal, Variable, MemberAccess, FunctionCall,
    LogicalOperator, ComparisonOperator, ArithmeticOperator,
    TemporalOperator, ModalOperator
)


class UnsupportedExpressionError(Exception):
    """Raised when CSL-Core cannot encode an AST expression into Z3."""
    pass


@dataclass(frozen=True)
class VerificationIssue:
    kind: str  # "CONTRADICTION" | "UNSUPPORTED" | "INTERNAL_ERROR" | "UNREACHABLE" | "COVERAGE"
    message: str
    rules: List[str]
    locations: List[Optional[tuple]]
    severity: str = "error"  # "error" | "warning"
    model: Optional[Dict[str, Any]] = None
    unsat_core: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class VerificationCoverage:
    total_constraints: int
    analyzed_pairs: int
    skipped_pairs_temporal: int
    skipped_pairs_unsupported: int
    unreachable_rules: int
    internally_inconsistent_rules: int
    policywide_checks: int


class LogicVerifier:
    def __init__(self):
        self.solver = z3.Solver()
        self.solver.set(unsat_core=True)
        self.z3_vars: Dict[str, z3.ExprRef] = {}
        self.constraint_map: Dict[str, Constraint] = {}
        self._domain_decls: List[VariableDeclaration] = []
        
        self.debug: bool = False
        self._trace: List[Dict[str, Any]] = []
        self._trace_max: int = 400
        self._active_rule: Optional[str] = None

    # -----------------------------
    # Public API
    # -----------------------------
    def verify(self, constitution: Constitution, *, debug: bool = False) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Main verification entry point.

        Returns:
            (is_valid, issues)

        issues is a list of dict payloads (structured, SuggestionEngine friendly).
        """
        self.debug = bool(debug)
        self._trace = []
        self._active_rule = None
        self._t("verify_start")
        
        self.solver.reset()
        self.z3_vars = {}
        self.constraint_map = {c.name: c for c in (constitution.constraints or [])}
        issues: List[VerificationIssue] = []

        coverage = VerificationCoverage(
            total_constraints=len(constitution.constraints or []),
            analyzed_pairs=0,
            skipped_pairs_temporal=0,
            skipped_pairs_unsupported=0,
            unreachable_rules=0,
            internally_inconsistent_rules=0,
            policywide_checks=0,
        )

        try:
            # 0) register domain variables / basic domain constraints
            domain = constitution.domain
            self._domain_decls = getattr(domain, "variable_declarations", []) if domain else []
            if self._domain_decls:
                self._register_variables(self._domain_decls)

            constraints = constitution.constraints or []
            if not constraints:
                ok = True
                return ok, []

            # 1) Reachability + internal consistency per rule
            for c in constraints:
                # debug context for this rule
                self._active_rule = c.name
                self._t("rule_start", name=c.name, phase="per_rule", action_var=getattr(c.action, "variable", None))
            
                if not self._is_static_analyzable(c):
                    issues.append(VerificationIssue(
                        kind="UNSUPPORTED",
                        message="Constraint uses a temporal operator CSL-Core does not statically analyze (only WHEN/ALWAYS).",
                        rules=[c.name],
                        locations=[getattr(c, "location", None)],
                        severity="error"
                    ))
                    coverage = self._cov_inc(coverage, skipped_temporal=1)
                    continue
            
                try:
                    cond = self._expr_to_z3(c.condition.condition)
                    act = self._action_to_z3(c)
                except UnsupportedExpressionError as ue:
                    issues.append(VerificationIssue(
                        kind="UNSUPPORTED",
                        message=f"Unsupported expression: {ue}",
                        rules=[c.name],
                        locations=[getattr(c, "location", None)],
                        severity="error"
                    ))
                    coverage = self._cov_inc(coverage, skipped_unsupported=1)
                    continue

                # reachability: cond SAT?
                self.solver.push()
                self.solver.add(cond)
                sat = self.solver.check()
                self.solver.pop()
                
                if sat == z3.unsat:
                    coverage = self._cov_inc(coverage, unreachable=1)
                    issues.append(VerificationIssue(
                        kind="UNREACHABLE",
                        message="Rule condition is unsatisfiable under current domain declarations (rule can never trigger).",
                        rules=[c.name],
                        locations=[getattr(c, "location", None)],
                        severity="warning",
                    ))
                    # IMPORTANT: unreachable rules should not produce internal inconsistency errors
                    continue
                
                # internal consistency: cond ∧ act SAT?
                self.solver.push()
                
                a_cond = z3.Bool(f"assume::{c.name}::cond")
                a_act  = z3.Bool(f"assume::{c.name}::act")
                
                self.solver.add(z3.Implies(a_cond, cond))
                self.solver.add(z3.Implies(a_act, act))
                
                sat2 = self.solver.check(a_cond, a_act)
                if sat2 == z3.unsat:
                    core = [str(x) for x in self.solver.unsat_core()]
                    coverage = self._cov_inc(coverage, internal_inconsistent=1)
                    issues.append(VerificationIssue(
                        kind="CONTRADICTION",
                        message="Rule is internally inconsistent: WHEN-condition can be true but THEN-action cannot be satisfied.",
                        rules=[c.name],
                        locations=[getattr(c, "location", None)],
                        severity="error",
                        unsat_core=core
                    ))
                
                self.solver.pop()

            # 2) Pairwise overlap + action conflict (as before, but with model + core)
            for i, c1 in enumerate(constraints):
                for j, c2 in enumerate(constraints):
                    if i >= j:
                        continue

                    if not self._is_static_analyzable(c1) or not self._is_static_analyzable(c2):
                        coverage = self._cov_inc(coverage, skipped_temporal=1)
                        continue

                    try:
                        cond1 = self._expr_to_z3(c1.condition.condition)
                        cond2 = self._expr_to_z3(c2.condition.condition)
                        act1 = self._action_to_z3(c1)
                        act2 = self._action_to_z3(c2)
                        
                    except UnsupportedExpressionError:
                        coverage = self._cov_inc(coverage, skipped_unsupported=1)
                        continue

                    coverage = self._cov_inc(coverage, analyzed_pairs=1)

                    # overlap check: cond1 ∧ cond2
                    self.solver.push()
                    self.solver.add(cond1)
                    self.solver.add(cond2)
                    if self.solver.check() == z3.sat:
                        m = self.solver.model()
                        model_dict = self._model_to_dict(m, self._collect_symbols_from_constraints([c1, c2]))

                        # full: cond1 ∧ cond2 ∧ act1 ∧ act2
                        self.solver.push()
                        a_cond1 = z3.Bool(f"assume::{c1.name}::cond")
                        a_cond2 = z3.Bool(f"assume::{c2.name}::cond")
                        a_act1 = z3.Bool(f"assume::{c1.name}::act")
                        a_act2 = z3.Bool(f"assume::{c2.name}::act")

                        self.solver.add(z3.Implies(a_cond1, cond1))
                        self.solver.add(z3.Implies(a_cond2, cond2))
                        self.solver.add(z3.Implies(a_act1, act1))
                        self.solver.add(z3.Implies(a_act2, act2))

                        res = self.solver.check(a_cond1, a_cond2, a_act1, a_act2)
                        if res == z3.unsat:
                            core = [str(x) for x in self.solver.unsat_core()]
                            issues.append(VerificationIssue(
                                kind="CONTRADICTION",
                                message="Rules overlap and their action constraints cannot be satisfied together.",
                                rules=[c1.name, c2.name],
                                locations=[getattr(c1, "location", None), getattr(c2, "location", None)],
                                severity="error",
                                model=model_dict,
                                unsat_core=core
                            ))
                        self.solver.pop()

                    self.solver.pop()

            # 3) Policy-wide action consistency per action variable (NEW)
            var_to_rules: Dict[str, List[Constraint]] = {}
            for c in constraints:
                var_to_rules.setdefault(c.action.variable, []).append(c)

            for var_name, rules in sorted(var_to_rules.items(), key=lambda kv: kv[0]):
                if len(rules) < 2:
                    continue
                coverage = self._cov_inc(coverage, policywide=1)
                issue = self._policywide_action_conflict(var_name, rules)
                if issue:
                    issues.append(issue)

            # 4) Coverage summary (warning-level, deterministic)
            issues.append(VerificationIssue(
                kind="COVERAGE",
                message="Verification coverage summary.",
                rules=[],
                locations=[],
                severity="warning",
                meta={
                    "total_constraints": coverage.total_constraints,
                    "analyzed_pairs": coverage.analyzed_pairs,
                    "skipped_pairs_temporal": coverage.skipped_pairs_temporal,
                    "skipped_pairs_unsupported": coverage.skipped_pairs_unsupported,
                    "unreachable_rules": coverage.unreachable_rules,
                    "internally_inconsistent_rules": coverage.internally_inconsistent_rules,
                    "policywide_checks": coverage.policywide_checks,
                }
            ))

            ok = (len([x for x in issues if x.severity == "error"]) == 0)
            return ok, [i.__dict__ for i in issues]

        except Exception as e:
            internal = VerificationIssue(
                kind="INTERNAL_ERROR",
                message=f"Critical Verification Error: {str(e)}",
                rules=[r for r in ([self._active_rule] if self._active_rule else [])],
                locations=[],
                severity="error",
                meta={
                    "active_rule": self._active_rule,
                    "trace_tail": self._trace[-80:],  # last 80 events
                } if self.debug else None
            )
            return False, [internal.__dict__]
        
    def _t(self, event: str, **data):
        if not self.debug:
            return
        rec = {"event": event, "rule": self._active_rule, **data}
        self._trace.append(rec)
        if len(self._trace) > self._trace_max:
            self._trace.pop(0)

    # -----------------------------
    # Policy-wide conflicts
    # -----------------------------
    def _policywide_action_conflict(self, action_var: str, rules: List[Constraint]) -> Optional[VerificationIssue]:
        """
        Try to find a conflicting subset of rules that constrain the same action variable.
    
        We "force-trigger" each rule by an assumption literal li:
          li -> (cond_i ∧ action_i)
    
        Then we ask Z3 for SAT with all li asserted.
        If UNSAT, we extract UNSAT core (subset of li) and then confirm whether the subset's CONDITIONS overlap.
        - If conditions overlap (SAT) but full (cond ∧ action) is UNSAT => real contradiction.
        - If conditions do NOT overlap => mutually exclusive combination (warning).
    
        Fail-closed details:
        - If we get UNSAT but Z3 returns an empty UNSAT core, we emit INTERNAL_ERROR (actionable).
        - Assumption literal names are sanitized for stability/readability.
        """
    
        def _sanitize_lit(s: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]", "_", str(s))
    
        solver = z3.Solver()
        solver.set(unsat_core=True)
    
        # Re-apply domain constraints into this solver (reuse self.z3_vars symbol table)
        for decl in self._domain_decls:
            self._add_domain_bounds_to(solver, decl)
    
        # Encode each rule (fail-closed on unsupported)
        encoded: List[Tuple[Constraint, z3.ExprRef, z3.ExprRef]] = []
        for c in rules:
            if not self._is_static_analyzable(c):
                return VerificationIssue(
                    kind="UNSUPPORTED",
                    message="Policy-wide check encountered temporal operator CSL-Core does not statically analyze.",
                    rules=[c.name],
                    locations=[getattr(c, "location", None)],
                    severity="error",
                )
            try:
                cond = self._expr_to_z3(c.condition.condition)
                act = self._action_to_z3(c)
            except UnsupportedExpressionError as ue:
                return VerificationIssue(
                    kind="UNSUPPORTED",
                    message=f"Policy-wide check encountered unsupported expression: {ue}",
                    rules=[c.name],
                    locations=[getattr(c, "location", None)],
                    severity="error",
                )
            encoded.append((c, cond, act))
    
        # Build assumptions once (deterministic, no solver-state contamination)
        assumptions: List[z3.BoolRef] = []
        lit_to_rule: Dict[z3.BoolRef, Constraint] = {}
    
        safe_action = _sanitize_lit(action_var)
        for (c, cond, act) in encoded:
            lit_name = f"pw__{safe_action}__{_sanitize_lit(c.name)}"
            lit = z3.Bool(lit_name)
            assumptions.append(lit)
            lit_to_rule[lit] = c
            solver.add(z3.Implies(lit, z3.And(cond, act)))
    
        res = solver.check(*assumptions)
        if res != z3.unsat:
            return None
    
        # Extract UNSAT core (BoolRef list) and map to constraints
        core_refs = solver.unsat_core()  # List[BoolRef]
        if not core_refs:
            # Fail-closed: UNSAT without a core is not actionable; treat as internal verifier error
            return VerificationIssue(
                kind="INTERNAL_ERROR",
                message=(
                    f"Policy-wide check is UNSAT for action variable '{action_var}' but Z3 returned an empty UNSAT core. "
                    f"Enable debug trace / check solver configuration."
                ),
                rules=[c.name for (c, _, _) in encoded],
                locations=[getattr(c, "location", None) for (c, _, _) in encoded],
                severity="error",
                meta={"action_var": action_var},
            )
    
        core_lits = [str(x) for x in core_refs]
        core_constraints = [lit_to_rule[x] for x in core_refs if x in lit_to_rule]
        core_rules = [c.name for c in core_constraints]
        core_locs = [getattr(c, "location", None) for c in core_constraints]
    
        # 1) Are conditions jointly satisfiable? (distinguish real contradiction vs mutual exclusivity)
        cond_solver = z3.Solver()
        cond_solver.set(unsat_core=True)
        
        for decl in self._domain_decls:
            self._add_domain_bounds_to(cond_solver, decl)
    
        cond_exprs: List[z3.ExprRef] = []
        for c in core_constraints:
            _, cond, _ = next((t for t in encoded if t[0].name == c.name), (None, None, None))
            if cond is not None:
                cond_exprs.append(cond)
    
        if cond_exprs:
            cond_solver.add(z3.And(*cond_exprs))
    
        cond_sat = cond_solver.check()
        if cond_sat == z3.unsat:
            return VerificationIssue(
                kind="UNREACHABLE",
                message=(
                    f"Policy-wide UNSAT core found for action variable '{action_var}', "
                    f"but the rules are mutually exclusive (their WHEN conditions cannot all hold together)."
                ),
                rules=core_rules,
                locations=core_locs,
                severity="warning",
                unsat_core=core_lits,
            )
    
        # Overlap model (conditions only) can be useful to explain "when" the conflict could occur
        model_dict = None
        if cond_sat == z3.sat:
            m = cond_solver.model()
            model_dict = self._model_to_dict(m, self._collect_symbols_from_constraints(core_constraints))
    
        # 2) Full check: conditions + actions
        full_solver = z3.Solver()
        full_solver.set(unsat_core=True)
        
        for decl in self._domain_decls:
            self._add_domain_bounds_to(full_solver, decl)
    
        if cond_exprs:
            full_solver.add(z3.And(*cond_exprs))
    
        for c in core_constraints:
            _, _, act = next((t for t in encoded if t[0].name == c.name), (None, None, None))
            if act is not None:
                full_solver.add(act)
    
        full_res = full_solver.check()
        if full_res == z3.unsat:
            return VerificationIssue(
                kind="CONTRADICTION",
                message=(
                    f"Policy-wide conflict for action variable '{action_var}': "
                    f"these rules can co-trigger but their THEN constraints are jointly unsatisfiable."
                ),
                rules=core_rules,
                locations=core_locs,
                severity="error",
                model=model_dict,
                unsat_core=core_lits,
            )
    
        # If full is sat, contradiction was not confirmed (rare, but safe)
        return VerificationIssue(
            kind="COVERAGE",
            message=(
                f"Policy-wide UNSAT core was reported by Z3 for '{action_var}', but confirmation checks did not reproduce "
                f"a definitive contradiction. (This can happen with mixed sorts / heuristic casting.)"
            ),
            rules=core_rules,
            locations=core_locs,
            severity="warning",
            unsat_core=core_lits,
        )

    # -----------------------------
    # Domain + helpers
    # -----------------------------
    def _register_variables(self, declarations: List[VariableDeclaration]):
        """
        Register Z3 symbols based on domain declarations and add bound constraints to self.solver.
    
        IMPORTANT:
        - Do NOT use substring checks like `"int" in domain_str` because enum values may contain "int"
          (e.g., "INTERNAL" -> falsely detected as int).
        - We only treat a domain as numeric if:
            (a) it's an explicit primitive token: int / integer / float / real / bool
            (b) it's an interval: 0..1000
          Otherwise (including enum sets like {"INTERNAL","EXTERNAL"}) default to String.
        """
        for decl in declarations:
            name = decl.name
            domain_raw = str(decl.domain).strip()
            domain_str = domain_raw.lower()
    
            # 1) Interval form: "a..b" (optionally spaces)
            if ".." in domain_str:
                self.z3_vars[name] = z3.Int(name)
                self._add_interval_bounds(self.solver, name, domain_str)
                self._t("register_var", name=name, domain=domain_raw, sort=str(self.z3_vars[name].sort()))
                continue
    
            # 2) Explicit primitive tokens (whole token match, not substring)
            token = re.sub(r"\s+", "", domain_str)  # remove whitespace
            if token in ("int", "integer"):
                self.z3_vars[name] = z3.Int(name)
                self._t("register_var", name=name, domain=domain_raw, sort=str(self.z3_vars[name].sort()))
                continue
    
            if token in ("float", "real"):
                self.z3_vars[name] = z3.Real(name)
                self._t("register_var", name=name, domain=domain_raw, sort=str(self.z3_vars[name].sort()))
                continue
    
            if token in ("bool", "boolean"):
                self.z3_vars[name] = z3.Bool(name)
                self._t("register_var", name=name, domain=domain_raw, sort=str(self.z3_vars[name].sort()))
                continue
    
            # 3) Everything else (including enum sets) => String
            self.z3_vars[name] = z3.String(name)
            
            enum_vals = self._parse_enum_set(domain_raw)
            if enum_vals:
                v = self.z3_vars[name]
                self.solver.add(z3.Or(*[v == z3.StringVal(x) for x in enum_vals]))
                self._t("register_enum", name=name, domain=domain_raw, values=enum_vals)
            
            # Always trace registration
            self._t("register_var", name=name, domain=domain_raw, sort=str(self.z3_vars[name].sort()))

    def _add_interval_bounds(self, solver: z3.Solver, name: str, domain_str: str):
        parts = [p.strip() for p in domain_str.split("..")]
        if len(parts) != 2:
            return
        try:
            low, high = int(parts[0]), int(parts[1])
        except Exception:
            return
    
        var = self.z3_vars.get(name)
        if var is None:
            var = z3.Int(name)
            self.z3_vars[name] = var
    
        solver.add(var >= low)
        solver.add(var <= high)

    def _action_to_z3(self, c: Constraint) -> z3.ExprRef:
        var_name = c.action.variable
        if var_name not in self.z3_vars:
            # safest default for unknown action variable: string
            self.z3_vars[var_name] = z3.String(var_name)

        z3_var = self.z3_vars[var_name]
        act_val = self._expr_to_z3(c.action.value)
        self._t(
            "action_encode",
            action_var=var_name,
            action_var_sort=str(z3_var.sort()),
            action_val_sort=str(act_val.sort()),
            action_val=str(act_val),
            modal=str(c.action.modal_operator),
        )
        return self._modal_to_z3(c.action.modal_operator, z3_var, act_val)
    
    def _is_int(self, x: z3.ExprRef) -> bool:
        return x.sort() == z3.IntSort()
    
    def _is_real(self, x: z3.ExprRef) -> bool:
        return x.sort() == z3.RealSort()
    
    def _is_num(self, x: z3.ExprRef) -> bool:
        return self._is_int(x) or self._is_real(x)
    
    def _coerce_numeric_pair(self, left: z3.ExprRef, right: z3.ExprRef, ctx: str) -> tuple[z3.ExprRef, z3.ExprRef]:
        """
        Ensure numeric ops don't crash on Int/Real mismatch.
        - If both numeric: coerce Int -> Real when needed.
        - If non-numeric encountered: fail-closed with actionable message.
        """
        if not self._is_num(left) or not self._is_num(right):
            raise UnsupportedExpressionError(
                f"Non-numeric operand in numeric operation '{ctx}': left_sort={left.sort()} right_sort={right.sort()}"
            )
    
        # If either side is Real, cast Int to Real so sorts match
        if self._is_real(left) and self._is_int(right):
            return left, z3.ToReal(right)
        if self._is_int(left) and self._is_real(right):
            return z3.ToReal(left), right
        return left, right
    
    def _safe_cmp(self, op_name: str, left: z3.ExprRef, right: z3.ExprRef) -> z3.BoolRef:
        try:
            if op_name == "EQ":
                return left == right
            if op_name == "NEQ":
                return left != right
            raise UnsupportedExpressionError(f"Unsupported safe comparator: {op_name}")
        except z3.Z3Exception as e:
            raise RuntimeError(
                f"sort mismatch in {op_name}: left_sort={left.sort()} right_sort={right.sort()} | left={left} right={right}"
            ) from e

    def _expr_to_z3(self, expr: Expression) -> z3.ExprRef:
        if isinstance(expr, BinaryOp):
            left = self._expr_to_z3(expr.left)
            right = self._expr_to_z3(expr.right)
            op = expr.operator
        
            # Debug trace for every binary op (SAFE: left/right/op exist here)
            self._t(
                "binop",
                op=str(op),
                left_sort=str(left.sort()),
                right_sort=str(right.sort()),
                left=str(left),
                right=str(right),
            )

            if op == LogicalOperator.AND: return z3.And(left, right)
            if op == LogicalOperator.OR: return z3.Or(left, right)

            if op == ComparisonOperator.EQ: return self._safe_cmp("EQ", left, right)
            if op == ComparisonOperator.NEQ: return self._safe_cmp("NEQ", left, right)
            
            if op == ComparisonOperator.LT:
                l, r = self._coerce_numeric_pair(left, right, "LT")
                return l < r
            if op == ComparisonOperator.GT:
                l, r = self._coerce_numeric_pair(left, right, "GT")
                return l > r
            if op == ComparisonOperator.LTE:
                l, r = self._coerce_numeric_pair(left, right, "LTE")
                return l <= r
            if op == ComparisonOperator.GTE:
                l, r = self._coerce_numeric_pair(left, right, "GTE")
                return l >= r
            
            if op == ArithmeticOperator.ADD:
                l, r = self._coerce_numeric_pair(left, right, "ADD")
                return l + r
            if op == ArithmeticOperator.SUB:
                l, r = self._coerce_numeric_pair(left, right, "SUB")
                return l - r
            if op == ArithmeticOperator.MUL:
                l, r = self._coerce_numeric_pair(left, right, "MUL")
                return l * r
            if op == ArithmeticOperator.DIV:
                l, r = self._coerce_numeric_pair(left, right, "DIV")
                return l / r

            raise UnsupportedExpressionError(f"Unsupported binary operator: {op}")

        if isinstance(expr, UnaryOp):
            operand = self._expr_to_z3(expr.operand)
            if expr.operator == LogicalOperator.NOT: return z3.Not(operand)
            if expr.operator == ArithmeticOperator.SUB: return -operand
            raise UnsupportedExpressionError(f"Unsupported unary operator: {expr.operator}")

        if isinstance(expr, Variable):
            if expr.name in self.z3_vars:
                self._t("var_ref", name=expr.name, sort=str(self.z3_vars[expr.name].sort()))
                return self.z3_vars[expr.name]
            # FAIL-CLOSED: undeclared variables must not auto-create fresh symbols
            raise UnsupportedExpressionError(
                f"Undeclared variable referenced in verifier: '{expr.name}'. "
                f"Declare it in DOMAIN (or fix typo)."
            )
        
        if isinstance(expr, MemberAccess):
            key = self._member_access_key(expr)
            if key in self.z3_vars:
                self._t("member_ref", key=key, sort=str(self.z3_vars[key].sort()))
                return self.z3_vars[key]
            # FAIL-CLOSED: member access must map to a declared symbol key
            raise UnsupportedExpressionError(
                f"Undeclared member access referenced in verifier: '{key}'. "
                f"Declare it in DOMAIN (or fix typo)."
            )

        if isinstance(expr, Literal):
            self._t("literal", lit_type=str(expr.type), value=str(expr.value))
            if expr.type == "string": return z3.StringVal(expr.value)
            if expr.type == "bool": return z3.BoolVal(bool(expr.value))
            if expr.type == "int": return z3.IntVal(int(expr.value))
            return z3.RealVal(str(expr.value))

        if isinstance(expr, FunctionCall):
            self._t("fn_call", name=str(expr.name), argc=len(expr.args or []))
            if getattr(expr, "kwargs", None):
                raise UnsupportedExpressionError(f"Keyword arguments not supported: {expr.name}()")
            

            name = expr.name
            args = [self._expr_to_z3(a) for a in expr.args]

            if name == "abs" and len(args) == 1: return z3.If(args[0] >= 0, args[0], -args[0])
            if name == "max" and len(args) == 2: return z3.If(args[0] >= args[1], args[0], args[1])
            if name == "min" and len(args) == 2: return z3.If(args[0] <= args[1], args[0], args[1])
            if name == "len" and len(args) == 1: return z3.Length(args[0])

            raise UnsupportedExpressionError(f"Unsupported function call: {name}({len(args)} args)")

        raise UnsupportedExpressionError(f"Unsupported expression node: {expr.__class__.__name__}")

    def _member_access_key(self, expr: MemberAccess) -> str:
        parts = []
        cur = expr
        while isinstance(cur, MemberAccess):
            parts.append(cur.member)
            cur = cur.object
        if isinstance(cur, Variable):
            parts.append(cur.name)
        else:
            parts.append("obj")
        return ".".join(reversed(parts))

    def _modal_to_z3(self, modal: ModalOperator, var, val):
        if modal == ModalOperator.MUST_BE or modal == ModalOperator.EQ: return var == val
        if modal == ModalOperator.MUST_NOT_BE or modal == ModalOperator.NEQ: return var != val
        
        if modal == ModalOperator.LT:
            l, r = self._coerce_numeric_pair(var, val, "MODAL_LT")
            return l < r
        if modal == ModalOperator.GT:
            l, r = self._coerce_numeric_pair(var, val, "MODAL_GT")
            return l > r
        if modal == ModalOperator.LTE:
            l, r = self._coerce_numeric_pair(var, val, "MODAL_LTE")
            return l <= r
        if modal == ModalOperator.GTE:
            l, r = self._coerce_numeric_pair(var, val, "MODAL_GTE")
            return l >= r
        
        if modal == ModalOperator.MAY_BE: return z3.BoolVal(True)
        raise UnsupportedExpressionError(f"Unsupported modal operator: {modal}")

    def _is_static_analyzable(self, constraint: Constraint) -> bool:
        op = constraint.condition.temporal_operator
        return op in (TemporalOperator.WHEN, TemporalOperator.ALWAYS)

    def _collect_symbols_from_constraints(self, constraints: List[Constraint]) -> Set[str]:
        symbols = set()
        for c in constraints:
            symbols |= self._collect_from_expr(c.condition.condition)
            symbols |= self._collect_from_expr(c.action.value)
            symbols.add(c.action.variable)
        return symbols

    def _collect_from_expr(self, expr: Expression) -> Set[str]:
        out = set()

        def walk(e):
            if isinstance(e, Variable):
                out.add(e.name)
            elif isinstance(e, MemberAccess):
                out.add(self._member_access_key(e))
                walk(e.object)
            elif isinstance(e, BinaryOp):
                walk(e.left); walk(e.right)
            elif isinstance(e, UnaryOp):
                walk(e.operand)
            elif isinstance(e, FunctionCall):
                for a in e.args: walk(a)

        walk(expr)
        return out

    def _model_to_dict(self, model: z3.ModelRef, symbols: Set[str]) -> Dict[str, Any]:
        result = {}
        for name in sorted(symbols):
            z = self.z3_vars.get(name)
            if z is None:
                continue
            try:
                val = model.eval(z, model_completion=True)
                if z3.is_int_value(val):
                    result[name] = int(str(val))
                elif z3.is_rational_value(val):
                    result[name] = str(val)
                elif z3.is_bool(val):
                    result[name] = bool(z3.is_true(val))
                else:
                    result[name] = str(val)
            except Exception:
                continue
        return result

    def _cov_inc(
        self,
        cov: VerificationCoverage,
        analyzed_pairs: int = 0,
        skipped_temporal: int = 0,
        skipped_unsupported: int = 0,
        unreachable: int = 0,
        internal_inconsistent: int = 0,
        policywide: int = 0,
    ) -> VerificationCoverage:
        return VerificationCoverage(
            total_constraints=cov.total_constraints,
            analyzed_pairs=cov.analyzed_pairs + analyzed_pairs,
            skipped_pairs_temporal=cov.skipped_pairs_temporal + skipped_temporal,
            skipped_pairs_unsupported=cov.skipped_pairs_unsupported + skipped_unsupported,
            unreachable_rules=cov.unreachable_rules + unreachable,
            internally_inconsistent_rules=cov.internally_inconsistent_rules + internal_inconsistent,
            policywide_checks=cov.policywide_checks + policywide,
        )
    
    def _parse_enum_set(self, domain_raw: str) -> Optional[List[str]]:
        s = domain_raw.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        inner = s[1:-1].strip()
        if not inner:
            return None
        parts = [p.strip() for p in inner.split(",")]
        vals = []
        for p in parts:
            if not p:
                continue
            # strip single/double quotes if present
            if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                p = p[1:-1]
            vals.append(p)
        return vals or None
    
    def _add_domain_bounds_to(self, solver: z3.Solver, decl: VariableDeclaration):
        """
        Re-apply domain bounds/membership constraints for a declaration into an arbitrary solver.
    
        This is used for:
        - policy-wide conflict solvers
        - condition-only overlap solvers
        - any auxiliary SAT checks
    
        Must be deterministic and must not assume the variable was already registered.
        """
        name = decl.name
        domain_raw = str(decl.domain).strip()
        domain_str = domain_raw.lower()
    
        # Ensure symbol exists in self.z3_vars (reuse consistent symbol table)
        if name not in self.z3_vars:
            # mirror registration logic (minimal):
            if ".." in domain_str:
                self.z3_vars[name] = z3.Int(name)
            else:
                token = re.sub(r"\s+", "", domain_str)
                if token in ("int", "integer"):
                    self.z3_vars[name] = z3.Int(name)
                elif token in ("float", "real"):
                    self.z3_vars[name] = z3.Real(name)
                elif token in ("bool", "boolean"):
                    self.z3_vars[name] = z3.Bool(name)
                else:
                    self.z3_vars[name] = z3.String(name)
    
        # Interval bounds
        if ".." in domain_str:
            self._add_interval_bounds(solver, name, domain_str)
            return
    
        # Enum membership bounds (string sets)
        enum_vals = self._parse_enum_set(domain_raw)
        if enum_vals:
            v = self.z3_vars[name]
            solver.add(z3.Or(*[v == z3.StringVal(x) for x in enum_vals]))
            return

        # No extra bounds for primitive tokens without intervals/enums
        return
