"""
CSL Core - Runtime Guard

Deterministic, fail-closed runtime enforcement for compiled CSL policies.
Designed for:
- zero external dependencies
- minimal latency
- stable, auditable outputs (policy meta, triggered rules, violations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
import time

from .language.compiler import CompiledConstitution, CompiledConstraint
from .language.ast import EnforcementMode, ModalOperator


MissingKeyBehavior = Literal["block", "warn", "ignore"]
EvaluationErrorBehavior = Literal["block", "warn", "ignore"]


class ChimeraError(Exception):
    """Raised when enforcement blocks an input."""
    def __init__(
        self,
        message: str,
        constraint_name: str,
        context: Dict[str, Any],
        *,
        result: Optional["GuardResult"] = None,
    ):
        super().__init__(message)
        self.constraint_name = constraint_name
        self.context = context
        self.result = result


@dataclass
class GuardResult:
    """Result of a verification run (stable, audit-friendly)."""
    allowed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    triggered_rule_ids: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    # Optional metadata (forward-compatible with future packaging/versioning)
    domain_name: Optional[str] = None
    policy_name: Optional[str] = None
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    policy_hash: Optional[str] = None
    engine_version: Optional[str] = None

    # Enforcement info
    enforcement: str = "ACTIVE"  # ACTIVE | DRY_RUN

    @property
    def is_clean(self) -> bool:
        return (len(self.violations) == 0) and (len(self.warnings) == 0)


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Runtime behavior toggles. Defaults are conservative (fail-closed).
    """
    raise_on_block: bool = True
    collect_all_violations: bool = True  # If False, fail fast on first BLOCK violation
    missing_key_behavior: MissingKeyBehavior = "block"
    evaluation_error_behavior: EvaluationErrorBehavior = "block"
    dry_run: bool = False  # If True, never raise; always allow, but keep violations


class ChimeraGuard:
    """
    High-performance runtime enforcer.

    Usage:
        guard = ChimeraGuard(compiled_policy)
        guard.verify({"risk": 0.9, "action": "TRANSFER"})
    """

    def __init__(self, constitution: CompiledConstitution, config: Optional[RuntimeConfig] = None):
        self.constitution = constitution
        self.domain_name = getattr(constitution, "domain_name", None)
        self.config = config or RuntimeConfig()

        # Optional metadata passthrough if compiler packaged them
        self._policy_meta = {
            "policy_name": getattr(constitution, "policy_name", None),
            "policy_id": getattr(constitution, "policy_id", None),
            "policy_version": getattr(constitution, "policy_version", None),
            "policy_hash": getattr(constitution, "policy_hash", None),
            "engine_version": getattr(constitution, "engine_version", None),
        }


    def verify(self, context: Dict[str, Any]) -> GuardResult:
        """
        Execute compiled constraints against provided context.
    
        Determinism guarantees:
        - constraints evaluated in compiled order
        - triggered_rule_ids recorded in evaluation order
        - warnings/violations appended deterministically
        """
        start_time = time.perf_counter()
    
        violations: List[str] = []
        warnings: List[str] = []
        triggered: List[str] = []
    
        # Track which constraint actually produced a BLOCK-level violation (most actionable)
        last_blocking_constraint: Optional[str] = None
    
        # 1) Evaluate constraints in order
        for constraint in self.constitution.constraints:
            # 1a) WHEN condition
            try:
                applies = bool(constraint.condition_expr.evaluate(context))
            except KeyError as e:
                applies = False
                blocked = self._handle_condition_failure(
                    constraint,
                    f"Missing key while evaluating condition: {str(e)}",
                    violations,
                    warnings,
                )
                if blocked:
                    last_blocking_constraint = constraint.name
                if self._should_hard_stop(constraint, violations):
                    break
                continue
            except Exception as e:
                applies = False
                blocked = self._handle_condition_failure(
                    constraint,
                    f"Error while evaluating condition: {type(e).__name__}: {str(e)}",
                    violations,
                    warnings,
                )
                if blocked:
                    last_blocking_constraint = constraint.name
                if self._should_hard_stop(constraint, violations):
                    break
                continue
    
            if not applies:
                continue
    
            # Condition triggered
            triggered.append(constraint.name)
    
            # 1b) THEN action compliance
            action_eval_failed = False
            expected_val = None
            actual_val = None
    
            try:
                expected_val = constraint.action_value_expr.evaluate(context)
    
                actual_val, missing_blocked = self._get_actual_value(
                    context, constraint.action_variable, violations, warnings, constraint
                )
    
                # If missing key behavior produced a BLOCK violation, do not add a second synthetic violation.
                if missing_blocked:
                    action_eval_failed = True
                    last_blocking_constraint = constraint.name
                    is_compliant = False
                else:
                    is_compliant = self._check_compliance(actual_val, expected_val, constraint.modal_operator)
    
            except KeyError as e:
                action_eval_failed = True
                is_compliant = False
                blocked = self._handle_action_failure(
                    constraint,
                    f"Missing key while evaluating action: {str(e)}",
                    violations,
                    warnings,
                )
                if blocked:
                    last_blocking_constraint = constraint.name
    
            except Exception as e:
                action_eval_failed = True
                is_compliant = False
                blocked = self._handle_action_failure(
                    constraint,
                    f"Error while evaluating action: {type(e).__name__}: {str(e)}",
                    violations,
                    warnings,
                )
                if blocked:
                    last_blocking_constraint = constraint.name
    
            # If action evaluation failed, do NOT emit a second, synthetic "Violation ..." line.
            if action_eval_failed:
                if self._should_hard_stop(constraint, violations):
                    break
                continue
    
            if is_compliant:
                continue
    
            msg = (
                f"Violation '{constraint.name}': "
                f"{constraint.action_variable}={self._safe_repr(actual_val)} "
                f"must be {constraint.modal_operator.value} {self._safe_repr(expected_val)}."
            )
    
            # 2) Enforce mode per-constraint
            if constraint.enforcement_mode == EnforcementMode.BLOCK:
                violations.append(msg)
                last_blocking_constraint = constraint.name
                if (not self.config.collect_all_violations) and (not self.config.dry_run):
                    break
            elif constraint.enforcement_mode == EnforcementMode.WARN:
                warnings.append(f"[WARN] {msg}")
            elif constraint.enforcement_mode == EnforcementMode.LOG:
                # LOG is non-blocking; keep a breadcrumb as warning-like line (optional)
                warnings.append(f"[LOG] {msg}")
    
        latency = (time.perf_counter() - start_time) * 1000.0
    
        # 3) Final decision (dry_run means never block)
        allowed = (len(violations) == 0) or self.config.dry_run
        result = GuardResult(
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            triggered_rule_ids=triggered,
            latency_ms=float(latency),
            domain_name=self.domain_name,
            enforcement=("DRY_RUN" if self.config.dry_run else "ACTIVE"),
            **self._policy_meta,
        )
    
        # 4) Raise if needed
        if (not result.allowed) and self.config.raise_on_block and (not self.config.dry_run):
            constraint_name = last_blocking_constraint or (
                "Multiple" if len(violations) > 1 else (triggered[-1] if triggered else "Unknown")
            )
            raise ChimeraError(
                message=violations[0] if violations else "Blocked by policy.",
                constraint_name=constraint_name,
                context=context,
                result=result,
            )
    
        return result


    # -----------------------
    # Internal helpers
    # -----------------------

    def _should_hard_stop(self, constraint: CompiledConstraint, violations: List[str]) -> bool:
        if constraint.enforcement_mode != EnforcementMode.BLOCK:
            return False
        if not violations:
            return False
        if self.config.dry_run:
            return False
        return not self.config.collect_all_violations

    def _handle_condition_failure(
        self,
        constraint: CompiledConstraint,
        message: str,
        violations: List[str],
        warnings: List[str],
    ) -> bool:
        behavior = self.config.evaluation_error_behavior
        line = f"Condition evaluation failed in '{constraint.name}': {message}"

        if behavior == "ignore":
            return False
        if behavior == "warn" or constraint.enforcement_mode != EnforcementMode.BLOCK:
            warnings.append(f"[WARN] {line}")
            return False

        # behavior == "block"
        violations.append(line)
        return True

    def _handle_action_failure(
        self,
        constraint: CompiledConstraint,
        message: str,
        violations: List[str],
        warnings: List[str],
    ) -> bool:
        behavior = self.config.evaluation_error_behavior
        line = f"Action evaluation failed in '{constraint.name}': {message}"

        if behavior == "ignore":
            return False
        if behavior == "warn" or constraint.enforcement_mode != EnforcementMode.BLOCK:
            warnings.append(f"[WARN] {line}")
            return False

        violations.append(line)
        return True

    def _get_actual_value(
        self,
        context: Dict[str, Any],
        key: str,
        violations: List[str],
        warnings: List[str],
        constraint: CompiledConstraint,
    ) -> tuple[Any, bool]:
        # Fast path: direct key
        if key in context:
            return context[key], False

        # Support dotted path (member access) e.g. "user.role"
        if "." in key:
            cur: Any = context
            try:
                for part in key.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        raise KeyError(part)
                return cur, False
            except KeyError:
                # fall through to missing key behavior
                pass

        # Missing key behavior
        b = self.config.missing_key_behavior
        msg = f"Missing required key '{key}' for rule '{constraint.name}'."

        if b == "ignore":
            return None, False

        if b == "warn" or constraint.enforcement_mode != EnforcementMode.BLOCK:
            warnings.append(f"[WARN] {msg}")
            return None, False

        # b == "block" and enforcement_mode == BLOCK
        violations.append(msg)
        return None, True
    
    def _check_compliance(self, actual: Any, expected: Any, modal_op: ModalOperator) -> bool:
        """
        Low-level comparison logic. Robust + deterministic + fail-closed.

        Rules:
        - If we cannot safely compare, treat as NON-compliant (fail-closed),
          except MAY_BE which is always compliant.
        - None is treated as "missing/unknown". For MUST_NOT_BE/NEQ, None is compliant
          only if expected is not None (because None != expected).
        - Ordering ops require both sides to be comparable and expected must not be None.
        - For numeric ordering comparisons: allow int/float mixing (cast to float safely),
          but do NOT coerce arbitrary strings to numbers.
        """

        # Permissive operator: always allow
        if modal_op == ModalOperator.MAY_BE:
            return True

        # None handling (fail-closed except NEQ family)
        if actual is None:
            if modal_op in (ModalOperator.MUST_NOT_BE, ModalOperator.NEQ):
                return expected is not None
            return False

        # Equality family: deterministic Python equality
        if modal_op in (ModalOperator.MUST_BE, ModalOperator.EQ):
            return actual == expected

        if modal_op in (ModalOperator.MUST_NOT_BE, ModalOperator.NEQ):
            return actual != expected

        # For ordering, expected None => cannot compare
        if expected is None:
            return False

        # Numeric ordering: allow int/float mix safely
        def _is_number(x: Any) -> bool:
            # bool is a subclass of int; exclude it to avoid weird comparisons
            return isinstance(x, (int, float)) and not isinstance(x, bool)

        if modal_op in (ModalOperator.LT, ModalOperator.GT, ModalOperator.LTE, ModalOperator.GTE):
            # If both are numbers, compare numerically (support int/float mix)
            if _is_number(actual) and _is_number(expected):
                a = float(actual)
                e = float(expected)
                if modal_op == ModalOperator.LT:
                    return a < e
                if modal_op == ModalOperator.GT:
                    return a > e
                if modal_op == ModalOperator.LTE:
                    return a <= e
                if modal_op == ModalOperator.GTE:
                    return a >= e
                return False

            # Otherwise, only allow ordering if both are same type and comparable
            # (fail-closed on mismatched types or non-orderable types)
            if type(actual) is not type(expected):
                return False

            try:
                if modal_op == ModalOperator.LT:
                    return actual < expected
                if modal_op == ModalOperator.GT:
                    return actual > expected
                if modal_op == ModalOperator.LTE:
                    return actual <= expected
                if modal_op == ModalOperator.GTE:
                    return actual >= expected
            except Exception:
                return False

            return False

        # Unknown operator -> fail-closed
        return False
    
            

    def _safe_repr(self, v: Any) -> str:
        s = repr(v)
        if len(s) > 120:
            s = s[:117] + "..."
        return s
