"""
CSL Validator - Semantic Validation

Performs semantic validation on CSL AST:
- Scope checking (Variable declaration)
- Function whitelist checks (CSL-Core safety)
- Basic structural checks (WHEN/THEN presence)
- Optional strictness for implicit action variables (Core-hardening)
"""

from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass

from .ast import (
    Constitution, Domain, Constraint, CausalGraph, StructuralEquation,
    Variable, Literal, BinaryOp, UnaryOp, FunctionCall, MemberAccess,
)


class ValidationError(Exception):
    """Base exception for semantic validation errors."""
    def __init__(self, message: str, location: Optional[tuple] = None):
        self.message = message
        self.location = location
        prefix = f"Line {location[0]}, Col {location[1]}: " if location else ""
        super().__init__(f"{prefix}{message}")


@dataclass
class ValidationContext:
    """Context tracking for validation scope."""
    declared_variables: Set[str]
    variable_types: Dict[str, str]
    causal_graph: Optional[CausalGraph] = None
    current_constraint: Optional[str] = None


class CSLValidator:
    """
    Ensures the logical integrity of the CSL Constitution.
    Acts as the 'Grammar Police' before Compilation.

    allow_implicit_action_var:
        - True  (default): action variable can be implicitly declared (backward/demo friendly)
        - False: strict mode; action variable must be in VARIABLES block (recommended for high-stakes)
    """

    def __init__(self, allow_implicit_action_var: bool = True):
        self.context: Optional[ValidationContext] = None
        self.errors: List[ValidationError] = []
        self.allow_implicit_action_var = allow_implicit_action_var

        # CSL-Core function whitelist (must match compiler/verifier)
        self._valid_functions = {"len", "max", "min", "abs"}

    def validate(self, constitution: Constitution) -> bool:
        """Main validation entry point."""
        self.errors = []

        # 1) Initialize Context
        graph = None
        if constitution.domain and getattr(constitution.domain, "causal_graph", None):
            graph = constitution.domain.causal_graph
        elif getattr(constitution, "causal_graph", None):
            graph = constitution.causal_graph

        self.context = ValidationContext(
            declared_variables=set(),
            variable_types={},
            causal_graph=graph
        )

        # 2) Domain Validation
        if constitution.domain:
            self._validate_domain(constitution.domain)

        # 3) Constraint Validation
        for constraint in (constitution.constraints or []):
            self._validate_constraint(constraint)

        # 4) Report
        if self.errors:
            raise self.errors[0]

        return True

    def _validate_domain(self, domain: Domain):
        """Validates variable declarations and domain-level structures."""
        if getattr(domain, "variable_declarations", None):
            for decl in domain.variable_declarations:
                name = decl.name
                if name in self.context.declared_variables:
                    self.errors.append(ValidationError(f"Duplicate variable declaration: '{name}'", getattr(decl, "location", None)))
                self.context.declared_variables.add(name)
                self.context.variable_types[name] = str(decl.domain)

        # Optional: causal graph checks can live here (cycle detection etc.)

    def _validate_constraint(self, constraint: Constraint):
        self.context.current_constraint = constraint.name

        if not getattr(constraint, "condition", None):
            self.errors.append(ValidationError(f"Constraint '{constraint.name}' missing WHEN clause.", getattr(constraint, "location", None)))
            self.context.current_constraint = None
            return

        if not getattr(constraint, "action", None):
            self.errors.append(ValidationError(f"Constraint '{constraint.name}' missing THEN clause.", getattr(constraint, "location", None)))
            self.context.current_constraint = None
            return

        # Validate condition expression
        self._validate_expression(constraint.condition.condition)

        # Validate action variable scope
        action_var = constraint.action.variable
        if action_var not in self.context.declared_variables:
            if self.allow_implicit_action_var:
                self.context.declared_variables.add(action_var)
            else:
                self.errors.append(ValidationError(
                    f"Action variable '{action_var}' is not declared in VARIABLES block.",
                    getattr(constraint.action, "location", getattr(constraint, "location", None))
                ))

        # Validate action value expression
        self._validate_expression(constraint.action.value)

        self.context.current_constraint = None

    def _validate_expression(self, expr: Any):
        if expr is None:
            return

        if isinstance(expr, (Variable, Literal)):
            return

        if isinstance(expr, MemberAccess):
            # validate object chain
            self._validate_expression(expr.object)
            return

        if isinstance(expr, UnaryOp):
            self._validate_expression(expr.operand)
            return

        if isinstance(expr, BinaryOp):
            self._validate_expression(expr.left)
            self._validate_expression(expr.right)
            return

        if isinstance(expr, FunctionCall):
            # kwargs: CSL-Core does not support keyword args (compiler is fail-closed)
            if getattr(expr, "kwargs", None):
                self.errors.append(ValidationError(
                    f"Keyword arguments are not supported in CSL-Core function calls: {expr.name}()",
                    getattr(expr, "location", None)
                ))

            if expr.name not in self._valid_functions:
                self.errors.append(ValidationError(
                    f"Unknown or unsupported function: '{expr.name}()'. Supported: {sorted(self._valid_functions)}",
                    getattr(expr, "location", None)
                ))

            for arg in (expr.args or []):
                self._validate_expression(arg)
            return

        # Unknown expression node => fail-closed (Core safety)
        self.errors.append(ValidationError(
            f"Unsupported expression type in CSL-Core: {expr.__class__.__name__}",
            getattr(expr, "location", None)
        ))
