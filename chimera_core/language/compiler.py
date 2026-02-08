"""
CSL Compiler - AST to Executable IR

Compiles Chimera Specification Language AST into executable
intermediate representation (IR) that the runtime can evaluate.

Architecture:
1. Orchestration: Validates syntax, verifies logic (Z3), checks checks permissions.
2. Translation: Converts AST nodes into optimized Python Functors (Op objects).
3. IR Generation: Packages everything into a pickle-safe CompiledConstitution.
"""

import pickle
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field

# AST Imports
from .ast import (
    Constitution, Domain, Constraint, Expression, Variable, Literal, 
    BinaryOp, UnaryOp, FunctionCall, MemberAccess,
    TemporalOperator, ModalOperator, ArithmeticOperator, 
    ComparisonOperator, LogicalOperator, EnforcementMode
)

# Core Dependencies
from .validator import CSLValidator, ValidationError
from chimera_core.engines.z3_engine import LogicVerifier
from chimera_core.engines.z3_engine import SuggestionEngine


# ============================================================================
# PART 1: RUNTIME FUNCTORS (THE ENGINE)
# ============================================================================

class OpLiteral:
    """Returns a constant value."""
    def __init__(self, value): self.value = value
    def __call__(self, state): return self.value

class OpVariable:
    """Fetches a variable from the state dictionary."""
    def __init__(self, name): self.name = name
    def __call__(self, state): return state.get(self.name)

class OpMemberAccess:
    """Handles object.property access safely."""
    def __init__(self, obj_func, member_name):
        self.obj_func = obj_func
        self.member_name = member_name
    def __call__(self, state):
        obj = self.obj_func(state)
        if isinstance(obj, dict): return obj.get(self.member_name)
        return getattr(obj, self.member_name, None)

class OpBinary:
    """Executes binary operations (A > B, A + B)."""
    def __init__(self, left, right, op_func):
        self.left = left; self.right = right; self.op_func = op_func
    def __call__(self, state): return self.op_func(self.left(state), self.right(state))

class OpUnary:
    """Executes unary operations (NOT A, -A)."""
    def __init__(self, operand, op_func):
        self.operand = operand; self.op_func = op_func
    def __call__(self, state): return self.op_func(self.operand(state))

class OpFunctionCall:
    """Executes built-in function calls."""
    def __init__(self, func_name, args, kwargs):
        self.func_name = func_name; self.args = args; self.kwargs = kwargs
    def __call__(self, state):
        evaluated_args = [f(state) for f in self.args]
        return CSLCompiler._static_call_builtin(self.func_name, evaluated_args)

def _op_add(l, r): return l + r
def _op_sub(l, r): return l - r
def _op_mul(l, r): return l * r
def _op_div(l, r): return l / r if r != 0 else float('inf')
def _op_eq(l, r): return l == r
def _op_neq(l, r): return l != r
def _op_lt(l, r): return l < r
def _op_gt(l, r): return l > r
def _op_lte(l, r): return l <= r
def _op_gte(l, r): return l >= r
def _op_and(l, r): return l and r
def _op_or(l, r): return l or r
def _op_not(o): return not o
def _op_neg(o): return -o


# ============================================================================
# PART 2: INTERMEDIATE REPRESENTATION (IR) - DATA STRUCTURES
# ============================================================================

@dataclass
class CompiledExpression:
    """Compiled form of an expression."""
    ast: Expression
    eval_func: Callable
    
    def evaluate(self, state: Dict[str, Any]) -> Any:
        return self.eval_func(state)

@dataclass
class CompiledConstraint:
    """Compiled form of a constraint."""
    name: str
    temporal_operator: TemporalOperator
    condition_expr: CompiledExpression
    action_variable: str
    modal_operator: ModalOperator
    action_value_expr: CompiledExpression
    
    # Metadata for Auditing & Enforcement
    enforcement_mode: EnforcementMode = EnforcementMode.BLOCK
    failure_message: str = ""
    location: Optional[tuple] = None

@dataclass
class CompiledConstitution:
    """
    The final artifact produced by the compiler.
    This is what gets loaded into the Runtime Guard.
    """
    domain_name: str
    constraints: List[CompiledConstraint]
    config: Any = None
    variable_domains: Dict[str, str] = field(default_factory=dict)
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f: pickle.dump(self, f)
        
    @staticmethod
    def load(filepath: str) -> 'CompiledConstitution':
        with open(filepath, 'rb') as f: return pickle.load(f)


class CompilationError(Exception):
    """Raised when compilation fails due to logical or permission errors."""
    pass


# ============================================================================
# PART 3: THE COMPILER (ORCHESTRATOR)
# ============================================================================

class CSLCompiler:
    """
    The Orchestrator.
    Pipeline: Validation -> Logic Verification -> Feature Check -> IR Generation.
    """
    
    def __init__(self):
        self.validator = CSLValidator()
        self.verifier = LogicVerifier()
        self.suggester = SuggestionEngine()

    @staticmethod
    def load(filepath: str) -> CompiledConstitution:
        """One-shot compile from file."""
        from .parser import parse_csl_file
        constitution = parse_csl_file(filepath)
        compiler = CSLCompiler()
        return compiler.compile(constitution)

    def compile(self, constitution: Constitution) -> CompiledConstitution:
        """
        Main compilation pipeline.
        """
        domain_name = constitution.domain.name if constitution.domain else "Unknown"
        print(f"⚙️  Compiling Domain: {domain_name}")
        
        # ---------------------------------------------------------
        # STAGE 1: Syntax & Semantics Validation
        # ---------------------------------------------------------
        print("   • Validating Syntax...", end=" ")
        self.validator.validate(constitution)
        print("✅ OK")
        
        # ---------------------------------------------------------
        # STAGE 2: Logic Verification (Z3 - Core)
        # ---------------------------------------------------------
        # (Default: True)
        if constitution.config and constitution.config.check_logical_consistency:
            print("   ├── Verifying Logic Model (Z3 Engine)...", end=" ")
            is_valid, issues = self.verifier.verify(constitution)
            
            if not is_valid:
                print("\n❌ [CRITICAL] LOGIC VERIFICATION FAILED!")
                self.suggester.report_issues(issues)
                raise CompilationError("CSL Logic verification failed. Artifact generation aborted.")
            else:
                print("✅ Mathematically Consistent")

        # ---------------------------------------------------------
        # STAGE 3: Enterprise Feature Guard (The Upsell)
        # ---------------------------------------------------------
        self._check_enterprise_features(constitution)

        # ---------------------------------------------------------
        # STAGE 4: IR Generation (AST -> Functors)
        # ---------------------------------------------------------
        print("   • Generating IR...", end=" ")
        compiled_constraints = []
        
        # (Default: BLOCK)
        global_mode = EnforcementMode.BLOCK
        if constitution.config:
            global_mode = constitution.config.enforcement_mode

        for constraint in constitution.constraints:
            # 1. Expressions Compilation (Recursive)
            cond_compiled = self._compile_expr(constraint.condition.condition)
            act_val_compiled = self._compile_expr(constraint.action.value)
            
            # 2. Constraint Assembly
            compiled = CompiledConstraint(
                name=constraint.name,
                temporal_operator=constraint.condition.temporal_operator,
                condition_expr=cond_compiled,
                action_variable=constraint.action.variable,
                modal_operator=constraint.action.modal_operator,
                action_value_expr=act_val_compiled,
                enforcement_mode=global_mode, # Constraints can override this in future
                failure_message=f"Constraint '{constraint.name}' violated.",
                location=constraint.location
            )
            compiled_constraints.append(compiled)
        
        # Extract metadata
        var_domains = {}
        if constitution.domain and constitution.domain.variable_declarations:
            for decl in constitution.domain.variable_declarations:
                var_domains[decl.name] = decl.domain

        print("✅ OK")
        
        return CompiledConstitution(
            domain_name=domain_name,
            constraints=compiled_constraints,
            config=constitution.config,
            variable_domains=var_domains
        )

    def _check_enterprise_features(self, constitution: Constitution):
        """
        Checks for usage of Enterprise-only configuration flags.
        Raises error with explanation if detected.
        """
        if not constitution.config:
            return

        # CHECK 1: TLA+ Formal Verification
        if constitution.config.enable_formal_verification:
            print("\n" + "="*60)
            print("🔒 ENTERPRISE FEATURE LOCKED: TLA+ Formal Verification")
            print("-" * 60)
            print("You have enabled 'enable_formal_verification: true'.")
            print("This requires the CSL Enterprise Engine (Java/TLC Model Checker).")
            print("CSL Core uses Z3 (check_logical_consistency) for logic checks.")
            print("\n👉 To fix: Set 'enable_formal_verification: false' in your CSL config.")
            print("="*60 + "\n")
            raise CompilationError("Enterprise Feature Requested: Formal Verification")

        # CHECK 2: Causal Inference
        if constitution.config.enable_causal_inference:
            print("\n" + "="*60)
            print("🔒 ENTERPRISE FEATURE LOCKED: Causal Inference Engine")
            print("-" * 60)
            print("You have enabled 'enable_causal_inference: true'.")
            print("Advanced counterfactual analysis is available in CSL Enterprise.")
            print("\n👉 To fix: Set 'enable_causal_inference: false' in your CSL config.")
            print("="*60 + "\n")
            raise CompilationError("Enterprise Feature Requested: Causal Inference")

    # ========================================================================
    # HELPER: EXPRESSION COMPILATION (The Recursive Magic)
    # ========================================================================

    def _compile_expr(self, expr: Expression) -> CompiledExpression:
        """
        Recursively converts AST Nodes into optimized Runtime Functors.
        """
        eval_obj = self._expression_to_op(expr)
        return CompiledExpression(ast=expr, eval_func=eval_obj)

    def _expression_to_op(self, expr: Expression):
        # 1. Base Cases
        if isinstance(expr, Literal):
            return OpLiteral(expr.value)
        elif isinstance(expr, Variable):
            return OpVariable(expr.name)

        # 2. Structure Access
        elif isinstance(expr, MemberAccess):
            return OpMemberAccess(self._expression_to_op(expr.object), expr.member)

        # 3. Operations
        elif isinstance(expr, BinaryOp):
            return OpBinary(
                self._expression_to_op(expr.left),
                self._expression_to_op(expr.right),
                self._get_operator_func(expr.operator)
            )

        elif isinstance(expr, UnaryOp):
            return OpUnary(
                self._expression_to_op(expr.operand),
                self._get_operator_func(expr.operator)
            )

        # 4. Functions
        elif isinstance(expr, FunctionCall):
            if expr.kwargs:
                raise CompilationError(
                    f"Core does not support keyword arguments in function calls yet: {expr.name}({list(expr.kwargs.keys())})"
                )
            compiled_args = [self._expression_to_op(arg) for arg in expr.args]
            return OpFunctionCall(expr.name, compiled_args, {})

        # Unsupported expression type -> fail closed
        raise CompilationError(f"Unsupported expression type in CSL-Core: {expr.__class__.__name__}")

    def _get_operator_func(self, op):
        """Maps AST Operator Enums to actual Python functions."""
        if op == ArithmeticOperator.ADD: return _op_add
        if op == ArithmeticOperator.SUB: return _op_sub
        if op == ArithmeticOperator.MUL: return _op_mul
        if op == ArithmeticOperator.DIV: return _op_div
        if op == ComparisonOperator.EQ: return _op_eq
        if op == ComparisonOperator.NEQ: return _op_neq
        if op == ComparisonOperator.LT: return _op_lt
        if op == ComparisonOperator.GT: return _op_gt
        if op == ComparisonOperator.LTE: return _op_lte
        if op == ComparisonOperator.GTE: return _op_gte
        if op == LogicalOperator.AND: return _op_and
        if op == LogicalOperator.OR: return _op_or
        if op == LogicalOperator.NOT: return _op_not

        # Unsupported operator -> fail closed at compile-time
        raise CompilationError(f"Unsupported operator in CSL-Core: {op}")


    @staticmethod
    def _static_call_builtin(name, args):
        """Executes safe built-in functions."""
        if name == "len": return len(args[0]) if args else 0
        if name == "max": return max(args) if args else 0
        if name == "min": return min(args) if args else 0
        if name == "abs": return abs(args[0]) if args else 0
        # std, sigmoid etc. can be added here
        return None