"""
Abstract Syntax Tree (AST) for Chimera Specification Language

Defines the node types for CSL programs.
Each node represents a syntactic element of the language.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field, is_dataclass, fields


# ============================================================================
# BASE NODE
# ============================================================================

@dataclass(kw_only=True)
class ASTNode:
    """
    Base class for all AST nodes.
    
    Attributes:
        location: Source location (line, column) for error reporting
    """
    location: Optional[tuple] = None  # (line, column)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(...)"


# ============================================================================
# OPERATORS (Enums)
# ============================================================================

class ConstraintType(Enum):
    """Constraint types distinguishing between current state and next state"""
    STATE = "STATE_CONSTRAINT"  # Invariant (s)
    NEXT = "NEXT_CONSTRAINT"    # Transition (s')


class TemporalOperator(Enum):
    """Temporal operators for constraint conditions"""
    WHEN = "WHEN"           # Point-in-time condition
    BEFORE = "BEFORE"       # Precedence requirement
    AFTER = "AFTER"         # Postcondition
    ALWAYS = "ALWAYS"       # Invariant (all states)
    EVENTUALLY = "EVENTUALLY"  # Liveness (some future state)

class EnforcementMode(Enum):
    """Runtime enforcement behavior"""
    BLOCK = "BLOCK"  # Raise exception on violation
    WARN = "WARN"    # Log warning but allow action
    LOG = "LOG"      # Silent audit logging only

class ModalOperator(Enum):
    """Modal operators for deontic logic"""
    MUST_BE = "MUST BE"         # Obligation
    MUST_NOT_BE = "MUST NOT BE" # Prohibition
    MAY_BE = "MAY BE"           # Permission
    EQ = "=="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="


class ComparisonOperator(Enum):
    """Comparison operators"""
    EQ = "=="   # Equal
    NEQ = "!="  # Not equal
    LT = "<"    # Less than
    GT = ">"    # Greater than
    LTE = "<="  # Less than or equal
    GTE = ">="  # Greater than or equal


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class ArithmeticOperator(Enum):
    """Arithmetic operators"""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


class SetOperator(Enum):
    """Set operators for TLA+ compatibility"""
    IN = "∈"
    NOTIN = "∉"
    SUBSET = "⊆"
    UNION = "∪"
    INTERSECT = "∩"


class QuantifierType(Enum):
    """Quantifier types for TLA+ compatibility"""
    FORALL = "∀"
    EXISTS = "∃"


# ============================================================================
# EXPRESSIONS
# ============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for expressions"""
    pass


@dataclass
class Variable(Expression):
    """Variable reference"""
    name: str
    
    def __repr__(self):
        return f"Variable({self.name})"


@dataclass
class Literal(Expression):
    """Literal value (number, string, boolean)"""
    value: Union[int, float, str, bool]
    type: str  # "int", "float", "string", "bool"
    
    def __repr__(self):
        return f"Literal({self.value})"


@dataclass
class BinaryOp(Expression):
    """Binary operation (e.g., a + b, x < y)"""
    left: Expression
    operator: Union[ComparisonOperator, LogicalOperator, ArithmeticOperator]
    right: Expression
    
    def __repr__(self):
        return f"BinaryOp({self.left} {self.operator.value} {self.right})"


@dataclass
class UnaryOp(Expression):
    """Unary operation (e.g., NOT x, -y)"""
    operator: Union[LogicalOperator, ArithmeticOperator]
    operand: Expression
    
    def __repr__(self):
        return f"UnaryOp({self.operator.value} {self.operand})"


@dataclass
class FunctionCall(Expression):
    """Function call (e.g., std(price_change, window=24h))"""
    name: str
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)
    
    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"{self.name}({all_args})"


@dataclass
class MemberAccess(Expression):
    """Member access (e.g., price.change, proposal.status)"""
    object: Expression
    member: str
    
    def __repr__(self):
        return f"{self.object}.{self.member}"


@dataclass
class ArrayAccess(Expression):
    """Array indexing (e.g., arr[i], values[0])"""
    array: Expression
    index: Expression
    
    def __repr__(self):
        return f"{self.array}[{self.index}]"


@dataclass
class SetOperation(Expression):
    """Set operations (e.g., x ∈ S, A ∪ B)"""
    left: Expression
    operator: SetOperator
    right: Expression
    
    def __repr__(self):
        return f"({self.left} {self.operator.value} {self.right})"


@dataclass
class Conditional(Expression):
    """Conditional expression (e.g., IF cond THEN x ELSE y)"""
    condition: Expression
    then_branch: Expression
    else_branch: Expression
    
    def __repr__(self):
        return f"IF {self.condition} THEN {self.then_branch} ELSE {self.else_branch}"


@dataclass
class Quantifier(Expression):
    """Quantified expression (e.g., ∀x ∈ S : P(x))"""
    quantifier_type: QuantifierType
    variable: str
    domain: Expression
    body: Expression
    
    def __repr__(self):
        return f"{self.quantifier_type.value}{self.variable} ∈ {self.domain} : {self.body}"


@dataclass
class LetExpression(Expression):
    """Let binding (e.g., LET x == expr IN body)"""
    bindings: Dict[str, Expression]
    body: Expression
    
    def __repr__(self):
        bindings_str = ", ".join(f"{k} == {v}" for k, v in self.bindings.items())
        return f"LET {bindings_str} IN {self.body}"

# ============================================================================
# VARIABLE DECLARATIONS
# ============================================================================

@dataclass
class VariableDeclaration(ASTNode):
    """
    Variable declaration with domain.
    
    Examples:
        price: 0..100000
        action: {"BUY", "SELL", "HOLD"}
        balance: Int
    """
    name: str
    domain: str  # TLA+ domain specification (e.g., "0..100", '{"A", "B"}', "Nat")
    
    def __repr__(self):
        return f"{self.name}: {self.domain}"



# ============================================================================
# CAUSAL MODEL
# ============================================================================

@dataclass
class CausalEdge(ASTNode):
    """
    Causal edge: source → target (Directed) OR source <-> target (Bidirected/Confounder)
    """
    source: str
    target: str
    edge_type: str = "directed"  # "directed" or "bidirected"
    mechanism: Optional[str] = None
    
    def __repr__(self):
        arrow = "↔" if self.edge_type == "bidirected" else "→"
        mech = f" ({self.mechanism})" if self.mechanism else ""
        return f"{self.source} {arrow} {self.target}{mech}"


@dataclass
class CausalGraph(ASTNode):
    """
    Causal graph structure.
    
    Defines the causal relationships between variables in the domain.
    """
    edges: List[CausalEdge] = field(default_factory=list)
    
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (direct causes) of a given node"""
        return [edge.source for edge in self.edges if edge.target == node]
    
    def get_children(self, node: str) -> List[str]:
        """Get child nodes (direct effects) of a given node"""
        return [edge.target for edge in self.edges if edge.source == node]
    
    def __repr__(self):
        return f"CausalGraph({len(self.edges)} edges)"


@dataclass
class StructuralEquation(ASTNode):
    """
    Structural equation: variable = expression
    
    Defines how a variable is computed from its causal parents.
    Example: volatility = std(price_change, window=24h)
    """
    variable: str
    expression: Expression
    
    def __repr__(self):
        return f"{self.variable} = {self.expression}"


# ============================================================================
# FORMAL MODEL
# ============================================================================

@dataclass
class Invariant(ASTNode):
    """
    Invariant (safety property)
    
    Must hold in ALL states.
    Example: ∀t: position(t) ∈ [0, MAX_POSITION]
    """
    name: str
    formula: Expression
    tla_spec: Optional[str] = None  # Optional TLA+ specification
    
    def __repr__(self):
        return f"Invariant({self.name}: {self.formula})"


@dataclass
class LivenessProperty(ASTNode):
    """
    Liveness property (progress property)
    
    Must EVENTUALLY hold in some future state.
    Example: ◇(action = BUY)
    """
    name: str
    formula: Expression
    tla_spec: Optional[str] = None
    
    def __repr__(self):
        return f"Liveness({self.name}: {self.formula})"


# ============================================================================
# DOMAIN
# ============================================================================

@dataclass
class Domain(ASTNode):
    """
    Domain declaration
    
    Defines the problem domain with causal and formal models.
    """
    name: str
    variable_declarations: List[VariableDeclaration] = field(default_factory=list)
    causal_graph: Optional[CausalGraph] = None
    structural_equations: List[StructuralEquation] = field(default_factory=list)
    invariants: List[Invariant] = field(default_factory=list)
    liveness_properties: List[LivenessProperty] = field(default_factory=list)
    
    def get_variable_domain(self, var_name: str) -> Optional[str]:
        """Get domain for variable"""
        for var_decl in self.variable_declarations:
            if var_decl.name == var_name:
                return var_decl.domain
        return None
    
    def __repr__(self):
        return f"Domain({self.name})"


# ============================================================================
# CONSTRAINT COMPONENTS
# ============================================================================

@dataclass
class ConditionClause(ASTNode):
    """
    Condition clause (WHEN/BEFORE/AFTER/ALWAYS/EVENTUALLY)
    
    Specifies when the constraint applies.
    """
    temporal_operator: TemporalOperator
    condition: Expression
    
    def __repr__(self):
        return f"{self.temporal_operator.value} {self.condition}"


@dataclass
class ActionClause(ASTNode):
    """
    Action clause (THEN)
    
    Specifies what must/must not happen.
    """
    variable: str
    modal_operator: ModalOperator
    value: Expression
    
    def __repr__(self):
        return f"THEN {self.variable} {self.modal_operator.value} {self.value}"


@dataclass
class CounterfactualStatement(ASTNode):
    """
    Counterfactual statement
    
    IF action | condition THEN outcome
    """
    intervention: Dict[str, Any]  # {"action": "SELL"}
    condition: Dict[str, Any]     # {"price_drop": True}
    outcome: Dict[str, Any]       # {"regret_probability": 0.67}
    data_source: Optional[str] = None
    
    def __repr__(self):
        return f"IF {self.intervention} | {self.condition} THEN {self.outcome}"


@dataclass
class IdentificationSpec(ASTNode):
    """
    Identification specification logic.
    Example: IDENTIFICATION { METHOD: BACKDOOR, ADJUSTMENT: {Gas, Vol} }
    """
    method: str  # "BACKDOOR", "FRONTDOOR", etc.
    variables: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"Identification({self.method}, vars={self.variables})"

@dataclass
class CausalProof(ASTNode):
    """
    Causal proof clause with Identification support.
    """
    mechanism: List[str] = field(default_factory=list)
    counterfactuals: List[CounterfactualStatement] = field(default_factory=list)
    identification: Optional[IdentificationSpec] = None  # Updated type
    confidence: float = 1.0


@dataclass
class TLASpec(ASTNode):
    """TLA+ specification"""
    property_name: str
    formula: str  # TLA+ temporal logic formula
    
    def __repr__(self):
        return f"{self.property_name} == {self.formula}"


@dataclass
class ModelCheckingResult(ASTNode):
    """Model checking result metadata"""
    states_explored: int = 0
    violations_found: int = 0
    deadlocks_found: int = 0
    time_elapsed_ms: int = 0
    
    def __repr__(self):
        return f"ModelChecking(states={self.states_explored})"


@dataclass
class FormalProof(ASTNode):
    """
    Formal proof clause
    
    Provides formal verification via TLA+.
    """
    tla_spec: List[TLASpec] = field(default_factory=list)
    model_checking: Optional[ModelCheckingResult] = None
    
    def __repr__(self):
        return f"FormalProof({len(self.tla_spec)} properties)"


@dataclass
class EnforcementClause(ASTNode):
    """
    Enforcement clause
    
    Specifies what to do when constraint is violated.
    """
    default_action: str
    notify: Optional[str] = None
    override_requires: Optional[Expression] = None
    
    def __repr__(self):
        return f"Enforcement(default={self.default_action})"


# ============================================================================
# CONSTRAINT
# ============================================================================

@dataclass
class Constraint(ASTNode):
    """
    Constraint definition
    
    Main unit of constitutional specification.
    """
    name: str
    constraint_type: ConstraintType
    condition: ConditionClause
    action: ActionClause
    causal_proof: Optional[CausalProof] = None
    formal_proof: Optional[FormalProof] = None
    enforcement: Optional[EnforcementClause] = None
    
    def __repr__(self):
        return f"Constraint({self.name})"


# ============================================================================
# CONSTITUTION (Top-level)
# ============================================================================

@dataclass
class Configuration(ASTNode):
    """
    Configuration block for compiler/runtime behavior.
    """
    # Runtime Behavior
    enforcement_mode: EnforcementMode = field(default=EnforcementMode.BLOCK)
    integration: str = "native"
    
    # Verification Engines
    check_logical_consistency: bool = True  # Z3 (Core Feature) - Default ON
    enable_formal_verification: bool = False     # TLA+ (Enterprise Feature) - Default OFF
    
    # Advanced Settings
    enable_causal_inference: bool = False
    optimize_verification_scope: bool = False
    
    def __repr__(self):
        return f"Config(mode={self.enforcement_mode.value}, logic={self.check_logical_consistency}, model={self.enable_formal_verification})"

@dataclass
class Constitution(ASTNode):
    """
    Top-level CSL program
    
    Contains domain definition and constraints.
    """
    domain: Optional[Domain] = None
    config: Optional[Configuration] = None
    constraints: List[Constraint] = field(default_factory=list)
    causal_graph: Optional[CausalGraph] = None
    
    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get constraint by name"""
        for c in self.constraints:
            if c.name == name:
                return c
        return None
    
    def __repr__(self):
        return f"Constitution({self.domain.name}, {len(self.constraints)} constraints)"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def visit_ast(node: ASTNode, visitor_func, _seen=None):
    """
    Visit all nodes in AST tree in a deterministic, dataclass-safe way.
    """
    if _seen is None:
        _seen = set()

    node_id = id(node)
    if node_id in _seen:
        return
    _seen.add(node_id)

    visitor_func(node)

    if not is_dataclass(node):
        return

    for f in fields(node):
        if f.name == "location":
            continue
        attr = getattr(node, f.name)

        if isinstance(attr, ASTNode):
            visit_ast(attr, visitor_func, _seen)
        elif isinstance(attr, list):
            for item in attr:
                if isinstance(item, ASTNode):
                    visit_ast(item, visitor_func, _seen)
        elif isinstance(attr, dict):
            for item in attr.values():
                if isinstance(item, ASTNode):
                    visit_ast(item, visitor_func, _seen)


def ast_to_dict(node: ASTNode) -> Dict[str, Any]:
    """
    Convert AST node to dictionary (for JSON serialization).
    
    Args:
        node: AST node
        
    Returns:
        Dictionary representation
    """
    result = {"type": node.__class__.__name__}
    
    for attr_name, attr_value in node.__dict__.items():
        if attr_name == "location":
            result[attr_name] = attr_value
            continue
        
        if isinstance(attr_value, ASTNode):
            result[attr_name] = ast_to_dict(attr_value)
        elif isinstance(attr_value, list):
            result[attr_name] = [
                ast_to_dict(item) if isinstance(item, ASTNode) else item
                for item in attr_value
            ]
        elif isinstance(attr_value, dict):
            result[attr_name] = {
                k: ast_to_dict(v) if isinstance(v, ASTNode) else v
                for k, v in attr_value.items()
            }
        elif isinstance(attr_value, Enum):
            result[attr_name] = attr_value.value
        else:
            result[attr_name] = attr_value
    
    return result


def pretty_print_ast(node: ASTNode, indent: int = 0) -> str:
    """
    Pretty print AST for debugging.
    
    Args:
        node: AST node
        indent: Indentation level
        
    Returns:
        Formatted string
    """
    prefix = "  " * indent
    lines = [f"{prefix}{node.__class__.__name__}"]
    
    for attr_name, attr_value in node.__dict__.items():
        if attr_name == "location" or attr_value is None:
            continue
        
        if isinstance(attr_value, ASTNode):
            lines.append(f"{prefix}  {attr_name}:")
            lines.append(pretty_print_ast(attr_value, indent + 2))
        elif isinstance(attr_value, list) and attr_value:
            lines.append(f"{prefix}  {attr_name}: [")
            for item in attr_value:
                if isinstance(item, ASTNode):
                    lines.append(pretty_print_ast(item, indent + 2))
                else:
                    lines.append(f"{prefix}    {item}")
            lines.append(f"{prefix}  ]")
        elif isinstance(attr_value, Enum):
            lines.append(f"{prefix}  {attr_name}: {attr_value.value}")
        elif not isinstance(attr_value, dict):
            lines.append(f"{prefix}  {attr_name}: {attr_value}")
    
    return "\n".join(lines)