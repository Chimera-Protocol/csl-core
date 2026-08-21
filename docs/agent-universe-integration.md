# CSL-Core v0.4.0 — Agent Universe Integration Reference

> Complete technical reference for integrating CSL-Core into a web application.
> Target feature: visual agent policy editor backed by real TLA+ model checking.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core API](#2-core-api)
3. [Language Pipeline](#3-language-pipeline)
4. [Z3 Verification Engine](#4-z3-verification-engine)
5. [TLA+ Formal Verification Engine](#5-tla-formal-verification-engine)
6. [CLI Commands](#6-cli-commands)
7. [MCP Server Tools](#7-mcp-server-tools)
8. [Plugins & Integrations](#8-plugins--integrations)
9. [CSL Syntax Reference](#9-csl-syntax-reference)
10. [Web Integration Patterns](#10-web-integration-patterns)

---

## 1. System Overview

```
.csl file
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  1. PARSER         Text → AST                                │
│     parse_csl(text) / parse_csl_file(path)                   │
├──────────────────────────────────────────────────────────────┤
│  2. VALIDATOR      Semantic checks (scope, types, functions) │
│     CSLValidator.validate(constitution)                      │
├──────────────────────────────────────────────────────────────┤
│  3. Z3 VERIFIER    Logical consistency — static analysis     │
│     LogicVerifier.verify(constitution)                       │
│     • Contradiction detection                                │
│     • Unreachable rule detection                             │
│     • Pairwise conflict analysis                             │
├──────────────────────────────────────────────────────────────┤
│  4. TLA⁺ VERIFIER  Temporal safety — model checking         │
│     TLAVerifier.verify(constitution)                         │
│     • Exhaustive state-space exploration via real TLC        │
│     • Predicate abstraction for large numeric domains        │
│     • Counterexample traces + automated fix suggestions      │
│     (opt-in: ENABLE_FORMAL_VERIFICATION: TRUE)               │
├──────────────────────────────────────────────────────────────┤
│  5. COMPILER       AST → Executable IR                       │
│     CSLCompiler.compile(constitution)                        │
│     CompiledConstitution (pickle-safe artifact)              │
├──────────────────────────────────────────────────────────────┤
│  6. RUNTIME        Deterministic enforcement <1ms            │
│     ChimeraGuard.verify(context) → GuardResult               │
└──────────────────────────────────────────────────────────────┘
```

**Key insight**: Stages 1–5 happen once at compile time. Stage 6 is pure evaluation with zero heavy dependencies.

---

## 2. Core API

### Installation

```bash
pip install csl-core==0.4.0
# With LangChain support:
pip install "csl-core[langchain]==0.4.0"
# With MCP server:
pip install "csl-core[mcp]==0.4.0"
```

### Public Exports (`chimera_core`)

```python
from chimera_core import (
    load_guard,                  # str/Path → ChimeraGuard
    create_guard_from_string,    # str → ChimeraGuard
    ChimeraGuard,                # Runtime enforcer
    GuardResult,                 # Verification result dataclass
    RuntimeConfig,               # Behavior toggles
    ChimeraError,                # Raised when blocked (raise_on_block=True)
    CSLCompiler,                 # Full compilation pipeline
    CompilationError,            # Raised on bad policy
)
```

### `load_guard`

```python
guard = load_guard(
    policy_path: Union[str, Path],
    config: Optional[RuntimeConfig] = None
) -> ChimeraGuard
```

### `create_guard_from_string`

```python
guard = create_guard_from_string(
    policy_content: str,           # Raw .csl text
    config: Optional[RuntimeConfig] = None
) -> ChimeraGuard
```

### `ChimeraGuard.verify`

```python
result: GuardResult = guard.verify(context: Dict[str, Any])
```

### `GuardResult`

```python
@dataclass
class GuardResult:
    allowed: bool
    violations: List[str]
    warnings: List[str]
    triggered_rule_ids: List[str]  # Rules whose WHEN condition matched (not necessarily violated)
    violated_rule_ids: List[str]   # Rules that actually produced a BLOCK violation — use this for audit/block-reason
    latency_ms: float
    domain_name: Optional[str]
    policy_name: Optional[str]     # Defaults to domain_name
    policy_id: Optional[str]       # From CONFIG.POLICY_ID, if set
    policy_version: Optional[str]  # From CONFIG.POLICY_VERSION, if set
    policy_hash: Optional[str]     # SHA-256 of the compiled policy's source text
    engine_version: Optional[str]
    enforcement: str               # "ACTIVE" | "DRY_RUN"
```

### `RuntimeConfig`

```python
@dataclass(frozen=True)
class RuntimeConfig:
    raise_on_block: bool = True
    collect_all_violations: bool = True
    missing_key_behavior: Literal["block", "warn", "ignore"] = "block"
    evaluation_error_behavior: Literal["block", "warn", "ignore"] = "block"
    dry_run: bool = False
```

### `ChimeraError`

```python
except ChimeraError as e:
    e.message            # str
    e.constraint_name    # str
    e.context            # Dict[str, Any]
    e.result             # GuardResult
```

---

## 3. Language Pipeline

### Parser

```python
from chimera_core.language.parser import parse_csl, parse_csl_file

constitution = parse_csl(csl_text: str) -> Constitution
constitution = parse_csl_file(filepath: str) -> Constitution
```

### AST Node Types

```
Constitution
├── Domain
│   └── VariableDeclaration[]    (name, domain_str)
├── Constraint[]
│   ├── ConditionClause          (temporal_operator, condition: Expression)
│   └── ActionClause             (modal_operator, variable, value: Expression)
└── Configuration
```

### Expressions

```python
Variable(name)                    # x
Literal(value)                    # 42, "ADMIN", True
BinaryOp(left, right, op)         # a > b, a AND b
UnaryOp(operand, op)              # NOT x
FunctionCall(name, args)          # len(x), max(a, b)
MemberAccess(object, member)      # user.role
```

### Temporal Operators

```python
class TemporalOperator(Enum):
    WHEN    = "WHEN"      # Point-in-time condition
    ALWAYS  = "ALWAYS"    # Invariant (all states)
```

### Modal Operators (THEN)

```python
class ModalOperator(Enum):
    MUST_BE     = "MUST BE"
    MUST_NOT_BE = "MUST NOT BE"
    MAY_BE      = "MAY BE"
    EQ, NEQ, LT, GT, LTE, GTE    # Numeric comparisons
```

### Compiler

```python
from chimera_core.language.compiler import CSLCompiler, CompiledConstitution

compiler = CSLCompiler()
compiled: CompiledConstitution = compiler.compile(constitution)

# Save/load artifact
compiled.save("policy.pkl")
loaded = CompiledConstitution.load("policy.pkl")
```

### `CompiledConstitution`

```python
@dataclass
class CompiledConstitution:
    domain_name: str
    constraints: List[CompiledConstraint]
    config: Any
    variable_domains: Dict[str, str]    # {"amount": "0..100000", "role": '{"ADMIN","USER"}'}
```

### `CompiledConstraint`

```python
@dataclass
class CompiledConstraint:
    name: str
    temporal_operator: TemporalOperator
    condition_expr: CompiledExpression
    action_variable: str
    modal_operator: ModalOperator
    action_value_expr: CompiledExpression
    enforcement_mode: EnforcementMode
    failure_message: str
    location: Optional[tuple]           # (line, col) in source
```

---

## 4. Z3 Verification Engine

### Usage

```python
from chimera_core.engines.z3_engine import LogicVerifier, SuggestionEngine

verifier = LogicVerifier()
is_valid, issues = verifier.verify(constitution, debug=False)
```

### `VerificationIssue`

```python
@dataclass(frozen=True)
class VerificationIssue:
    kind: str                           # CONTRADICTION | UNREACHABLE | UNSUPPORTED | INTERNAL_ERROR
    message: str
    rules: List[str]
    severity: str                       # "error" | "warning"
    model: Optional[Dict[str, Any]]     # Z3 counterexample values
    unsat_core: Optional[List[str]]     # Conflicting assertions
```

### What Z3 Checks

| Check | Description |
|-------|-------------|
| **Rule Reachability** | Each `WHEN` condition is satisfiable — not logically impossible |
| **Per-Rule Consistency** | condition ∧ action constraint is simultaneously satisfiable |
| **Pairwise Overlaps** | Co-triggering rules have compatible actions |
| **Policy-Wide** | All rules constraining the same variable are mutually compatible |

### `SuggestionEngine`

```python
suggester = SuggestionEngine()
suggester.report_issues(issues)    # Pretty-prints actionable feedback
```

---

## 5. TLA+ Formal Verification Engine

### Usage

```python
from chimera_core.engines.tla_engine import TLAVerifier

verifier = TLAVerifier(
    animate=True,            # Show terminal animations
    use_real_tlc=True,       # Use java -jar tla2tools.jar (auto-download)
    tlc_timeout=60,          # Subprocess timeout in seconds
    tlc_auto_download=True,  # Download JAR if not found
)

is_valid, issues = verifier.verify(constitution)
```

### `TLAIssue`

```python
class TLAIssue:
    kind: str                               # "SAFETY_VIOLATION"
    constraint: str                         # Constraint name
    message: str
    counterexample: Optional[List[Dict]]    # Violating state trace
```

### TLA+ Spec Builder

```python
from chimera_core.engines.tla_engine import TLASpecBuilder, TLASpecResult

builder = TLASpecBuilder()
spec: TLASpecResult = builder.build(constitution)

# spec.module_name    str
# spec.tla_content    str    (full .tla file content)
# spec.cfg_content    str    (full .cfg file content)
# spec.domain_info    List[Dict]   (variable domains + cardinalities)

tla_path, cfg_path = spec.write(Path("/tmp/myspec"))
```

### TLC Runner

```python
from chimera_core.engines.tla_engine import (
    TLCRunner, TLCResult, TLCViolation,
    java_available, find_jar, ensure_jar, run_tlc_on_spec,
)

# Check availability
if java_available() and find_jar():
    runner = TLCRunner(jar_path=None, auto_download=True, workers=1)
    result: TLCResult = runner.run(tla_path, cfg_path, timeout=60)
```

### `TLCResult`

```python
@dataclass
class TLCResult:
    success: bool
    violations: List[TLCViolation]
    states_explored: int
    distinct_states: int
    time_ms: int
    error: str
    tlc_output: str             # Raw TLC stdout+stderr
    used_real_tlc: bool
    # Identity proof fields (impossible to fake with Python BFS):
    tlc_version: str            # "TLC2 Version 2026.03.31.154134 (rev: becec35)"
    tlc_pid: int                # OS process ID of TLC JVM
    java_workers: int           # Number of TLC worker threads
```

### `TLCViolation`

```python
@dataclass
class TLCViolation:
    invariant: str                       # Constraint name
    state_vars: Dict[str, str]           # Violating state (raw TLC values)
    trace: List[Dict[str, str]]          # Ordered counterexample states
```

### Python BFS Mock (Fallback)

```python
from chimera_core.engines.tla_engine.model_checker import MockModelChecker

checker = MockModelChecker(max_states=5000, max_depth=50)
result = checker.check_safety(
    initial_state=initial,
    next_state_func=transitions,
    invariant=invariant_fn,
    property_name="my_constraint",
)
# result.result: CheckResult (VALID | VIOLATED | UNKNOWN | TIMEOUT)
# result.counterexample: CounterExample or None
# result.states_explored: int
```

### Predicate Abstraction

For large numeric ranges, TLA+ spec builder automatically reduces them to a finite abstract domain:

```
amount: 0..100000  with threshold 50000
→ abstract domain: {0, 49999, 50000, 50001, 100000}
```

This is sound for linear arithmetic: the truth value of `amount > T` is constant within each interval.

### Suggestion Engine (TLA+)

```python
from chimera_core.engines.tla_engine.suggestion_engine import TLASuggestionEngine

suggestion_engine = TLASuggestionEngine()
analysis = suggestion_engine.analyze(constraint, counterexample_dicts, constitution)

# analysis.constraint_name    str
# analysis.violation_state    Dict
# analysis.violation_vars     List[str]
# analysis.root_cause         str
# analysis.suggestions        List[ViolationSuggestion]
```

### `ViolationSuggestion`

```python
@dataclass
class ViolationSuggestion:
    title: str
    explanation: str
    fix_type: str               # DOMAIN_RESTRICTION | CONDITION_STRENGTHENING | GUARD_ADDITION | BOUND_TIGHTENING | POLICY_INVERSION
    confidence: str             # HIGH | MEDIUM | LOW
    before_snippet: str         # CSL code before fix
    after_snippet: str          # CSL code after fix
```

---

## 6. CLI Commands

### `cslcore verify`

```bash
cslcore verify <policy.csl> [--skip-verify] [--skip-validate] [--debug-z3]
```

Runs stages 1–5 (parse → validate → Z3 → compile). Fails loudly if any stage fails.

### `cslcore simulate`

```bash
cslcore simulate <policy.csl> \
  --input '{"amount": 5000, "role": "USER"}' \
  [--input-file test_cases.json] \
  [--dashboard] \
  [--dry-run] \
  [--json] \
  [--json-out results.jsonl] \
  [--quiet] \
  [--missing-key-behavior block|warn|ignore]
```

### `cslcore formal`

```bash
cslcore formal <policy.csl> \
  [--mock]              # Force Python BFS (no Java needed)
  [--timeout N]         # TLC subprocess timeout (default: 60s)
  [--no-download]       # Don't auto-download tla2tools.jar
```

### `cslcore repl`

```bash
cslcore repl <policy.csl> [--dashboard] [--dry-run]
# Interactive: type JSON, press Enter, see result
```

---

## 7. MCP Server Tools

```bash
pip install "csl-core[mcp]"
csl-core-mcp    # Start server
```

### Tools

| Tool | Signature | Description |
|------|-----------|-------------|
| `verify_policy` | `(csl_content: str) → str` | Parse + validate + Z3 verify + compile |
| `simulate_policy` | `(csl_content: str, context_json: str, dry_run: bool) → str` | Run against JSON input |
| `explain_policy` | `(csl_content: str) → str` | Human-readable policy summary |
| `scaffold_policy` | `(domain_name: str, description: str, variables: str) → str` | Generate CSL template |

### Built-in Resources (Example Policies)

```
policy://examples/hello_world
policy://examples/age_verification
policy://examples/banking_guard
policy://examples/agent_tool_guard
policy://examples/dao_treasury_guard
```

---

## 8. Plugins & Integrations

### LangChain

```python
from chimera_core.plugins.langchain import gate, ChimeraRunnableGate
from chimera_core import load_guard

guard = load_guard("policy.csl")

# In an LCEL chain
chain = (
    {"query": RunnablePassthrough()}
    | gate(guard, inject={"user_role": "USER"})
    | prompt | llm | StrOutputParser()
)

# Or wrap tools
from chimera_core.plugins.langchain import guard_tools
safe_tools = guard_tools(tools, guard=guard, inject={"env": "prod"})
```

### OpenClaw

```python
from chimera_core.plugins.openclaw import OpenClawGuard

guard = OpenClawGuard("policy.csl")
result = guard.evaluate(
    tool_name="transfer_funds",
    params={"amount": 5000, "currency": "USD"},
    metadata={"user_id": "u_123"}
)
# result.allowed: bool
# result.violations: List[str]
# result.latency_us: float
```

### Runtime Visualizer (Audit Dashboard)

```python
from chimera_core.audit.visualizer import RuntimeVisualizer

viz = RuntimeVisualizer()
viz.visualize(result, context, title="Payment Guard")
# Prints rich terminal dashboard: header, context table, violations, warnings
```

---

## 9. CSL Syntax Reference

### Full Policy Structure

```js
CONFIG {
  ENFORCEMENT_MODE: BLOCK           // BLOCK | WARN | LOG
  CHECK_LOGICAL_CONSISTENCY: TRUE   // Z3 verification (default: TRUE)
  ENABLE_FORMAL_VERIFICATION: FALSE // TLA+ model checking (default: FALSE)
}

DOMAIN DomainName {
  VARIABLES {
    amount:      0..100000                      // integer range
    score:       0.0..1.0                       // float range
    role:        {"ADMIN", "USER", "GUEST"}     // string enum
    approved:    BOOLEAN                        // boolean
  }

  STATE_CONSTRAINT constraint_name {
    WHEN  <condition>
    THEN  <variable> <modal_operator> <value>
  }

  STATE_CONSTRAINT always_holds {
    ALWAYS True
    THEN  <variable> <modal_operator> <value>
  }
}
```

### Condition Operators

```
==  !=  <  >  <=  >=
AND  OR  NOT
+  -  *  /  %
len(x)  max(a,b,...)  min(a,b,...)  abs(x)
```

### Action (THEN) Operators

```
MUST BE <value>       →  variable == value
MUST NOT BE <value>   →  variable != value
MAY BE <value>        →  always allowed
<= <value>            →  variable <= value  (and other comparison ops)
```

### Variable Domain Formats

```
integer range:  amount: 0..100000
float range:    confidence: 0.0..1.0
string enum:    role: {"ADMIN", "USER"}
boolean:        approved: BOOLEAN
```

---

## 10. Web Integration Patterns

### Pattern A: Verify policy from string (backend API)

```python
from chimera_core.language.parser import parse_csl
from chimera_core.language.compiler import CSLCompiler
from chimera_core.engines.z3_engine import LogicVerifier
from chimera_core.language.validator import CSLValidator

def verify_policy_api(csl_text: str) -> dict:
    try:
        constitution = parse_csl(csl_text)
        CSLValidator().validate(constitution)
        is_valid, issues = LogicVerifier().verify(constitution)
        return {
            "valid": is_valid,
            "issues": [{"kind": i.kind, "message": i.message, "rules": i.rules} for i in issues]
        }
    except Exception as e:
        return {"valid": False, "issues": [{"kind": "PARSE_ERROR", "message": str(e)}]}
```

### Pattern B: Run TLA+ verification and return results

```python
from chimera_core.language.parser import parse_csl
from chimera_core.engines.tla_engine import TLAVerifier

def run_formal_verification(csl_text: str, timeout: int = 60) -> dict:
    constitution = parse_csl(csl_text)
    verifier = TLAVerifier(animate=False, tlc_timeout=timeout)
    is_valid, issues = verifier.verify(constitution)
    return {
        "passed": is_valid,
        "violations": [
            {
                "constraint": i.constraint,
                "message": i.message,
                "counterexample": i.counterexample,
            }
            for i in issues
        ]
    }
```

### Pattern C: Get TLC result with identity proof

```python
from chimera_core.language.parser import parse_csl
from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
from chimera_core.engines.tla_engine.tlc_runner import run_tlc_on_spec

def run_tlc_raw(csl_text: str) -> dict:
    constitution = parse_csl(csl_text)
    spec = TLASpecBuilder().build(constitution)
    result = run_tlc_on_spec(spec, timeout=60)
    return {
        "success": result.success,
        "states_explored": result.states_explored,
        "time_ms": result.time_ms,
        "tlc_version": result.tlc_version,   # proof it's real TLC
        "tlc_pid": result.tlc_pid,
        "java_workers": result.java_workers,
        "violations": [
            {"invariant": v.invariant, "trace": v.trace}
            for v in result.violations
        ],
    }
```

### Pattern D: Get variable domain info for canvas

```python
from chimera_core.language.parser import parse_csl
from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder

def get_universe_info(csl_text: str) -> dict:
    constitution = parse_csl(csl_text)
    spec = TLASpecBuilder().build(constitution)
    return {
        "domain_name": spec.module_name,
        "variables": spec.domain_info,
        # domain_info items: {"name": str, "domain": str, "tla_set": str, "card": str}
        "constraints": [
            {"name": c.name, "condition": str(c.condition), "action": str(c.action)}
            for c in constitution.constraints
        ]
    }
```

### Pattern E: Runtime enforcement in web request handler

```python
from chimera_core import create_guard_from_string, RuntimeConfig, ChimeraError

guard = create_guard_from_string(policy_text, config=RuntimeConfig(
    raise_on_block=False,
    collect_all_violations=True,
    dry_run=False,
))

def handle_agent_action(action_context: dict) -> dict:
    result = guard.verify(action_context)
    return {
        "allowed": result.allowed,
        "violations": result.violations,
        "latency_ms": result.latency_ms,
    }
```

---

## Appendix: Key File Locations

| File | Purpose |
|------|---------|
| `chimera_core/__init__.py` | Public API exports, version |
| `chimera_core/cli.py` | CLI entry point (`cslcore`) |
| `chimera_core/runtime.py` | `ChimeraGuard`, `GuardResult`, `RuntimeConfig` |
| `chimera_core/factory.py` | `load_guard`, `create_guard_from_string` |
| `chimera_core/language/parser.py` | `parse_csl`, `parse_csl_file` |
| `chimera_core/language/compiler.py` | `CSLCompiler`, `CompiledConstitution` |
| `chimera_core/language/ast.py` | All AST node types and operator enums |
| `chimera_core/language/validator.py` | `CSLValidator` |
| `chimera_core/engines/z3_engine/verifier.py` | `LogicVerifier` |
| `chimera_core/engines/tla_engine/verifier.py` | `TLAVerifier` |
| `chimera_core/engines/tla_engine/tla_spec_builder.py` | `TLASpecBuilder`, `TLASpecResult` |
| `chimera_core/engines/tla_engine/tlc_runner.py` | `TLCRunner`, `TLCResult`, `run_tlc_on_spec` |
| `chimera_core/engines/tla_engine/suggestion_engine.py` | `TLASuggestionEngine`, `ViolationSuggestion` |
| `chimera_core/engines/tla_engine/model_checker.py` | `MockModelChecker` (BFS fallback) |
| `chimera_core/mcp/server.py` | MCP server + 4 tools |
| `chimera_core/plugins/langchain.py` | LangChain `gate`, `guard_tools` |
| `chimera_core/plugins/openclaw/guard.py` | `OpenClawGuard` |
| `chimera_core/audit/visualizer.py` | `RuntimeVisualizer` (terminal dashboard) |
