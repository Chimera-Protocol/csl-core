"""
CSL-Core MCP Server v2
=======================
Exposes CSL-Core's compile-time verification and runtime enforcement
to any MCP-compatible client (Claude Desktop, Cursor, VS Code, etc.)

Tools:
    - verify_policy:   Parse + validate + Z3 verify a CSL policy
    - simulate_policy: Run a policy against JSON input(s)
    - explain_policy:  Human-readable summary of a policy's structure
    - scaffold_policy: Generate a CSL policy template from description
    - tla_verify:      TLA+ formal verification with full TLC results
    - universe_info:   State space analysis for a CSL policy

Resources:
    - policy://examples/* : Built-in example policies

Prompts:
    - csl_expert: CSL syntax reference and common patterns
"""

import json
import io
import math
from contextlib import redirect_stdout
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# CSL-Core imports (package: chimera_core)
from chimera_core.language.parser import parse_csl
from chimera_core.language.compiler import CSLCompiler, CompilationError
from chimera_core.language.validator import CSLValidator
from chimera_core.runtime import ChimeraGuard, RuntimeConfig, GuardResult
from chimera_core.language.ast import TemporalOperator, ModalOperator

# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------

mcp = FastMCP("csl-core")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Package-bundled examples (works from PyPI install)
_PACKAGE_EXAMPLES = Path(__file__).resolve().parent / "examples"

# Project root examples (works in dev mode)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_EXAMPLES = _PROJECT_ROOT / "examples"
_REPO_QUICKSTART = _PROJECT_ROOT / "quickstart"


def _compile_silent(csl_content: str):
    """
    Parse + compile CSL content while capturing stdout.

    The compiler prints progress messages to stdout which would corrupt
    MCP's JSON-RPC over STDIO transport.

    Returns: (compiled, compiler_output, error)
    """
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ast = parse_csl(csl_content)
            compiler = CSLCompiler()
            compiled = compiler.compile(ast)
        return compiled, buf.getvalue(), None
    except Exception as e:
        return None, buf.getvalue(), e


def _parse_silent(csl_content: str):
    """Parse only (no compile), with stdout capture."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ast = parse_csl(csl_content)
        return ast, None
    except Exception as e:
        return None, e


def _get_error_info(err) -> dict:
    """Extract line/column/message from various CSL error types."""
    info = {
        "type": type(err).__name__,
        "message": str(err),
        "line": getattr(err, "line", None) or getattr(err, "lineno", None),
        "column": getattr(err, "column", None) or getattr(err, "col", None),
    }
    loc = getattr(err, "location", None)
    if loc and isinstance(loc, (tuple, list)) and len(loc) >= 2:
        info["line"] = info["line"] or loc[0]
        info["column"] = info["column"] or loc[1]
    return info


def _format_verification_issues(issues: list) -> str:
    """Convert LogicVerifier issue dicts into LLM-friendly Markdown."""
    if not issues:
        return ""

    sections = []

    for i, issue in enumerate(issues, 1):
        kind = issue.get("kind", "UNKNOWN")
        severity = issue.get("severity", "error")
        message = issue.get("message", "")
        rules = issue.get("rules", []) or []
        model = issue.get("model")
        unsat_core = issue.get("unsat_core")
        meta = issue.get("meta")

        if kind == "COVERAGE":
            if meta:
                sections.append(
                    f"**Verification Coverage:** "
                    f"{meta.get('total_constraints', '?')} constraints, "
                    f"{meta.get('analyzed_pairs', '?')} pairs analyzed, "
                    f"{meta.get('skipped_pairs_unsupported', 0)} skipped"
                )
            continue

        icon = "\u274c" if severity == "error" else "\u26a0\ufe0f"
        header = f"{icon} **{kind}** (#{i})"
        if rules:
            header += f" \u2014 Rules: `{'`, `'.join(rules)}`"

        parts = [header, message]

        if model:
            parts.append("\n**Example triggering state:**")
            for k in sorted(model.keys()):
                parts.append(f"  - `{k}` = `{model[k]}`")

        if unsat_core:
            parts.append("\n**UNSAT Core (conflicting assumptions):**")
            for idx, lit in enumerate(unsat_core, 1):
                parts.append(f"  {idx}. `{lit}`")

        if kind == "CONTRADICTION":
            parts.append("\n**How to fix:**")
            parts.append("- Make the conflicting rules mutually exclusive (add a discriminating predicate to WHEN)")
            parts.append("- Or align their THEN actions so both can be satisfied simultaneously")
            parts.append("- The UNSAT core above shows exactly which assumptions conflict")
        elif kind == "UNREACHABLE":
            parts.append("\n**How to fix:**")
            parts.append("- Check VARIABLES domain bounds \u2014 the WHEN condition may be impossible given the declared ranges")
            parts.append("- Look for contradictory conjunctions (e.g., `x > 5 AND x < 3`)")
        elif kind == "UNSUPPORTED":
            parts.append("\n**How to fix:**")
            parts.append("- CSL-Core supports WHEN/ALWAYS with boolean/arithmetic comparisons")
            parts.append("- Remove unsupported temporal operators (BEFORE/AFTER/EVENTUALLY are Enterprise)")

        sections.append("\n".join(parts))

    return "\n\n---\n\n".join(sections)


def _format_guard_result(result: GuardResult) -> str:
    """Format a GuardResult into LLM-friendly Markdown."""
    lines = []

    if result.allowed:
        lines.append("\u2705 **ALLOWED**")
    else:
        lines.append("\u274c **BLOCKED**")

    if result.violations:
        lines.append(f"\n**Violations ({len(result.violations)}):**")
        for v in result.violations:
            lines.append(f"- {v}")

    if result.warnings:
        lines.append(f"\n**Warnings ({len(result.warnings)}):**")
        for w in result.warnings:
            lines.append(f"- {w}")

    if result.violated_rule_ids:
        lines.append(f"\n**Violated rules:** `{'`, `'.join(result.violated_rule_ids)}`")

    if result.triggered_rule_ids:
        lines.append(f"\n**Triggered rules (condition matched, not necessarily violated):** `{'`, `'.join(result.triggered_rule_ids)}`")

    lines.append(f"\n**Latency:** {result.latency_ms:.3f}ms")

    if result.enforcement == "DRY_RUN":
        lines.append("**Mode:** DRY_RUN (violations logged but not enforced)")

    return "\n".join(lines)


def _explain_ast(ast) -> str:
    """Walk AST and produce a Markdown summary."""
    lines = []

    if ast.domain:
        lines.append(f"## Domain: `{ast.domain.name}`")

        if ast.domain.variable_declarations:
            lines.append(f"\n### Variables ({len(ast.domain.variable_declarations)})")
            for v in ast.domain.variable_declarations:
                domain_str = str(v.domain)
                if ".." in domain_str:
                    vtype = "range"
                elif domain_str.startswith("{"):
                    vtype = "enum"
                else:
                    vtype = "primitive"
                lines.append(f"- `{v.name}`: `{domain_str}` ({vtype})")

    if ast.config:
        lines.append(f"\n### Configuration")
        lines.append(f"- Enforcement mode: `{ast.config.enforcement_mode.value}`")
        lines.append(f"- Z3 logical consistency: `{ast.config.check_logical_consistency}`")
        if ast.config.enable_formal_verification:
            lines.append(f"- TLA+ formal verification: `True`")
        if ast.config.enable_causal_inference:
            lines.append(f"- Causal inference: `True`")

    if ast.constraints:
        lines.append(f"\n### Constraints ({len(ast.constraints)})")
        for c in ast.constraints:
            temporal = c.condition.temporal_operator.value
            modal = c.action.modal_operator.value
            action_var = c.action.variable

            lines.append(f"\n**`{c.name}`**")
            lines.append(f"- Trigger: `{temporal} ...`")
            lines.append(f"- Action: `{action_var}` `{modal}` ...")

            if temporal == "ALWAYS":
                lines.append(f"- Behavior: Fires on every input (invariant)")
            elif temporal == "WHEN":
                lines.append(f"- Behavior: Fires conditionally")

    return "\n".join(lines)


def _load_example_file(name: str) -> str:
    """
    Load a .csl example. Priority:
    1. Package-bundled (chimera_core/mcp/examples/) \u2014 works from PyPI
    2. Repo examples/ and quickstart/ \u2014 works in dev mode
    """
    # 1. Package-bundled (always available after install)
    bundled = _PACKAGE_EXAMPLES / f"{name}.csl"
    if bundled.exists():
        return bundled.read_text(encoding="utf-8")

    # 2. Repo directories (dev mode fallback)
    for directory in [_REPO_EXAMPLES, _REPO_QUICKSTART]:
        filepath = directory / f"{name}.csl"
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")

    # 3. Fuzzy match in quickstart (01_hello_world.csl format)
    if _REPO_QUICKSTART.exists():
        for f in _REPO_QUICKSTART.iterdir():
            if f.suffix == ".csl" and name in f.stem:
                return f.read_text(encoding="utf-8")

    # Not found \u2014 list available
    available = []
    if _PACKAGE_EXAMPLES.exists():
        available.extend(f.stem for f in _PACKAGE_EXAMPLES.iterdir() if f.suffix == ".csl")
    for directory in [_REPO_EXAMPLES, _REPO_QUICKSTART]:
        if directory.exists():
            available.extend(f.stem for f in directory.iterdir() if f.suffix == ".csl")

    return f"Example '{name}' not found. Available: {', '.join(sorted(set(available)))}"


def _extract_variable_names(expr, found=None) -> set:
    """Recursively extract variable names from an AST expression node."""
    if found is None:
        found = set()
    if expr is None:
        return found
    # Variable node: has .name but is not a function call
    if hasattr(expr, "name") and not hasattr(expr, "args"):
        found.add(expr.name)
    for attr in ("left", "right", "operand"):
        child = getattr(expr, attr, None)
        if child is not None:
            _extract_variable_names(child, found)
    if hasattr(expr, "args") and isinstance(getattr(expr, "args"), (list, tuple)):
        for arg in expr.args:
            _extract_variable_names(arg, found)
    return found


# ---------------------------------------------------------------------------
# Tools: Core (verify, simulate, explain, scaffold)
# ---------------------------------------------------------------------------

@mcp.tool()
def verify_policy(csl_content: str) -> str:
    """
    Verify a CSL policy for logical consistency using Z3 formal verification.

    Performs four-stage analysis:
    1. Syntax validation (parser)
    2. Semantic validation (scope, types, function whitelist)
    3. Z3 logic verification (reachability, internal consistency, pairwise conflicts, policy-wide conflicts)
    4. IR compilation

    Returns verification result with actionable error details if any issues are found.

    Args:
        csl_content: The complete CSL policy source code as a string.
    """
    # Stage 1: Parse
    ast, parse_err = _parse_silent(csl_content)
    if parse_err:
        info = _get_error_info(parse_err)
        loc = f"Line {info['line']}, Column {info['column']}: " if info["line"] else ""
        return (
            f"\u274c **PARSE ERROR**\n\n"
            f"{loc}{info['message']}\n\n"
            f"**How to fix:** Check CSL syntax at the indicated location. "
            f"Common issues: missing braces, unquoted strings, typos in keywords."
        )

    # Stage 2-4: Compile (includes validation + Z3 verification)
    compiled, compiler_output, compile_err = _compile_silent(csl_content)

    if compile_err:
        info = _get_error_info(compile_err)

        # If it's a compilation error, try to get Z3 verification details
        if isinstance(compile_err, CompilationError):
            try:
                from chimera_core.engines.z3_engine.verifier import LogicVerifier
                verifier = LogicVerifier()
                is_valid, issues = verifier.verify(ast)
                if issues:
                    error_issues = [i for i in issues if i.get("severity") == "error"]
                    if error_issues:
                        return (
                            f"\u274c **LOGIC VERIFICATION FAILED**\n\n"
                            f"{_format_verification_issues(issues)}"
                        )
            except Exception:
                pass

            return f"\u274c **COMPILATION ERROR**\n\n{info['message']}"

        # Validation error
        loc = f"Line {info['line']}, Column {info['column']}: " if info["line"] else ""
        return (
            f"\u274c **VALIDATION ERROR**\n\n"
            f"{loc}{info['message']}\n\n"
            f"**How to fix:** Ensure all variables are declared in VARIABLES. "
            f"Supported functions: `len`, `max`, `min`, `abs`."
        )

    # Success
    domain = getattr(compiled, "domain_name", "Unknown")
    num_constraints = len(getattr(compiled, "constraints", []))
    num_vars = len(getattr(compiled, "variable_domains", {}))

    return (
        f"\u2705 **VERIFIED** \u2014 Domain `{domain}` is mathematically consistent.\n\n"
        f"- **Constraints:** {num_constraints}\n"
        f"- **Variables:** {num_vars}\n"
        f"- **Z3 result:** No logical contradictions found\n"
        f"- **Artifact:** Compiled IR ready for runtime enforcement"
    )


@mcp.tool()
def simulate_policy(
    csl_content: str,
    context_json: str,
    dry_run: bool = False,
) -> str:
    """
    Simulate a CSL policy against one or more JSON inputs.

    Compiles the policy, then runs the runtime guard against the provided context.
    Returns ALLOWED or BLOCKED with full violation details.

    Supports batch simulation: pass a JSON array of objects to test multiple inputs.

    Args:
        csl_content: The complete CSL policy source code as a string.
        context_json: JSON object (single input) or JSON array (batch) to test.
        dry_run: If true, evaluates all rules but never blocks. Useful for shadow testing.
    """
    # 1. Compile
    compiled, _, compile_err = _compile_silent(csl_content)
    if compile_err:
        return f"\u274c **COMPILATION FAILED** \u2014 Cannot simulate.\n\n{type(compile_err).__name__}: {compile_err}"

    # 2. Parse input(s)
    try:
        parsed = json.loads(context_json)
    except json.JSONDecodeError as e:
        return f"\u274c **INVALID JSON**\n\nCannot parse context_json: {e}"

    # Normalize to list
    if isinstance(parsed, dict):
        inputs = [parsed]
    elif isinstance(parsed, list):
        inputs = parsed
    else:
        return "\u274c **INVALID INPUT** \u2014 context_json must be a JSON object or array of objects."

    # 3. Create guard
    config = RuntimeConfig(
        raise_on_block=False,
        collect_all_violations=True,
        dry_run=dry_run,
    )
    guard = ChimeraGuard(compiled, config)

    # 4. Simulate each input
    results = []
    for i, ctx in enumerate(inputs):
        if not isinstance(ctx, dict):
            results.append(f"### Input #{i+1}\n\u274c **INVALID** \u2014 Expected JSON object, got {type(ctx).__name__}")
            continue

        result = guard.verify(ctx)
        if len(inputs) > 1:
            results.append(f"### Input #{i+1}\n{_format_guard_result(result)}")
        else:
            results.append(_format_guard_result(result))

    # Batch summary
    if len(inputs) > 1:
        allowed_count = 0
        for ctx in inputs:
            if isinstance(ctx, dict):
                r = guard.verify(ctx)
                if r.allowed:
                    allowed_count += 1
        blocked_count = len(inputs) - allowed_count
        results.append(
            f"\n---\n**Batch Summary:** {len(inputs)} inputs \u2014 "
            f"{allowed_count} allowed, {blocked_count} blocked"
        )

    return "\n\n".join(results)


@mcp.tool()
def explain_policy(csl_content: str) -> str:
    """
    Parse a CSL policy and return a structured Markdown summary.

    Shows: domain name, all variables with types/ranges, all constraints
    with triggers and actions, and configuration settings.
    Does NOT compile or verify \u2014 use verify_policy for that.

    Args:
        csl_content: The complete CSL policy source code as a string.
    """
    ast, err = _parse_silent(csl_content)
    if err:
        info = _get_error_info(err)
        loc = f"Line {info['line']}, Column {info['column']}: " if info["line"] else ""
        return f"\u274c **PARSE ERROR**\n\n{loc}{info['message']}"

    return _explain_ast(ast)


@mcp.tool()
def scaffold_policy(
    domain_name: str,
    description: str,
    variables: str = "",
) -> str:
    """
    Generate a CSL policy scaffold from a description.

    Returns a ready-to-edit .csl template with CONFIG, DOMAIN, VARIABLES,
    and placeholder constraints.

    Common CSL patterns:
        WHEN amount > 1000 THEN role MUST BE "ADMIN"
        WHEN risk_score > 0.8 THEN action MUST NOT BE "TRANSFER"
        ALWAYS True THEN tool MUST NOT BE "DELETE"
        WHEN user_age < 18 AND category == "ALCOHOL" THEN allowed MUST BE "NO"

    Variable types:
        amount: 0..100000        (integer range)
        role: {"ADMIN", "USER"}  (enum / string set)
        score: 0..1              (numeric range)

    Args:
        domain_name: Name for the policy domain (e.g., "PaymentGuard", "AgentSafety").
        description: Plain-English description of what the policy should enforce.
        variables: Optional comma-separated variable hints (e.g., "amount, role, risk_score").
    """
    var_lines = ""
    if variables:
        var_hints = [v.strip() for v in variables.split(",") if v.strip()]
        var_lines = "\n".join(
            f"    {v}: // TODO: define type (e.g., 0..1000 or {{\"A\", \"B\"}})"
            for v in var_hints
        )
    else:
        var_lines = "    // TODO: declare your variables here"

    return f"""```csl
// CSL Policy: {description}
// Generated scaffold \u2014 edit and verify with verify_policy

CONFIG {{
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}}

DOMAIN {domain_name} {{

  VARIABLES {{
{var_lines}
  }}

  // TODO: Add your constraints
  // Example:
  // STATE_CONSTRAINT example_rule {{
  //   WHEN amount > 1000
  //   THEN role MUST BE "ADMIN"
  // }}
}}
```

**Next steps:**
1. Define variable types in the VARIABLES block
2. Write STATE_CONSTRAINT rules using WHEN/THEN
3. Run `verify_policy` to check for logical consistency
4. Run `simulate_policy` with test JSON to validate behavior"""


# ---------------------------------------------------------------------------
# Tools: TLA+ Verification & Universe Analysis (NEW in v2)
# ---------------------------------------------------------------------------

@mcp.tool()
def tla_verify(
    csl_content: str,
    timeout: int = 60,
    use_mock: bool = False,
) -> str:
    """
    Run TLA+ formal verification (real TLC model checking) on a CSL policy.

    Performs exhaustive state-space exploration to verify temporal safety
    properties. Unlike Z3 (which checks static logical consistency),
    TLA+ checks ALL possible state transitions over time.

    Returns:
    - Whether all safety properties hold
    - Number of states explored / distinct states
    - Counterexample traces for any violations
    - TLC identity proof (version, PID, workers)
    - Automated fix suggestions for violations
    - Generated TLA+ spec (for transparency)

    Use verify_policy for quick Z3 consistency checks.
    Use tla_verify when you need exhaustive temporal verification.

    Args:
        csl_content: The complete CSL policy source code as a string.
        timeout: TLC subprocess timeout in seconds (default: 60).
        use_mock: If true, use Python BFS fallback instead of real TLC.
    """
    # Stage 1: Parse
    ast, parse_err = _parse_silent(csl_content)
    if parse_err:
        info = _get_error_info(parse_err)
        loc = f"Line {info['line']}, Column {info['column']}: " if info["line"] else ""
        return f"\u274c **PARSE ERROR**\n\n{loc}{info['message']}"

    # Stage 2: Validate
    try:
        with redirect_stdout(io.StringIO()):
            CSLValidator().validate(ast)
    except Exception as e:
        return f"\u274c **VALIDATION ERROR**\n\n{e}"

    # Stage 3: Build TLA+ spec
    try:
        from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
        builder = TLASpecBuilder()
        with redirect_stdout(io.StringIO()):
            spec = builder.build(ast)
    except Exception as e:
        return f"\u274c **TLA+ SPEC BUILD ERROR**\n\n{e}"

    # Stage 4: Run TLA+ verifier
    try:
        from chimera_core.engines.tla_engine import TLAVerifier
        with redirect_stdout(io.StringIO()):
            verifier = TLAVerifier(
                animate=False,
                use_real_tlc=not use_mock,
                tlc_timeout=timeout,
                tlc_auto_download=True,
            )
            is_valid, issues = verifier.verify(ast)
    except Exception as e:
        return f"\u274c **TLA+ VERIFICATION ERROR**\n\n{e}"

    # Stage 5: Run raw TLC for identity proof and stats (best-effort)
    tlc_result = None
    try:
        from chimera_core.engines.tla_engine.tlc_runner import run_tlc_on_spec
        with redirect_stdout(io.StringIO()):
            tlc_result = run_tlc_on_spec(spec, timeout=timeout)
    except Exception:
        pass

    # --------------- Format output ---------------
    lines = []

    if is_valid:
        lines.append("\u2705 **TLA+ VERIFIED** \u2014 All temporal safety properties hold.\n")
    else:
        lines.append("\u274c **TLA+ VERIFICATION FAILED**\n")

    lines.append(f"**Domain:** `{spec.module_name}`")
    lines.append(f"**Variables:** {len(spec.domain_info)}")
    lines.append(f"**Constraints:** {len(ast.constraints)}")

    # State space stats
    if tlc_result:
        lines.append(f"\n---\n### State Space Exploration\n")
        lines.append(f"- **States explored:** {tlc_result.states_explored:,}")
        lines.append(f"- **Distinct states:** {tlc_result.distinct_states:,}")
        lines.append(f"- **Time:** {tlc_result.time_ms}ms")
        engine = "Real TLC" if tlc_result.used_real_tlc else "Python BFS (mock)"
        lines.append(f"- **Engine:** {engine}")

        if tlc_result.used_real_tlc and tlc_result.tlc_version:
            lines.append(f"\n### TLC Identity Proof\n")
            lines.append(f"- **Version:** `{tlc_result.tlc_version}`")
            if tlc_result.tlc_pid:
                lines.append(f"- **PID:** `{tlc_result.tlc_pid}`")
            if tlc_result.java_workers:
                lines.append(f"- **Workers:** `{tlc_result.java_workers}`")

    # Variable domains
    if spec.domain_info:
        lines.append(f"\n---\n### Variable Domains\n")
        for var in spec.domain_info:
            card = var.get("card", "?")
            lines.append(f"- `{var['name']}`: {var['domain']} (cardinality: {card})")

    # Violations
    if issues:
        lines.append(f"\n---\n### Violations ({len(issues)})\n")
        for idx, issue in enumerate(issues, 1):
            constraint = getattr(issue, "constraint", None) or getattr(issue, "invariant", "?")
            message = getattr(issue, "message", str(issue))
            counterexample = getattr(issue, "counterexample", None)

            lines.append(f"**Violation #{idx}: `{constraint}`**\n")
            lines.append(f"{message}\n")

            if counterexample:
                lines.append("**Counterexample trace:**\n")
                for step_idx, step in enumerate(counterexample):
                    if isinstance(step, dict):
                        step_str = ", ".join(f"`{k}` = `{v}`" for k, v in step.items())
                        lines.append(f"  State {step_idx}: {step_str}")
                lines.append("")

        # Suggestion engine (best-effort)
        try:
            from chimera_core.engines.tla_engine.suggestion_engine import TLASuggestionEngine
            suggestion_engine = TLASuggestionEngine()
            for issue in issues:
                constraint = getattr(issue, "constraint", None)
                counterexample = getattr(issue, "counterexample", None)
                if constraint and counterexample:
                    with redirect_stdout(io.StringIO()):
                        analysis = suggestion_engine.analyze(constraint, counterexample, ast)
                    if analysis and getattr(analysis, "suggestions", None):
                        lines.append(f"\n**Fix suggestions for `{constraint}`:**\n")
                        if analysis.root_cause:
                            lines.append(f"Root cause: {analysis.root_cause}\n")
                        for sug in analysis.suggestions:
                            lines.append(f"- **{sug.title}** ({sug.fix_type}, confidence: {sug.confidence})")
                            lines.append(f"  {sug.explanation}")
                            if sug.before_snippet and sug.after_snippet:
                                lines.append(f"  Before: `{sug.before_snippet}`")
                                lines.append(f"  After:  `{sug.after_snippet}`")
                            lines.append("")
        except Exception:
            pass

    # Generated TLA+ spec (for transparency)
    if hasattr(spec, "tla_content") and spec.tla_content:
        lines.append(f"\n---\n### Generated TLA+ Spec\n")
        lines.append(f"```tla\n{spec.tla_content}\n```\n")
        if hasattr(spec, "cfg_content") and spec.cfg_content:
            lines.append(f"```cfg\n{spec.cfg_content}\n```")

    return "\n".join(lines)


@mcp.tool()
def universe_info(csl_content: str) -> str:
    """
    Analyze the state space "universe" of a CSL policy.

    Returns structural information about the policy's state space:
    - All variables with their domains, TLA+ set representations, and cardinalities
    - Total state space size (product of all variable cardinalities)
    - All constraints with their conditions and actions
    - Constraint coverage analysis (which variables are constrained vs unconstrained)
    - State space breakdown visualization

    Essential for understanding the "universe" an agent lives in,
    planning Evolving Universe experiments, and estimating TLC
    verification cost before running tla_verify.

    Args:
        csl_content: The complete CSL policy source code as a string.
    """
    # Parse
    ast, parse_err = _parse_silent(csl_content)
    if parse_err:
        info = _get_error_info(parse_err)
        loc = f"Line {info['line']}, Column {info['column']}: " if info["line"] else ""
        return f"\u274c **PARSE ERROR**\n\n{loc}{info['message']}"

    # Validate
    try:
        with redirect_stdout(io.StringIO()):
            CSLValidator().validate(ast)
    except Exception as e:
        return f"\u274c **VALIDATION ERROR**\n\n{e}"

    # Build TLA+ spec for domain analysis
    try:
        from chimera_core.engines.tla_engine.tla_spec_builder import TLASpecBuilder
        builder = TLASpecBuilder()
        with redirect_stdout(io.StringIO()):
            spec = builder.build(ast)
    except Exception as e:
        return f"\u274c **SPEC BUILD ERROR**\n\n{e}"

    lines = []
    lines.append(f"## Universe Analysis: `{spec.module_name}`\n")

    # --- Variables and domains ---
    lines.append("### Variables\n")
    lines.append("| Variable | Domain | TLA+ Set | Cardinality |")
    lines.append("|----------|--------|----------|-------------|")

    total_states = 1
    var_cards = {}
    for var in spec.domain_info:
        name = var.get("name", "?")
        domain = var.get("domain", "?")
        tla_set = var.get("tla_set", "?")
        card_str = var.get("card", "?")

        try:
            card = int(card_str)
        except (ValueError, TypeError):
            card = None

        if card:
            total_states *= card
            var_cards[name] = card

        lines.append(f"| `{name}` | {domain} | `{tla_set}` | {card_str} |")

    lines.append(f"\n**Total state space size:** {total_states:,} states")

    if total_states > 1_000_000:
        lines.append(f"\u26a0\ufe0f Large state space \u2014 TLC may need significant time or predicate abstraction")
    elif total_states > 100_000:
        lines.append(f"\u26a1 Medium state space \u2014 TLC should handle in <60s")
    else:
        lines.append(f"\u2705 Small state space \u2014 TLC will be fast")

    # --- Constraints ---
    lines.append(f"\n---\n### Constraints ({len(ast.constraints)})\n")

    constrained_action_vars = set()
    constrained_condition_vars = set()

    for c in ast.constraints:
        temporal = c.condition.temporal_operator.value
        modal = c.action.modal_operator.value
        action_var = c.action.variable

        lines.append(f"**`{c.name}`**")
        lines.append(f"  {temporal} ... THEN `{action_var}` {modal} ...\n")

        if action_var:
            constrained_action_vars.add(action_var)

        try:
            cond_vars = _extract_variable_names(c.condition.condition)
            constrained_condition_vars.update(cond_vars)
        except Exception:
            pass

    # --- Coverage analysis ---
    all_var_names = set()
    for v in (ast.domain.variable_declarations if ast.domain else []):
        all_var_names.add(v.name)

    unconstrained = all_var_names - constrained_action_vars - constrained_condition_vars
    action_only = constrained_action_vars - constrained_condition_vars
    condition_only = constrained_condition_vars - constrained_action_vars
    both = constrained_action_vars & constrained_condition_vars

    lines.append("---\n### Constraint Coverage\n")
    if both:
        lines.append(f"- **Fully constrained** (in conditions AND actions): {', '.join(f'`{v}`' for v in sorted(both))}")
    if action_only:
        lines.append(f"- **Action targets only** (constrained but not checked): {', '.join(f'`{v}`' for v in sorted(action_only))}")
    if condition_only:
        lines.append(f"- **Condition checks only** (checked but not constrained): {', '.join(f'`{v}`' for v in sorted(condition_only))}")
    if unconstrained:
        lines.append(f"- \u26a0\ufe0f **Unconstrained** (not in any rule): {', '.join(f'`{v}`' for v in sorted(unconstrained))}")
    else:
        lines.append(f"- \u2705 All variables participate in at least one constraint")

    # --- State space breakdown ---
    if var_cards:
        lines.append(f"\n---\n### State Space Breakdown\n")
        sorted_vars = sorted(var_cards.items(), key=lambda x: x[1], reverse=True)
        for vname, card in sorted_vars:
            bar_len = min(int(math.log2(card + 1) * 3), 40)
            bar = "\u2588" * bar_len
            lines.append(f"- `{vname}`: {card:,} values {bar}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resources: Example policies
# ---------------------------------------------------------------------------

@mcp.resource("policy://examples/hello_world")
def example_hello_world() -> str:
    """Hello World \u2014 simplest possible CSL policy (1 rule)."""
    return _load_example_file("hello_world")


@mcp.resource("policy://examples/age_verification")
def example_age_verification() -> str:
    """Age verification \u2014 multi-rule interaction with MUST BE / MUST NOT BE."""
    return _load_example_file("age_verification")


@mcp.resource("policy://examples/banking_guard")
def example_banking_guard() -> str:
    """Banking transfer guard \u2014 tiered limits, sanctions, risk scoring."""
    return _load_example_file("banking_guard")


@mcp.resource("policy://examples/agent_tool_guard")
def example_agent_tool_guard() -> str:
    """AI agent tool guard \u2014 RBAC, DLP, hard bans with ALWAYS."""
    return _load_example_file("agent_tool_guard")


@mcp.resource("policy://examples/dao_treasury_guard")
def example_dao_treasury_guard() -> str:
    """DAO treasury guard \u2014 arithmetic expressions, escalating approvals."""
    return _load_example_file("dao_treasury_guard")


@mcp.resource("policy://examples/tla_demo")
def example_tla_demo() -> str:
    """TLA+ demo \u2014 all temporal safety properties hold (ENABLE_FORMAL_VERIFICATION: TRUE)."""
    return _load_example_file("tla_demo")


@mcp.resource("policy://examples/tla_demo_violation")
def example_tla_demo_violation() -> str:
    """TLA+ violation demo \u2014 TLC finds reachable safety violations with counterexample traces."""
    return _load_example_file("tla_demo_violation")


# ---------------------------------------------------------------------------
# Prompt: CSL Expert
# ---------------------------------------------------------------------------

@mcp.prompt()
def csl_expert() -> str:
    """Load CSL-Core syntax reference and common patterns for writing policies."""
    return """You are working with CSL-Core (Constitutional Specification Language), a deterministic policy engine with Z3 formal verification and TLA+ model checking.

## CSL File Structure

```
CONFIG {
  ENFORCEMENT_MODE: BLOCK          // BLOCK | WARN | LOG
  CHECK_LOGICAL_CONSISTENCY: TRUE  // Z3 verification (recommended: TRUE)
  ENABLE_FORMAL_VERIFICATION: FALSE // TLA+ model checking (opt-in)
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}

DOMAIN DomainName {
  VARIABLES {
    var_name: type_definition
  }

  STATE_CONSTRAINT rule_name {
    WHEN condition
    THEN action
  }
}
```

## Variable Types

```
amount: 0..100000              // Integer range
score: 0..1                    // Numeric range (integers)
role: {"ADMIN", "USER"}        // String enum
country: {"US", "EU", "NK"}    // String enum
flag: {"TRUE", "FALSE"}        // Boolean-like enum
```

## Condition Operators

```
WHEN amount > 1000                          // Comparison: ==, !=, <, >, <=, >=
WHEN action == "TRANSFER" AND amount > 500  // Logical: AND, OR
WHEN NOT is_verified                        // Negation
WHEN amount > (balance * 0.1)               // Arithmetic: +, -, *, /
ALWAYS True                                 // Fires on every input (invariant)
```

## Action Operators (THEN clause)

```
THEN role MUST BE "ADMIN"           // Obligation (must equal)
THEN category MUST NOT BE "BANNED"  // Prohibition (must not equal)
THEN amount <= 5000                 // Comparison constraint
THEN approval_count >= 3            // Minimum threshold
```

## Common Patterns

**Hard ban (unconditional):**
```
STATE_CONSTRAINT no_delete {
  ALWAYS True
  THEN tool MUST NOT BE "DELETE"
}
```

**Tiered access:**
```
STATE_CONSTRAINT user_limit {
  WHEN role == "USER"
  THEN amount <= 1000
}
STATE_CONSTRAINT admin_limit {
  WHEN role == "ADMIN"
  THEN amount <= 10000
}
```

**Dynamic threshold:**
```
STATE_CONSTRAINT catastrophic_protection {
  WHEN transfer > (total_balance * 0.1)
  THEN approval_count >= 3
}
```

**DLP rule:**
```
STATE_CONSTRAINT no_pii_external {
  WHEN tool == "SEND_EMAIL" AND pii_present == "YES"
  THEN recipient MUST NOT BE "EXTERNAL"
}
```

## Workflow
1. Write .csl policy
2. `verify_policy` \u2014 Z3 checks logical consistency at compile time
3. `tla_verify` \u2014 TLA+ exhaustive model checking for temporal safety
4. `universe_info` \u2014 Analyze state space size and constraint coverage
5. `simulate_policy` \u2014 Test against JSON inputs at runtime
6. `explain_policy` \u2014 Get human-readable summary

## Key Principles
- **Fail-closed:** Missing keys, evaluation errors, unknown operators \u2192 BLOCK by default
- **Compile-time verification:** Z3 catches contradictions before runtime
- **Temporal verification:** TLA+ proves safety across ALL possible state transitions
- **Deterministic:** Same input always produces same output, no probabilistic behavior
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()