# CSL-Core

[![PyPI version](https://img.shields.io/pypi/v/csl-core?color=blue)](https://pypi.org/project/csl-core/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/csl-core?period=total&units=abbreviation&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/csl-core)
[![Python](https://img.shields.io/pypi/pyversions/csl-core.svg)](https://pypi.org/project/csl-core/)
[![License](https://img.shields.io/pypi/l/csl-core.svg)](LICENSE)
[![Z3 Verified](https://img.shields.io/badge/Z3-Formally%20Verified-purple.svg)](https://github.com/Z3Prover/z3)
[![TLA+ Verified](https://img.shields.io/badge/TLA%E2%81%BA-Model%20Checked-brightgreen.svg)](https://github.com/tlaplus/tlaplus)

## ❤️ Our Contributors!

[![Contributors](https://contrib.rocks/image?repo=Chimera-Protocol/csl-core&v=5)](https://github.com/Chimera-Protocol/csl-core/graphs/contributors)

**CSL-Core** (Chimera Specification Language) is a deterministic safety layer for AI agents. Write rules in `.csl` files, verify them mathematically with Z3, enforce them at runtime — outside the model. The LLM never sees the rules. It simply cannot violate them.

```bash
pip install csl-core
```

Originally built for [**Project Chimera**](https://github.com/Chimera-Protocol/Project-Chimera), now open-source for any AI system.

---

## Why?

```python
prompt = """You are a helpful assistant. IMPORTANT RULES:
- Never transfer more than $1000 for junior users
- Never send PII to external emails
- Never query the secrets table"""
```

This doesn't work. LLMs can be prompt-injected, rules are probabilistic (99% ≠ 100%), and there's no audit trail when something goes wrong.

**CSL-Core flips this**: rules live outside the model in compiled, Z3-verified policy files. Enforcement is deterministic — not a suggestion.

---

## Quick Start (60 Seconds)

### 1. Write a Policy

Create `my_policy.csl`:

```js
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
}

DOMAIN MyGuard {
  VARIABLES {
    action: {"READ", "WRITE", "DELETE"}
    user_level: 0..5
  }

  STATE_CONSTRAINT strict_delete {
    WHEN action == "DELETE"
    THEN user_level >= 4
  }
}
```

`WHEN` conditions support `AND` / `OR` for compound rules, e.g. `WHEN action == "TRANSFER" AND user_tier == "BASIC"` — this is what lets a policy be proportional (block writes without blocking reads) instead of an all-or-nothing gate. Full grammar in [`docs/syntax-spec.md`](docs/syntax-spec.md).

### 2. Verify & Test (CLI)

```bash
# Compile + Z3 formal verification
cslcore verify my_policy.csl

# Test a scenario
cslcore simulate my_policy.csl --input '{"action": "DELETE", "user_level": 2}'
# → BLOCKED: Constraint 'strict_delete' violated.

# Interactive REPL
cslcore repl my_policy.csl
```

### 3. Use in Python

```python
from chimera_core import load_guard

guard = load_guard("my_policy.csl")

result = guard.verify({"action": "READ", "user_level": 1})
print(result.allowed)  # True

result = guard.verify({"action": "DELETE", "user_level": 2})
print(result.allowed)  # False
```

---

## Benchmark: Adversarial Attack Resistance

We tested prompt-based safety rules vs CSL-Core enforcement across 4 frontier LLMs with 22 adversarial attacks and 15 legitimate operations:

| Approach | Attacks Blocked | Bypass Rate | Legit Ops Passed | Latency |
|----------|----------------|-------------|------------------|---------|
| GPT-4.1 (prompt rules) | 10/22 (45%) | 55% | 15/15 (100%) | ~850ms |
| GPT-4o (prompt rules) | 15/22 (68%) | 32% | 15/15 (100%) | ~620ms |
| Claude Sonnet 4 (prompt rules) | 19/22 (86%) | 14% | 15/15 (100%) | ~480ms |
| Gemini 2.0 Flash (prompt rules) | 11/22 (50%) | 50% | 15/15 (100%) | ~410ms |
| **CSL-Core (deterministic)** | **22/22 (100%)** | **0%** | **15/15 (100%)** | **~0.84ms** |


**Why 100%?** Enforcement happens outside the model. Prompt injection is irrelevant because there's nothing to inject against. Attack categories: direct instruction override, role-play jailbreaks, encoding tricks, multi-turn escalation, tool-name spoofing, and more.

> Full methodology: [`benchmarks/`](benchmarks/)

---

## LangChain Integration

Protect any LangChain agent with 3 lines — no prompt changes, no fine-tuning:

```python
from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

guard = load_guard("agent_policy.csl")

# Wrap tools — enforcement is automatic
safe_tools = guard_tools(
    tools=[search_tool, transfer_tool, delete_tool],
    guard=guard,
    inject={"user_role": "JUNIOR", "environment": "prod"},  # LLM can't override these
    tool_field="tool"  # Auto-inject tool name
)

agent = create_tool_calling_agent(llm, safe_tools, prompt)
executor = AgentExecutor(agent=agent, tools=safe_tools)
```

Every tool call is intercepted before execution. If the policy says no, the tool doesn't run. Period.

### Context Injection

Pass runtime context that the LLM **cannot override** — user roles, environment, rate limits:

```python
safe_tools = guard_tools(
    tools=tools,
    guard=guard,
    inject={
        "user_role": current_user.role,         # From your auth system
        "environment": os.getenv("ENV"),        # prod/dev/staging
        "rate_limit_remaining": quota.remaining # Dynamic limits
    }
)
```

### LCEL Chain Protection

```python
from chimera_core.plugins.langchain import gate

chain = (
    {"query": RunnablePassthrough()}
    | gate(guard, inject={"user_role": "USER"})  # Policy checkpoint
    | prompt | llm | StrOutputParser()
)
```

---

## CLI Tools

The CLI is a complete development environment for policies — test, debug, and deploy without writing Python.

### `verify` — Compile + Z3 Proof

```bash
cslcore verify my_policy.csl

# ⚙️  Compiling Domain: MyGuard
#    • Validating Syntax... ✅ OK
#    ├── Verifying Logic Model (Z3 Engine)... ✅ Mathematically Consistent
#    • Generating IR... ✅ OK
```

### `simulate` — Test Scenarios

```bash
# Single input
cslcore simulate policy.csl --input '{"action": "DELETE", "user_level": 2}'

# Batch testing from file
cslcore simulate policy.csl --input-file test_cases.json --dashboard

# CI/CD: JSON output
cslcore simulate policy.csl --input-file tests.json --json --quiet
```

### `repl` — Interactive Development

```bash
cslcore repl my_policy.csl --dashboard

cslcore> {"action": "DELETE", "user_level": 2}
🛡️ BLOCKED: Constraint 'strict_delete' violated.

cslcore> {"action": "DELETE", "user_level": 5}
✅ ALLOWED
```

### `formal` — TLA⁺ Model Checking

```bash
cslcore formal my_policy.csl
```

Runs the official TLC model checker (`java -jar tla2tools.jar`) against your policy. TLC exhaustively explores every reachable state in the abstract state space and proves each temporal property holds — or returns a concrete counterexample trace with the exact state that breaks your invariant.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       TLA⁺ FORMAL VERIFICATION ENGINE                        ║
║          Chimera Specification Language · Temporal Logic of Actions          ║
║                                                                              ║
║    ⚡  REAL TLC  ·  java -jar tla2tools.jar  ·  Exhaustive Model Checking    ║
║       TLC2 Version 2026.03.31.154134 (rev: becec35)  ·  pid 48146  ·  1      ║
║                                  worker(s)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Variable      Domain                         Cardinality
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  agent_tier    {"STANDARD", "PREMIUM"}                |2|
  task_type     {"READ", "WRITE", "ANALYZE"}           |3|
  risk_score    0..5                                   |6|

  ├─ □(no_destructive_ops)      ✅  HOLDS  [288 states  349ms]
  ├─ □(no_production_access)    ✅  HOLDS  [288 states  349ms]
  ├─ □(bounded_risk)            ✅  HOLDS  [288 states  349ms]

  └─ Proof hash: 17dd1564897d242fc045a3a884a52bbb… ✅

╔══════════════ TLA⁺ VERIFICATION COMPLETE — ALL PROPERTIES HOLD ══════════════╗
║  ✅  Domain: AIAgentSafetyDemo  ·  ⬡ 144 states  ·  ⏱ 1047ms               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

Enable in any policy by adding one line to `CONFIG`:

```js
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  ENABLE_FORMAL_VERIFICATION: TRUE   // ← triggers cslcore formal automatically
}
```

Or run standalone:

```bash
cslcore formal policy.csl              # real TLC (Java required, JAR auto-downloaded)
cslcore formal policy.csl --mock       # Python BFS fallback (no Java needed)
cslcore formal policy.csl --timeout 120
cslcore formal policy.csl --export-tla ./specs/   # save .tla + .cfg for TLA+ Toolbox
```

> **No Java?** CSL-Core falls back to a Python BFS model checker automatically. The banner clearly labels which engine ran. JAR is auto-downloaded on first use (~4MB from the official TLA+ GitHub release).

### CI/CD Pipeline

```yaml
# GitHub Actions
- name: Verify policies
  run: |
    for policy in policies/*.csl; do
      cslcore verify "$policy" || exit 1
    done
```

---

## MCP Server (Claude Desktop / Cursor / VS Code)

Write, verify, and enforce safety policies directly from your AI assistant — no code required.

```bash
pip install "csl-core[mcp]"
```

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "csl-core": {
      "command": "uv",
      "args": ["run", "--with", "csl-core[mcp]", "csl-core-mcp"]
    }
  }
}
```

| Tool | What It Does |
|---|---|
| `verify_policy` | Z3 formal verification — catches contradictions at compile time |
| `simulate_policy` | Test policies against JSON inputs — ALLOWED/BLOCKED |
| `explain_policy` | Human-readable summary of any CSL policy |
| `scaffold_policy` | Generate a CSL template from plain-English description |

> **You:** "Write me a safety policy that prevents transfers over $5000 without admin approval"
>
> **Claude:** *scaffold_policy → you edit → verify_policy catches a contradiction → you fix → simulate_policy confirms it works*

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  1. COMPILER    .csl → AST → IR → Compiled Artifact      │
│     Syntax validation, semantic checks, functor gen       │
├──────────────────────────────────────────────────────────┤
│  2. Z3 VERIFIER    Theorem Prover — Static Analysis       │
│     Contradiction detection, reachability, rule shadowing │
│     ⚠️ If verification fails → policy will NOT compile    │
├──────────────────────────────────────────────────────────┤
│  3. TLA⁺ VERIFIER  Model Checker — Temporal Safety        │
│     Exhaustive state-space exploration via TLC            │
│     Predicate abstraction for large numeric domains       │
│     Counterexample traces + automated fix suggestions     │
│     (opt-in: ENABLE_FORMAL_VERIFICATION: TRUE)            │
├──────────────────────────────────────────────────────────┤
│  4. RUNTIME     Deterministic Policy Enforcement          │
│     Fail-closed, zero dependencies, <1ms latency          │
└──────────────────────────────────────────────────────────┘
```

Heavy computation happens once at compile-time. Runtime is pure evaluation.

---

## Used in Production

<table>
  <tr>
    <td width="80" align="center">
      <a href="https://github.com/Chimera-Protocol/Project-Chimera">🏛️</a>
    </td>
    <td>
      <a href="https://github.com/Chimera-Protocol/Project-Chimera"><b>Project Chimera</b></a> — Neuro-Symbolic AI Agent<br/>
      CSL-Core powers all safety policies across e-commerce and quantitative trading domains. Both are Z3-verified at startup.
    </td>
  </tr>
</table>

*Using CSL-Core? [Let us know](https://github.com/Chimera-Protocol/csl-core/discussions) and we'll add you here.*

---

## Example Policies

| Example | Domain | Key Features |
|---------|--------|--------------|
| [`agent_tool_guard.csl`](examples/agent_tool_guard.csl) | AI Safety | RBAC, PII protection, tool permissions |
| [`chimera_banking_case_study.csl`](examples/chimera_banking_case_study.csl) | Finance | Risk scoring, VIP tiers, sanctions |
| [`dao_treasury_guard.csl`](examples/dao_treasury_guard.csl) | Web3 | Multi-sig, timelocks, emergency bypass |
| [`tla_demo.csl`](examples/tla_demo.csl) | Formal Methods | TLA⁺ model checking — all properties hold |
| [`tla_demo_violation.csl`](examples/tla_demo_violation.csl) | Formal Methods | TLA⁺ counterexample trace + fix suggestions |

```bash
python examples/run_examples.py          # Run all with test suites
python examples/run_examples.py banking  # Run specific example
```

---

## API Reference

```python
from chimera_core import load_guard, RuntimeConfig

# Load + compile + verify
guard = load_guard("policy.csl")

# With custom config
guard = load_guard("policy.csl", config=RuntimeConfig(
    raise_on_block=False,          # Return result instead of raising
    collect_all_violations=True,   # Report all violations, not just first
    missing_key_behavior="block"   # "block", "warn", or "ignore"
))

# Verify
result = guard.verify({"action": "DELETE", "user_level": 2})
print(result.allowed)     # False
print(result.violations)  # ['strict_delete']
```

Full docs: [**Getting Started**](docs/getting-started.md) · [**Syntax Spec**](docs/syntax-spec.md) · [**CLI Reference**](docs/cli-reference.md) · [**Philosophy**](docs/philosophy.md)

---

## Roadmap

**✅ Done:** Core language & parser · Z3 verification · Fail-closed runtime · LangChain integration · CLI (verify, simulate, repl, formal) · MCP Server · TLA⁺ model checking with real TLC · Predicate abstraction · Counterexample analysis · Production deployment in Chimera v1.7.0

**🚧 In Progress:** Policy versioning · LangGraph integration

**🔮 Planned:** LlamaIndex & AutoGen · Multi-policy composition · Hot-reload · Policy marketplace · Cloud templates

**🔒 Enterprise (Research):** Causal inference · Multi-tenancy

---

## Contributing

We welcome contributions! Start with [`good first issue`](https://github.com/Chimera-Protocol/csl-core/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or check [`CONTRIBUTING.md`](CONTRIBUTING.md).

**High-impact areas:** Real-world example policies · Framework integrations · Web-based policy editor · Test coverage

---

## License

**Apache 2.0**. CSL-Core is intentionally open: the policy language, compiler, Z3 verifier, CLI, MCP server, and all examples are free for any use — commercial, research, or personal. See [LICENSE](LICENSE).

This is a deliberate open-core posture. The policy DSL stays open so engineers, researchers, and the broader community can write, share, and verify policies without friction. The commercial layer (Chimera Runtime — production enforcement engine, multi-tenant dashboard, audit infrastructure) is licensed separately.

**Trademarks** — `Chimera Protocol`, `CSL`, and `AgentScan` are trademarks of Chimera Protocol. Apache 2.0 grants you rights to the code; trademarks are reserved.

For commercial Runtime licensing or partnership inquiries: aytug@chimera-protocol.com

---

**Built with ❤️ by [Chimera Protocol](https://github.com/Chimera-Protocol)** · [Issues](https://github.com/Chimera-Protocol/csl-core/issues) · [Discussions](https://github.com/Chimera-Protocol/csl-core/discussions) · [Email](mailto:akarlaraytu@gmail.com)