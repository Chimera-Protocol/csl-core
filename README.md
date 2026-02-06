# CSL-Core

> **"Solidity for AI Policies"** — Deterministic safety for probabilistic AI systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Z3 Verified](https://img.shields.io/badge/Z3-Formally%20Verified-green.svg)](https://github.com/Z3Prover/z3)
[![Open-Core](https://img.shields.io/badge/Model-Open--Core-orange.svg)](https://en.wikipedia.org/wiki/Open-core_model)

**CSL-Core** (Chimera Specification Language) is an open-source policy language and runtime that brings **mathematical rigor** to AI agent governance. Define rigid, formally verified "laws" for your AI systems and enforce them at runtime with deterministic guarantees; completely independent of prompts, fine-tuning, or model training.

## 📖 Table of Contents

- [Why CSL-Core?](#-why-csl-core)
- [The Problem](#-the-problem)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start-60-seconds)
- [Learning Path](#-learning-path)
  - [Step 1: Quickstart](#-step-1-quickstart-5-minutes--quickstart)
  - [Step 2: Real-World Examples](#-step-2-real-world-examples-30-minutes--examples)
  - [Step 3: Production Deployment](#-step-3-production-deployment)
- [Architecture](#️-architecture-the-3-stage-pipeline)
- [Documentation](#-documentation)
- [CLI Tools](#️-cli-tools--the-power-of-no-code-policy-development)
- [LangChain Integration](#-langchain-integration-deep-dive)
- [API Quick Reference](#-api-quick-reference)
- [Testing](#-testing)
- [Plugin Architecture](#-plugin-architecture)
- [Use Cases](#-use-cases)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [License](#-license--open-core-model)
- [Contact](#-contact--support)

---

## 💡 Why CSL-Core?

**Scenario**: You're building a LangChain or any AI agent for a fintech app. The agent can transfer funds, query databases, and send emails. You want to ensure:
- ❌ Junior users **cannot** transfer more than $1,000
- ❌ PII **cannot** be sent to external email domains
- ❌ The `secrets` table **cannot** be queried by anyone

**Traditional Approach** (Prompt Engineering):
```python
prompt = """You are a helpful assistant. IMPORTANT RULES:
- Never transfer more than $1000 for junior users
- Never send PII to external emails
- Never query the secrets table
[10 more pages of rules...]"""
```

**Problems:**
- ⚠️ LLM can be **prompt-injected** ("Ignore previous instructions...")
- ⚠️ Rules are **probabilistic** (99% compliance ≠ 100%)
- ⚠️ **No auditability** (which rule was violated?)
- ⚠️ **Fragile** (adding a rule might break existing behavior)


**CSL-Core Approach**:
```python
# 1. Define policy (my_policy.csl)
"""
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
}
DOMAIN AgentGuard {
  VARIABLES { 
    user_tier: {"JUNIOR", "SENIOR"}
    amount: 0..100000
  }
  STATE_CONSTRAINT junior_limit {
    WHEN user_tier == "JUNIOR"
    THEN amount <= 1000
  }
}
"""

# 2. Load and enforce (3 lines)
guard = load_guard("my_policy.csl")
safe_tools = guard_tools(tools, guard, inject={"user_tier": "JUNIOR"})
agent = create_openai_tools_agent(llm, safe_tools, prompt)

# 3. Sleep well
# - Mathematically proven consistent (Z3)
# - LLM cannot bypass (enforcement is external)
# - Every violation logged with constraint name
```

---

## 🎯 The Problem

Modern AI is inherently probabilistic. While this enables creativity, it makes systems **fundamentally unreliable** for critical constraints:

- ❌ **Prompts are suggestions**, not rules
- ❌ **Fine-tuning** biases behavior but guarantees nothing
- ❌ **Post-hoc classifiers** add another probabilistic layer (more AI watching AI)

**CSL-Core flips this model**: Instead of *asking* AI to behave, you **force it to comply** using an external, deterministic logic layer.

---

## ✨ Key Features

### 🔒 **Formal Verification (Z3)**
Policies are mathematically proven consistent at compile-time. Contradictions, unreachable rules, and logic errors are caught before deployment.

### ⚡ **Zero-Latency Runtime**
Compiled policies execute as lightweight Python functors. No heavy parsing, no API calls — just pure deterministic evaluation.

### 🔌 **LangChain-First Integration**
Drop-in protection for LangChain agents with **3 lines of code**:
- **Context Injection**: Pass runtime context (user roles, environment) that the LLM cannot override
- **Automatic Tool Mapping**: Tool names auto-injected into policy evaluation
- **Custom Context Mappers**: Map complex LangChain inputs to policy variables
- **Zero Boilerplate**: Wrap tools, chains, or entire agents with a single function call

### 🏭 **Factory Pattern for Convenience**
One-line policy loading with automatic compilation and verification:
```python
guard = load_guard("policy.csl")  # Parse + Compile + Verify in one call
```

### 🛡️ **Fail-Closed by Design**
If something goes wrong (missing data, type mismatch, evaluation error), the system blocks by default. Safety over convenience.

### 🔌 **Drop-in Integrations**
Native support for:
- **LangChain** (Tools, Runnables, LCEL chains)
- **Python Functions** (any callable)
- **REST APIs** (via plugins)

### 📊 **Built-in Observability**
Every decision produces an audit trail with:
- Triggered rules
- Violations (if any)
- Latency metrics
- Optional Rich terminal visualization

### 🧪 **Production-Ready**
Battle-tested with comprehensive test suites:
- ✅ Smoke tests (parser, compiler)
- ✅ Logic verification (Z3 engine integrity)
- ✅ Runtime decisions (allow vs block)
- ✅ Framework integrations (LangChain)
- ✅ CLI end-to-end tests
- ✅ Real-world example policies with full test coverage

Run the entire test suite:
```bash
pytest  # tests covering all components
```

---

## 🚀 Quick Start (60 Seconds)

### Installation

```bash
pip install csl-core
```

### Your First Policy

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

### Test It (No Code Required!)

CSL-Core provides a **powerful CLI** for testing policies without writing any Python code:

```bash
# 1. Verify policy (syntax + Z3 formal verification)
cslcore verify my_policy.csl

# 2. Test with single input
cslcore simulate my_policy.csl --input '{"action": "DELETE", "user_level": 2}'

# 3. Interactive REPL for rapid testing
cslcore repl my_policy.csl

> {"action": "DELETE", "user_level": 2}
allowed=False violations=1 warnings=0

> {"action": "DELETE", "user_level": 5}
allowed=True violations=0 warnings=0
```

### Use in Code (Python)

```python
from chimera_core import load_guard

# Factory method - handles parsing, compilation, and Z3 verification
guard = load_guard("my_policy.csl")

# This will pass
result = guard.verify({"action": "READ", "user_level": 1})
print(result.allowed)  # True

# This will be blocked
try:
    guard.verify({"action": "DELETE", "user_level": 2})
except ChimeraError as e:
    print(f"Blocked: {e}")
```

### Use in Code (LangChain) — **3 Lines of Code**

```python
from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools

# 1. Load policy (auto-compile with Z3 verification)
guard = load_guard("my_policy.csl")

# 2. Wrap tools with policy enforcement
safe_tools = guard_tools(
    tools=[search_tool, delete_tool, transfer_tool],
    guard=guard,
    inject={"user_level": 2, "environment": "prod"},  # Runtime context the LLM can't override
    tool_field="tool",  # Auto-inject tool name into policy context
    enable_dashboard=True  # Optional: Rich terminal visualization
)

# 3. Use in agent - enforcement is automatic and transparent
agent = create_openai_tools_agent(llm, safe_tools, prompt)
executor = AgentExecutor(agent=agent, tools=safe_tools)
```

**What happens under the hood:**
- Every tool call is intercepted before execution
- Policy is evaluated with injected context + tool inputs
- Violations block execution with detailed error messages
- Allowed actions pass through with zero overhead

---

## 📚 Learning Path

CSL-Core provides a structured learning journey from beginner to production:

### 🟢 **Step 1: Quickstart** (5 minutes) → [`quickstart/`](quickstart/)

No-code exploration of CSL basics:

```bash
cd quickstart/
cslcore verify 01_hello_world.csl
cslcore simulate 01_hello_world.csl --input '{"amount": 500, "destination": "EXTERNAL"}'
```

**What's included:**
- `01_hello_world.csl` - Simplest possible policy (1 rule)
- `02_age_verification.csl` - Multi-rule logic with numeric comparisons
- `03_langchain_template.py` - Copy-paste LangChain integration

**Goal:** Understand CSL syntax and CLI workflow in 5 minutes.

### 🟡 **Step 2: Real-World Examples** (30 minutes) → [`examples/`](examples/)

Production-ready policies with comprehensive test coverage:

```bash
cd examples/
python run_examples.py  # Run all examples with test suites
python run_examples.py agent_tool_guard  # Run specific example
```

**Available Examples:**

| Example | Domain | Complexity | Key Features |
|---------|--------|------------|--------------|
| [`agent_tool_guard.csl`](examples/agent_tool_guard.csl) | AI Safety | ⭐⭐ | RBAC, PII protection, tool permissions |
| [`chimera_banking_case_study.csl`](examples/chimera_banking_case_study.csl) | Finance | ⭐⭐⭐ | Risk scoring, VIP tiers, sanctions |
| [`dao_treasury_guard.csl`](examples/dao_treasury_guard.csl) | Web3 Governance | ⭐⭐⭐⭐ | Multi-sig, timelocks, emergency bypass |

**Interactive Demos:**
```bash
# See LangChain integration with visual dashboard
python examples/integrations/langchain_agent_demo.py
```

**Goal:** Explore production patterns and run comprehensive test suites.

### 🔵 **Step 3: Production Deployment**

Once you understand the patterns, integrate into your application:

1. **Write your policy** (or adapt from examples)
2. **Test thoroughly** using CLI batch simulation
3. **Integrate** with 3-line LangChain wrapper
4. **Deploy** with CI/CD verification (policy as code)

See [**Getting Started Guide**](docs/getting-started.md) for detailed walkthrough.

---

## 🏗️ Architecture: The 3-Stage Pipeline

CSL-Core separates **Policy Definition** from **Runtime Enforcement** through a clean 3-stage architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. COMPILER (compiler.py)                    │
│  .csl file → AST → Intermediate Representation (IR) → Artifact  │
│  • Syntax validation                                             │
│  • Semantic validation                                           │
│  • Optimized functor generation                                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. VERIFIER (verifier.py)                     │
│              Z3 Theorem Prover - Static Analysis                 │
│  • Reachability analysis                                         │
│  • Contradiction detection                                       │
│  • Rule shadowing detection                                      │
│  ✅ If verification fails → Policy WILL NOT compile             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    3. RUNTIME GUARD (runtime.py)                 │
│                 Deterministic Policy Enforcement                 │
│  • Fail-closed evaluation                                        │
│  • Zero dependencies (pure Python functors)                      │
│  • Audit trail generation                                        │
│  • <1ms latency for typical policies                            │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Heavy computation (parsing, Z3 verification) happens once at compile-time. Runtime is pure evaluation — no symbolic solver, no heavy libraries.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**Getting Started**](docs/getting-started.md) | Installation, first policy, integration guide |
| [**Syntax Specification**](docs/syntax-spec.md) | Complete CSL language reference |
| [**CLI Reference**](docs/cli-reference.md) | Command-line tools (`verify`, `simulate`, `repl`) |
| [**Philosophy**](docs/philosophy.md) | Design principles and vision |
| [**What is CSL?**](docs/what-is-csl.md) | Deep dive into the problem & solution |

---

## 🎓 Example Policies Deep Dive

The [`examples/`](examples/) directory contains **production-ready policies** with comprehensive test suites. Each example demonstrates real-world patterns and includes:
- ✅ Complete `.csl` policy file
- ✅ JSON test cases (allow + block scenarios)
- ✅ Automated test runner with visual reports
- ✅ Expected violations for each blocked case

### Running Examples

**Run all examples** with the test runner:

```bash
python examples/run_examples.py
```

**Run specific example:**
```bash
python examples/run_examples.py agent_tool_guard
python examples/run_examples.py banking
```

**Show detailed failures:**
```bash
python examples/run_examples.py --details
```

---

### Policy Pattern Library

Common patterns extracted from examples for reuse:

**Pattern 1: Role-Based Access Control (RBAC)**
```js
STATE_CONSTRAINT admin_only {
    WHEN operation == "SENSITIVE_ACTION"
    THEN user_role MUST BE "ADMIN"
}
```
**Source:** `agent_tool_guard.csl` (lines 30-33)

**Pattern 2: PII Protection**
```js
STATE_CONSTRAINT no_external_pii {
    WHEN pii_present == "YES"
    THEN destination MUST NOT BE "EXTERNAL"
}
```
**Source:** `agent_tool_guard.csl` (lines 55-58)

**Pattern 3: Progressive Limits by Tier**
```js
STATE_CONSTRAINT basic_tier_limit {
    WHEN tier == "BASIC"
    THEN amount <= 1000
}

STATE_CONSTRAINT premium_tier_limit {
    WHEN tier == "PREMIUM"
    THEN amount <= 50000
}
```
**Source:** `chimera_banking_case_study.csl` (lines 28-38)

**Pattern 4: Hard Sanctions (Fail-Closed)**
```js
STATE_CONSTRAINT sanctions {
    ALWAYS True  // Always enforced
    THEN country MUST NOT BE "SANCTIONED_COUNTRY"
}
```
**Source:** `chimera_banking_case_study.csl` (lines 22-25)

**Pattern 5: Emergency Bypass**
```js
// Normal rule with bypass
STATE_CONSTRAINT normal_with_bypass {
    WHEN condition AND action != "EMERGENCY"
    THEN requirement
}

// Emergency gate (higher threshold)
STATE_CONSTRAINT emergency_gate {
    WHEN action == "EMERGENCY"
    THEN approval_count >= 10
}
```
**Source:** `dao_treasury_guard.csl` (lines 60-67)

See [`examples/README.md`](examples/README.md) for the complete policy catalog.

---

## 🧪 Testing

CSL-Core includes a comprehensive test suite following the Testing Pyramid:

```bash
# Run all tests
pytest

# Run specific categories
pytest tests/integration          # LangChain plugin tests
pytest tests/test_cli_e2e.py      # End-to-end CLI tests
pytest -k "verifier"              # Z3 verification tests
```

**Test Coverage**:
- ✅ Smoke tests (parser, compiler)
- ✅ Logic verification (Z3 engine integrity)
- ✅ Runtime decisions (allow vs block scenarios)
- ✅ LangChain integration (tool wrapping, LCEL gates)
- ✅ CLI end-to-end (subprocess simulation)

See [tests/README.md](tests/README.md) for detailed test architecture.

---

## 🔗 LangChain Integration Deep Dive

CSL-Core provides **the easiest way to add deterministic safety to LangChain agents**. No prompting required, no fine-tuning needed — just wrap and run.

### Why LangChain + CSL-Core?

| Problem | LangChain Alone | With CSL-Core |
|---------|----------------|---------------|
| **Prompt Injection** | LLM can be tricked to bypass rules | Policy enforcement happens **before** tool execution |
| **Role-Based Access** | Must trust LLM to respect roles | Roles injected at runtime, **LLM cannot override** |
| **Business Logic** | Encoded in fragile prompts | Mathematically verified constraints |
| **Auditability** | Parse LLM outputs after the fact | Every decision logged with violations |

### Basic Tool Wrapping

```python
from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools

# Your existing tools
from langchain.tools import DuckDuckGoSearchRun, ShellTool
tools = [DuckDuckGoSearchRun(), ShellTool()]

# Load policy
guard = load_guard("agent_policy.csl")

# Wrap tools (one line)
safe_tools = guard_tools(tools, guard)

# Use in agent - that's it!
agent = create_openai_tools_agent(llm, safe_tools, prompt)
```

### Advanced: Context Injection

The `inject` parameter lets you pass runtime context that the LLM **cannot override**:

```python
safe_tools = guard_tools(
    tools=tools,
    guard=guard,
    inject={
        "user_role": current_user.role,           # From your auth system
        "environment": os.getenv("ENV"),          # prod/dev/staging
        "tenant_id": session.tenant_id,           # Multi-tenancy
        "rate_limit_remaining": quota.remaining   # Dynamic limits
    }
)
```

**Policy Example (agent_policy.csl):**
```js
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}

DOMAIN AgentGuard {
  VARIABLES {
    tool: String
    user_role: {"ADMIN", "USER", "ANALYST"}
    environment: {"prod", "dev"}
  }
  
  // Block shell access in production
  STATE_CONSTRAINT no_shell_in_prod {
    WHEN environment == "prod"
    THEN tool MUST NOT BE "ShellTool"
  }
  
  // Only admins can delete
  STATE_CONSTRAINT admin_only_delete {
    WHEN tool == "DeleteRecordTool"
    THEN user_role MUST BE "ADMIN"
  }
}
```

### Advanced: Custom Context Mapping

Map complex LangChain inputs to your policy variables:

```python
def my_context_mapper(tool_input: Dict) -> Dict:
    """
    LangChain tools receive kwargs like:
      {"query": "...", "limit": 10, "metadata": {...}}
    
    Your policy expects:
      {"search_query": "...", "result_limit": 10, "source": "..."}
    """
    return {
        "search_query": tool_input.get("query"),
        "result_limit": tool_input.get("limit"),
        "source": tool_input.get("metadata", {}).get("source", "unknown")
    }

safe_tools = guard_tools(
    tools=tools,
    guard=guard,
    context_mapper=my_context_mapper
)
```

### Advanced: LCEL Chain Protection

Insert a policy gate into LCEL chains:

```python
from chimera_core.plugins.langchain import gate

chain = (
    {"query": RunnablePassthrough()}
    | gate(guard, inject={"user_role": "USER"})  # Policy checkpoint
    | prompt
    | llm
    | StrOutputParser()
)

# If policy blocks, chain stops with ChimeraError
result = chain.invoke({"query": "DELETE * FROM users"})  # Blocked!
```

### Live Demo

See a complete working example in [`examples/integrations/langchain_agent_demo.py`](examples/integrations/langchain_agent_demo.py):
- Simulated financial agent with transfer tools
- Role-based access control (USER vs ADMIN)
- PII protection rules
- Rich terminal visualization

```bash
python examples/integrations/langchain_agent_demo.py
```

---

## 🔌 Plugin Architecture

CSL-Core provides a universal plugin system for integrating with AI frameworks.

**Available Plugins**:
- ✅ **LangChain** (`chimera_core.plugins.langchain`)
- 🚧 **LlamaIndex** (coming soon)
- 🚧 **AutoGen** (coming soon)

**Create Your Own Plugin**:

```python
from chimera_core.plugins.base import ChimeraPlugin

class MyFrameworkPlugin(ChimeraPlugin):
    def process(self, input_data):
        # Enforce policy
        self.run_guard(input_data)
        
        # Continue framework execution
        return input_data
```

All lifecycle behavior (fail-closed semantics, visualization, context mapping) is inherited automatically from `ChimeraPlugin`.

See [chimera_core/plugins/README.md](chimera_core/plugins/README.md) for the integration guide.

---

## 📖 API Quick Reference

### Loading Policies (Factory Pattern)

```python
from chimera_core import load_guard, create_guard_from_string

# From file (recommended - handles paths automatically)
guard = load_guard("policies/my_policy.csl")

# From string (useful for testing or dynamic policies)
policy_code = """
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
}

DOMAIN Test {
  VARIABLES { x: 0..10 }
  STATE_CONSTRAINT limit { ALWAYS True THEN x <= 5 }
}
"""
guard = create_guard_from_string(policy_code)
```

### Runtime Verification

```python
# Basic verification
result = guard.verify({"x": 3})
print(result.allowed)  # True
print(result.violations)  # []

# Error handling
from chimera_core import ChimeraError

try:
    guard.verify({"x": 15})
except ChimeraError as e:
    print(f"Blocked: {e}")
    print(f"Violations: {e.violations}")
```

### LangChain Integration

```python
from chimera_core.plugins.langchain import guard_tools, gate

# Tool wrapping
safe_tools = guard_tools(
    tools=[tool1, tool2],
    guard=guard,
    inject={"user": "alice"},
    tool_field="tool_name",
    enable_dashboard=True
)

# LCEL gate
chain = prompt | gate(guard) | llm
```

### Runtime Configuration

```python
from chimera_core import RuntimeConfig

config = RuntimeConfig(
    raise_on_block=True,           # Raise ChimeraError on violations
    collect_all_violations=True,   # Report all violations, not just first
    missing_key_behavior="block",  # "block", "warn", or "ignore"
    evaluation_error_behavior="block"
)

guard = load_guard("policy.csl", config=config)
```

---

## 🛠️ CLI Tools — The Power of No-Code Policy Development

CSL-Core's CLI is **not just a utility — it's a complete development environment** for policies. Test, debug, and deploy without writing a single line of Python.

### Why CLI-First?

- ⚡ **Instant Feedback**: Test policy changes in milliseconds
- 🔍 **Interactive Debugging**: REPL for exploring edge cases
- 🤖 **CI/CD Ready**: Integrate verification into your pipeline
- 📊 **Batch Testing**: Run hundreds of test cases with visual reports
- 🎨 **Rich Visualization**: See exactly which rules triggered

---

### 1. `verify` — Compile & Formally Verify

The `verify` command is your first line of defense. It checks syntax, semantics, and **mathematical consistency** using Z3.

```bash
# Basic verification
cslcore verify my_policy.csl

# Output:
# ⚙️  Compiling Domain: MyGuard
#    • Validating Syntax... ✅ OK
#    ├── Verifying Logic Model (Z3 Engine)... ✅ Mathematically Consistent
#    • Generating IR... ✅ OK
```

**Advanced Debugging:**

```bash
# Show Z3 trace on verification failures
cslcore verify complex_policy.csl --debug-z3
```

**Skip verification** (not recommended for production):
```bash
cslcore verify policy.csl --skip-verify
```

---

### 2. `simulate` — Test Without Writing Code

The `simulate` command is your **policy test harness**. Pass inputs, see decisions, validate behavior.

**Single Input Testing:**

```bash
# Test one scenario
cslcore simulate agent_policy.csl \
  --input '{"tool": "TRANSFER_FUNDS", "user_role": "ADMIN", "amount": 5000}'

# Output:
# ✅ ALLOWED
```

**Batch Testing with JSON Files:**

Create `test_cases.json`:
```json
[
  {
    "name": "Junior user tries transfer",
    "input": {"tool": "TRANSFER_FUNDS", "user_role": "JUNIOR", "amount": 100},
    "expected": "BLOCK"
  },
  {
    "name": "Admin transfers within limit",
    "input": {"tool": "TRANSFER_FUNDS", "user_role": "ADMIN", "amount": 4000},
    "expected": "ALLOW"
  }
]
```

Run all tests:
```bash
cslcore simulate agent_policy.csl --input-file test_cases.json --dashboard
```

**Machine-Readable Output (CI/CD):**

```bash
# JSON output for automated testing
cslcore simulate policy.csl --input-file tests.json --json --quiet

# Output to file (JSON Lines format)
cslcore simulate policy.csl --input-file tests.json --json-out results.jsonl
```

**Runtime Behavior Flags:**

```bash
# Dry-run: Report what WOULD be blocked without actually blocking
cslcore simulate policy.csl --input-file tests.json --dry-run

# Fast-fail: Stop at first violation
cslcore simulate policy.csl --input-file tests.json --fast-fail

# Lenient mode: Missing keys warn instead of block
cslcore simulate policy.csl \
  --input '{"incomplete": "data"}' \
  --missing-key-behavior warn
```

---

### 3. `repl` — Interactive Policy Development

The REPL (Read-Eval-Print Loop) is **the fastest way to explore policy behavior**. Load a policy once, then test dozens of scenarios interactively.

```bash
cslcore repl my_policy.csl --dashboard
```

**Interactive Session:**

```
cslcore> {"action": "DELETE", "user_level": 2}
🛡️ BLOCKED: Constraint 'strict_delete' violated.
  Rule: user_level >= 4 (got: 2)

cslcore> {"action": "DELETE", "user_level": 5}
✅ ALLOWED

cslcore> {"action": "READ", "user_level": 0}
✅ ALLOWED

cslcore> exit
```

**Use Cases:**
- 🧪 **Rapid Prototyping**: Test edge cases without reloading
- 🐛 **Debugging**: Explore why a specific input is blocked
- 📚 **Learning**: Understand policy behavior interactively
- 🎓 **Demos**: Show stakeholders real-time policy decisions

---

### CLI in CI/CD Pipelines

**Example: GitHub Actions**

```yaml
name: Verify Policies
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install CSL-Core
        run: pip install csl-core
      
      - name: Verify all policies
        run: |
          for policy in policies/*.csl; do
            cslcore verify "$policy" || exit 1
          done
      
      - name: Run test suites
        run: |
          cslcore simulate policies/prod_policy.csl \
            --input-file tests/prod_tests.json \
            --json --quiet > results.json
      
      - name: Check for violations
        run: |
          if grep -q '"allowed": false' results.json; then
            echo "❌ Policy tests failed"
            exit 1
          fi
```

**Exit Codes for Automation:**

| Code | Meaning | Use Case |
|------|---------|----------|
| `0` | Success / Allowed | Policy valid or input allowed |
| `2` | Compilation Failed | Syntax error or Z3 contradiction |
| `3` | System Error | Internal error or missing file |
| `10` | Runtime Blocked | Policy violation detected |

---

### Advanced CLI Usage

**Debug Z3 Solver Issues:**
```bash
# When verification fails with internal errors
cslcore verify complex_policy.csl --debug-z3 > z3_trace.log
```

**Skip Validation Steps:**
```bash
# Skip semantic validation (not recommended)
cslcore verify policy.csl --skip-validate

# Skip Z3 verification (DANGEROUS - only for development)
cslcore verify policy.csl --skip-verify
```

**Custom Runtime Behavior:**
```bash
# Block on missing keys (default)
cslcore simulate policy.csl --input '{"incomplete": "data"}' --missing-key-behavior block

# Warn on evaluation errors instead of blocking
cslcore simulate policy.csl --input '{"bad": "type"}' --evaluation-error-behavior warn
```

See [**CLI Reference**](docs/cli-reference.md) for complete documentation.

---

## 🎯 Use Cases

CSL-Core is production-ready for:

### 🏦 **Financial Services**
- Transaction limits by user tier
- Sanctions enforcement
- Risk-based blocking
- Fraud prevention rules

### 🤖 **AI Agent Safety**
- Tool permission management
- PII protection
- Rate limiting
- Dangerous operation blocking

### 🏛️ **DAO Governance**
- Multi-sig requirements
- Timelock enforcement
- Reputation-based access
- Treasury protection

### 🏥 **Healthcare**
- HIPAA compliance rules
- Patient data access control
- Treatment protocol validation
- Audit trail requirements

### ⚖️ **Legal & Compliance**
- Regulatory rule enforcement
- Contract validation
- Policy adherence verification
- Automated compliance checks

** CSL-Core is currently in Alpha, provided 'as-is' without any warranties; the developers accept no liability for any direct or indirect damages resulting from its use. **

---

## 🗺️ Roadmap

### ✅ Completed
- [x] Core language (CSL syntax, parser, AST)
- [x] Z3 formal verification engine
- [x] Python runtime with fail-closed semantics
- [x] LangChain integration (Tools, LCEL, Runnables)
- [x] Factory pattern for easy policy loading
- [x] CLI tools (verify, simulate, repl)
- [x] Rich terminal visualization
- [x] Comprehensive test suite
- [x] Custom context mappers for framework integration

### 🚧 In Progress
- [ ] Policy versioning & migration tools
- [ ] Web-based policy editor
- [ ] LangGraph integration

### 🔮 Planned
- [ ] LlamaIndex integration
- [ ] AutoGen integration
- [ ] Haystack integration
- [ ] Policy marketplace (community-contributed policies)
- [ ] Cloud deployment templates (AWS Lambda, GCP Functions, Azure Functions)
- [ ] Policy analytics dashboard
- [ ] Multi-policy composition
- [ ] Hot-reload support for development

### 🔒 Enterprise (Commercial)
- [x] TLA+ temporal logic verification
- [x] Causal inference engine
- [ ] Multi-tenancy support
- [ ] Advanced policy migration tooling
- [ ] Priority support & SLA

---

## 🤝 Contributing

We welcome contributions! CSL-Core is open-source and community-driven.

**Ways to Contribute**:
- 🐛 Report bugs via [GitHub Issues](https://github.com/yourusername/csl-core/issues)
- 💡 Suggest features or improvements
- 📝 Improve documentation
- 🧪 Add test cases
- 🎓 Create example policies for new domains
- 🔌 Build framework integrations (LlamaIndex, AutoGen, Haystack)
- 🌟 Share your LangChain use cases and integration patterns

**High-Impact Contributions We'd Love:**
- 📚 More real-world example policies (healthcare, legal, supply chain)
- 🔗 Framework integrations (see `chimera_core/plugins/base.py` for the pattern)
- 🎨 Web-based policy editor
- 📊 Policy analytics and visualization tools
- 🧪 Additional test coverage for edge cases

**Contribution Process**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

---

## 📄 License & Open-Core Model

### Core (This Repository)
CSL-Core is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

**What's included in the open-source core:**
- ✅ Complete CSL language (parser, compiler, runtime)
- ✅ Z3-based formal verification
- ✅ LangChain integration
- ✅ CLI tools (verify, simulate, repl)
- ✅ Rich terminal visualization
- ✅ All example policies and test suites

### Enterprise Edition (Optional / Under Research & Deployment)
Advanced capabilities for large-scale deployments:
- 🔒 **TLA+ Temporal Logic Verification**: Beyond Z3, full temporal model checking
- 🔒 **Causal Inference Engine**: Counterfactual analysis and causal reasoning
- 🔒 **Multi-tenancy Support**: Policy isolation and tenant-scoped enforcement
- 🔒 **Policy Migration Tools**: Version control and backward compatibility
- 🔒 **Cloud Deployment Templates**: Production-ready Kubernetes/Lambda configs
- 🔒 **Priority Support**: SLA-backed engineering support

---

## 🙏 Acknowledgments

CSL-Core is built on the shoulders of giants:

- **[Z3 Theorem Prover](https://github.com/Z3Prover/z3)** - Microsoft Research (Leonardo de Moura, Nikolaj Bjørner)
- **[LangChain](https://github.com/langchain-ai/langchain)** - Harrison Chase and contributors
- **[Rich](https://github.com/Textualize/rich)** - Will McGugan (terminal visualization)

---

## 📬 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/akarlaraytu/csl-core/issues)
- **Discussions**: [Ask questions, share use cases](https://github.com/akarlaraytu/csl-core/discussions)
- **Email**: [akarlaraytu@gmail.com](mailto:akarlaraytu@gmail.com)

---

## ⭐ Star History

If you find CSL-Core useful, please consider giving it a star on GitHub! It helps others discover the project.

---

**Built with ❤️ by the Chimera Team**

*Making AI systems mathematically safe, one policy at a time.*
