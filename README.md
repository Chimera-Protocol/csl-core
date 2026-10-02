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

<p align="center">
  <img src="https://raw.githubusercontent.com/Chimera-Protocol/csl-core/main/docs/assets/venom.gif" alt="cslcore venom: discovery of the AI agents on a host" width="820">
</p>

<p align="center"><b>New in 0.6: <code>cslcore venom</code></b> finds every AI agent on a machine and what it can reach.
<code>cslcore setup</code> writes, verifies and wires their policies. <code>cslcore watch</code> runs them.
<a href="#venom-discover-write-map-watch-06">See it below</a>.</p>

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

## Venom: discover, write, map, watch (0.6)

Writing a policy assumes you already know which agents you run, what they can reach and how their
real tool calls map onto policy variables. Venom makes that part of the tool.

```bash
pip install csl-core
cslcore venom          # read-only discovery of every AI agent on this machine
cslcore setup          # guided: findings, policies, mapping, enforcement mode, wiring
cslcore studio         # write and verify policies in the terminal (Z3, TLA+), bind agents, go live
cslcore watch          # live management panel
```

### Discover: `cslcore venom`

Reads code (parsed, never run), assistant and MCP configs, cron / systemd / launchd, the process
list, run history metadata and existing `.csl` policies. Nothing on the host is changed; credential
values, prompt text and transcripts never reach any output. Every agent gets its tools classified by
risk, its guard coverage and findings V01 to V16, from "agent runs as root" to "inbound webhook
reaches a public posting tool". JSON and Markdown reports land in `.csl/venom/reports/`;
`cslcore venom --check` gates CI.

### Set up: `cslcore setup`

<p align="center">
  <img src="https://raw.githubusercontent.com/Chimera-Protocol/csl-core/main/docs/assets/setup.gif" alt="cslcore setup: the guided flow" width="820">
</p>

Ten resumable steps, one per screen: scope, discovery, inventory, findings with what to do about
each, exemptions, policies, verification, mapping, enforcement mode and wiring, activation.

- **Policies**: per agent, keep the policy it already uses (0.5.1 installations are adopted in
  place), draft from risk-class templates, write it in the studio, or let your own assistant draft
  it through the MCP tools. Every draft passes the same gate (parse, validate, Z3, diff, your
  confirmation) before it becomes active.
- **Enforcement mode**: `log` (nothing blocked, every decision recorded as ALLOW or WOULD BLOCK) or
  `block`, with one default for all agents and exceptions by pattern.
- **Wiring**: one change per agent (Python, LangChain, Claude Code `PreToolUse` hook via
  `cslcore hook`), written to `.csl/venom/wiring.md`.

### Write and prove: `cslcore studio`

<p align="center">
  <img src="https://raw.githubusercontent.com/Chimera-Protocol/csl-core/main/docs/assets/studio.gif" alt="cslcore studio: Z3 and TLA+ checks of a policy" width="820">
</p>

The policy editor, inside the terminal. It opens on its own, from setup's policy step, from the
watch panel (`o` on a rule) and from the line the scan suggests. One policy per session.

- `F5` / `Ctrl+R` runs Z3 (contradictions between rules, rules that can never trigger) and
  `F8` / `Ctrl+T` runs TLA+ (real TLC when Java is available, the built-in model checker
  otherwise), read as a guard: which states each rule blocks, with blocked calls as examples.
  Nothing is verified on save.
- Suggestions come from Z3, TLA+ and the agents themselves (tool names the policy does not know,
  risky tools without a rule) and apply with Enter.
- The Agents tab replays the agent's recorded decisions against the edited text: how many calls
  would now be blocked or allowed, before anything changes.
- `Ctrl+B` binds agents (one policy can guard many, each with its own fail-closed mapping);
  `Ctrl+L` goes live after a current Z3 pass, keeps the previous version, and running guards
  switch on their next call.

### Map without gaps: `cslcore map --test`

<p align="center">
  <img src="https://raw.githubusercontent.com/Chimera-Protocol/csl-core/main/docs/assets/mapping.png" alt="cslcore map --test: bypass tricks against a hand-written mapper" width="720">
</p>

The policy is proven; the code that turns a real tool call into policy variables is not. Every
mapping is tested with case variants, unknown values, missing keys, wrong types and range edges,
and every derived check (path in scope, command allowlisted, destination allowed) with bypass
tricks: traversal, look-alike folders, chained and substituted commands, credentials and look-alike
hosts in URLs, and more, each built so the right answer is known. It works on your own mapper as it
is, keeps red-team findings as regression cases, and the hardened classifiers `in_scope`,
`command_allowed` and `destination_allowed` pass every trick family. Guide:
[docs/venom/MAPPING.md](docs/venom/MAPPING.md).

### Run: `cslcore watch`

<p align="center">
  <img src="https://raw.githubusercontent.com/Chimera-Protocol/csl-core/main/docs/assets/watch.gif" alt="cslcore watch: the live management panel" width="820">
</p>

Agents with their mode and block rate, the live decision stream, and rules ranked by how often they
would block (the ones to tune before switching to block). `m` switches an agent between log and
block, `d` disables an agent (kill switch), Enter opens its tools to disable or exempt one, Tab moves
to the rules to exempt an agent or edit the rule in the studio, `/` searches, `M` switches every
agent at once. Changes reach running agents on their next tool call, without a restart, and are
recorded in `.csl/venom/audit.jsonl`. The same controls exist on the command line: `cslcore mode`.

### Try it on a sample host

```bash
cslcore setup --root tests/venom/fixtures/host_ops --workspace /tmp/venom-demo
python scripts/venom_demo_traffic.py --workspace /tmp/venom-demo
cslcore watch --workspace /tmp/venom-demo
```

Already on 0.5.1? Upgrading changes nothing: `cslcore setup` recognises your integration, keeps your
policies in place and tests your own mapper. See [docs/venom/MIGRATION.md](docs/venom/MIGRATION.md).

Other commands: `cslcore venom report --agent NAME`, `cslcore policy list|show|new|edit|extend|fix|verify|diff|activate`,
`cslcore exempt add|list|approve|remove` (exemptions need a reason and an approver and are encoded in
the verified policy) and `--probe` to ask configured MCP servers for their real tool lists.
Optional: `pip install "csl-core[venom]"` adds psutil for process discovery.

**0.5.1 behavior is unchanged.** A contract suite generated from the v0.5.1 tag (public API, CLI
output, MCP tools and outputs, policy hashes, thousands of guard decisions) runs first in CI, together
with the unmodified v0.5.1 test suite.

The recordings above are made with [VHS](https://github.com/charmbracelet/vhs) on the sample host;
the scripts are in [docs/assets/tapes](docs/assets/tapes).

---

## Benchmark: Adversarial Attack Resistance

We tested prompt-based safety rules vs CSL-Core enforcement across 4 frontier LLMs with 22 adversarial attacks and 15 legitimate operations (run 2026-02-18, model versions as of that date — re-run pending against current models):

| Approach | Attacks Blocked | Bypass Rate | Legit Ops Passed | Latency |
|----------|----------------|-------------|------------------|---------|
| GPT-4.1 (prompt rules) | 10/22 (45%) | 55% | 15/15 (100%) | ~850ms |
| GPT-4o (prompt rules) | 15/22 (68%) | 32% | 15/15 (100%) | ~620ms |
| Claude Sonnet 4 (prompt rules) | 19/22 (86%) | 14% | 15/15 (100%) | ~480ms |
| Gemini 2.0 Flash (prompt rules) | 11/22 (50%) | 50% | 15/15 (100%) | ~410ms |
| **CSL-Core (deterministic)** | **22/22 (100%)** | **0%** | **15/15 (100%)** | **~0.78ms (median)** |

CSL-Core's own runtime hot path (`ChimeraGuard.verify()`, no compilation) measured in isolation is sub-0.1ms even at 40 compiled rules — see [`paper/PAPER_FACTS.md`](paper/PAPER_FACTS.md#64-runtime-enforcement-latency-e3--newly-measured-this-session-real) for the full methodology and per-size breakdown.


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
| `venom_inventory`, `venom_agent` | (0.6) The agents Venom discovered: tools, risk classes, guard status (redacted) |
| `venom_policy_context` | (0.6) Active policy or a starting draft, findings and drift for one agent |
| `venom_save_draft` | (0.6) Verifies a draft and saves it to `.csl/venom/drafts/`; never activates |
| `venom_propose_exemption` | (0.6) Proposes an exemption; only the operator approves it in the CLI |

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

**✅ Done:** Core language & parser · Z3 verification · Fail-closed runtime · LangChain integration · CLI (verify, simulate, repl, formal) · MCP Server · TLA⁺ model checking with real TLC · Predicate abstraction · Counterexample analysis · Production deployment in Chimera v1.7.0 · Venom: agent discovery, policy workbench, fail-closed mapping test, live management panel (0.6)

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