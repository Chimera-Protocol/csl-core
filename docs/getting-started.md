# Getting Started

Welcome to **CSL-Core**. This guide will get you up and running with the Chimera Specification Language environment, from installation to running your first Agent Guard.

---

## 1. Installation

CSL-Core is available via pip. It requires Python 3.10+.

```bash
pip install csl-core
```

To verify the installation and check the version:

```bash
cslcore --version
```

---

## 2. Quickstart (The First 5 Minutes)

If you have cloned the repository, we have prepared a `quickstart/` directory to help you learn the syntax basics immediately.

Navigate to the quickstart folder:

```bash
cd quickstart
```

### Step A: Verify a Policy

Try compiling the "Hello World" policy. This checks for syntax errors and logic consistency using the Z3 engine.

```bash
cslcore verify 01_hello_world.csl
```

### Step B: Simulate Runtime Behavior

Now, let's see how the policy behaves against input data without writing any Python code.

** Scenario 1: ** Small external transfer (Should be ALLOWED)

```bash
cslcore simulate 01_hello_world.csl --input '{"amount": 50, "destination": "EXTERNAL"}'
```

** Scenario 2: ** Large external transfer (Should be BLOCKED)

```bash
cslcore simulate 01_hello_world.csl --input '{"amount": 500, "destination": "EXTERNAL"}'
```

You can also explore:

- `02_age_verification.csl`: A simple logic gate for numerical constraints.
- `03_langchain_template.py`: A minimal Python script showing how to load CSL in code.

---

## 3. Explore Real-World Examples

Once you understand the basics, check the `examples/` directory in the repository root. These represent production-grade use cases.

### Core Policies (`examples/`)

- `agent_tool_guard.csl`: Protects an LLM Agent from calling dangerous tools.
- `chimera_banking_case_study.csl`: A complex financial policy with VIP limits and risk scoring.
- `dao_treasury_guard.csl`: Governance rules for a blockchain DAO.

To run the batched examples and see the output in action:

```bash
# Runs the Python runner which loads CSL files and simulates various scenarios
python examples/run_examples.py
```

---

## 4. Integrate with Your Agents (Plugins)

CSL-Core is designed to be a drop-in middleware for your AI framework.

### LangChain Integration

We provide a native plugin for LangChain in `chimera_core/plugins/langchain.py`.

You can see a full implementation in `examples/integrations/langchain_agent_demo.py`.

---

## 5. Write Your Own Policy

Ready to build your own? Create a file named `my_policy.csl`.

### 1. Define the Domain:

```csl
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

### 2. Verify it:

```bash
cslcore verify my_policy.csl
```

### 3. Run the REPL to test edge cases:

```bash
cslcore repl my_policy.csl
> {"action": "DELETE", "user_level": 2}
BLOCKED
```

---

## Next Steps

- Read the **Syntax Specification** to master the language.
- Check the **CLI Reference** for advanced debugging flags.
