# Chimera Core Test Suite

This directory contains the comprehensive test suite for **CSL-Core** (ChimeraGuard). The tests are designed to verify everything from the basic parser syntax to the formal verification engine (Z3) and third-party integrations (LangChain).

## 🚀 Quickstart

**Prerequisites:**
Ensure you have the development dependencies installed:
```bash
pip install pytest langchain-core pydantic
```

**Run All Tests:**
```bash
pytest
# OR for less verbose output:
pytest -q
```

**Run Specific Categories:**
```bash
# Run only integration tests (LangChain)
pytest tests/integration

# Run only CLI End-to-End tests
pytest tests/test_cli_e2e.py

# Run only Formal Verification logic tests
pytest -k "verifier"
```

---

## 📂 Test Structure & Strategy

The test suite is organized into layers, following the **Testing Pyramid** approach:

```text
tests/
├── conftest.py ................. Shared fixtures (policy loaders, paths)
├── integration/
│   └── test_langchain.py ....... Plugin tests (Gate & Tool Wrapping)
├── test_cli_e2e.py ............. End-to-End CLI commands (verify, simulate)
├── test_compiler_smoke.py ...... Basic compilation sanity checks
├── test_parser_smoke.py ........ CSL Syntax parsing checks
├── test_verifier_agent_tool_guard.py ......... Z3 Formal Verification engine integrity
└── test_verifier_agent_tool_guard_decisions.py Logic validation (Allow vs Block scenarios)
```

### 1. Smoke Tests (`test_*_smoke.py`)
* **Goal:** Ensure the `parser` and `compiler` modules don't crash on valid inputs.
* **Speed:** Very Fast.

### 2. Logic & Verification Tests (`test_verifier_*.py`)
* **Goal:** Verify the "Brain" of Chimera.
* **Scope:**
    * Checks if the Z3 solver correctly identifies logical contradictions (e.g., Int vs String mismatches).
    * Validates that policies allow/block requests exactly as defined in the `.csl` file.

### 3. Integration Tests (`integration/`)
* **Goal:** Verify compatibility with external ecosystems.
* **Scope:**
    * **LangChain:** Tests `ChimeraRunnableGate` and `wrap_tool` to ensure they enforce policies without breaking the LangChain contract.
    * *Note:* These tests are automatically skipped if `langchain-core` is not installed.

### 4. End-to-End (E2E) Tests (`test_cli_e2e.py`)
* **Goal:** Simulate user behavior from the terminal.
* **Scope:** Spawns subprocesses to run `python -m chimera_core.cli` commands (`verify`, `simulate`) and asserts exit codes and standard output.

---

## 🧪 Configuration (`conftest.py`)

The `conftest.py` file handles:
1.  **Path Resolution:** Automatically finds the `examples/` directory regardless of where `pytest` is executed.
2.  **Policy Compilation:** Compiles the `agent_tool_guard.csl` example **once per session** (`scope="session"`). This significantly speeds up tests by avoiding repetitive compilation.