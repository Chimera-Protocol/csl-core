# CSL-Core Plugin Architecture & LangChain Integration

This directory (`chimera_core.plugins`) contains the official integration layer between **ChimeraGuard** (CSL runtime enforcement) and external AI frameworks.

The design follows a single guiding principle:

**Universal Logic, Specific Implementations**

- **`base.py`**  
  Framework-agnostic core logic: data normalization, policy enforcement lifecycle, and optional visualization.

- **`langchain.py`**  
  Thin, ergonomic adapters for LangChain primitives (Tools, Runnables, LCEL chains).

All integrations are **fail-closed by default** and designed for enterprise-grade safety, extensibility, and determinism.

---

## 📁 File Overview

### `base.py` — Core Plugin Engine

`base.py` defines the shared infrastructure used by all framework integrations.

It provides:

- A **universal context mapping layer** that converts arbitrary Python objects into CSL-compatible dictionaries.
- A **deterministic enforcement lifecycle**:  
  normalize → verify → (optional) visualize → allow / raise.
- Optional **Rich-based terminal visualization**, enabled via a single flag.
- A single abstract base class (`ChimeraPlugin`) that new integrations can extend.

This module contains **no framework-specific logic** and is reusable across all agent frameworks.

---

### `langchain.py` — LangChain Integration

`langchain.py` builds on top of `base.py` and exposes ready-to-use helpers for LangChain:

- Guarded Tools (wrappers around `BaseTool`)
- LCEL-compatible policy gates (`Runnable`)
- One-line helpers for wrapping multiple tools

It does not reimplement enforcement logic.  
All security, normalization, and visualization flows come directly from `base.py`.

---

## 🚀 Quick Start (LangChain)

### 1. Protect an Agent’s Tools

Prevent an agent from taking dangerous actions (for example, transferring funds above a threshold) by wrapping its tools.

```python
from chimera_core.runtime import ChimeraGuard
from chimera_core.plugins.langchain import guard_tools
```


Wrap existing tools in a single line:

```python
safe_tools = guard_tools(
    raw_tools,
    guard,
    inject={"user_role": "JUNIOR_TRADER"},
    enable_dashboard=True
)
```

Create the agent using guarded tools:
```python
agent = create_openai_tools_agent(llm, safe_tools, prompt)
```

All tool executions are now policy-checked before execution.

---

### 2. Protect an LCEL Chain

Insert a policy gate directly into a LangChain Expression Language (LCEL) pipeline.

```python
from chimera_core.plugins.langchain import gate
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"input": RunnablePassthrough()}
    | gate(guard, enable_dashboard=True)
    | prompt
    | llm
)
```

Behavior:
- If the policy allows → input passes through unchanged.
- If the policy blocks → execution stops with a ChimeraError.

---

## Core Concepts (base.py)

### Universal Context Mapper

Policy enforcement requires converting runtime inputs into structured data the policy understands.

`base.py` provides a **universal context mapper** that automatically normalizes common Python objects:
- Mapping / dict → dict
- Pydantic models (v1 & v2) → dict
- Dataclasses → dict
- Strings → {"content": "..."}
- Message-like objects with a .content attribute → {"content": ..., "role": ...}
- Fallback → vars(obj) or {"content": str(obj)}

This prevents runtime crashes and dramatically reduces integration friction.

###  Custom Context Mapping
For advanced use cases, integrations may override the default mapping with a custom function.

```python
def my_mapper(obj):
    return {
        "risk_score": obj.metadata.risk,
        "action": obj.type,
    }

wrap_tool(tool, guard, context_mapper=my_mapper)
```
Custom mappers apply only to the integration where they are explicitly provided.

---

##  🔌 API Reference

### chimera_core.plugins.langchain 

**guard_tools(tools, guard, ...)**
Wraps an iterable of LangChain tools with ChimeraGuard enforcement.

Parameters:
- inject (Dict): Static context added to every evaluation (e.g., user role, environment, tenant).
- tool_field (str, optional): If provided, inserts the tool name into context under this key.
- enable_dashboard (bool): Enables real-time terminal visualization (if Rich is available).

### gate(guard, ...)
Creates an LCEL-compatible Runnable that acts as a policy checkpoint.

Behavior:
- Pass-through on ALLOW
- Raises ChimeraError on BLOCK

### chimera_core.plugins.base

**ChimeraPlugin**

Abstract base class for building new framework integrations
(e.g., LlamaIndex, Haystack, AutoGen, custom agent runtimes).

It guarantees:
- Fail-closed error handling
- Deterministic policy evaluation
- Standardized data normalization
- Optional Rich-based visualization

---

## How to Extend CSL Plugins (New Framework Integration)
CSL-Core is designed so new framework integrations are **easy, safe, and consistent.***

To integrate a new framework (for example **LlamaIndex** or a custom agent runtime):

### Step 1 — Create a New Plugin File
```python
chimera_core/plugins/llamaindex.py
```

### Step 2 — Subclass ChimeraPlugin
```python
from chimera_core.plugins.base import ChimeraPlugin

class LlamaIndexPlugin(ChimeraPlugin):
    def process(self, input_data):
        # Enforce policy
        self.run_guard(input_data)

        # Continue framework-specific execution
        return input_data
```

### Step 3 — Integrate with the Framework Runtime

Your subclass should:
- Accept framework-specific inputs
- Call self.run_guard(input_data) before execution
- Pass through input/output unchanged on ALLOW
- Let ChimeraError propagate on BLOCK

You do **NOT** need to:
- Reimplement context mapping
- Handle visualization
- Handle error semantics
- Duplicate policy logic

All lifecycle behavior is inherited **automatically** from `ChimeraPlugin`.

---

## Security Guarantees

Fail-Closed
If policy evaluation crashes, encounters invalid data, or raises an error, execution is blocked.

### No Data Leakage
Inputs are forwarded downstream only if the policy explicitly allows them.

### Composition over Inheritance
LangChain tools are wrapped using object composition to avoid Pydantic metaclass conflicts and ensure stability across framework versions.

## Developer Experience Summary

Import → Wrap → Run.

Complexity lives inside the library.
Simplicity is exposed to the developer.

