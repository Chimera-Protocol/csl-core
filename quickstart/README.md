# CSL-Core Quickstart

This folder contains copy-paste friendly examples to get started with **CSL policies** and the **LangChain integration** in minutes.

## Contents

- `01_hello_world.csl`  
  Minimal “Hello World” policy: **restrict large external transfers**  
  Rule: If `amount >= 100` then `destination MUST BE "INTERNAL"`.

- `02_age_verification.csl`  
  Real-world policy example: **age gating + restricted categories**  
  Rules:
  - `< 18` cannot access `MATURE`
  - `< 18` cannot access `ALCOHOL`
  - `ALCOHOL` requires verification for adults (`user_age >= 18`)

- `03_langchain_template.py`  
  Copy-paste template that shows how to:
  - load a `.csl` policy via `load_guard()`
  - wrap LangChain tools via `guard_tools()`
  - inject static context (`user_role`, `environment`)
  - enable a terminal dashboard (optional)

---

## Prerequisites

### CSL-Core
You should be running this from the CSL-Core repo (or have `chimera_core` installed/importable).

### LangChain (only for the template)
The `chimera_core.plugins.langchain` integration requires `langchain-core`.

Install:
```bash
pip install langchain-core
```
(If you also want to run a real agent end-to-end, you’ll likely install your LangChain provider, e.g. langchain-openai, plus the main langchain package depending on your setup.)

---

## 1) Try a policy (Hello World)
Use the runtime directly (no LangChain needed):

```python
from chimera_core import load_guard, ChimeraError

guard = load_guard("quickstart/01_hello_world.csl")

# Allowed: amount < 100 can go anywhere
print(guard.verify({"amount": 50, "destination": "EXTERNAL"}).allowed)

# Blocked: amount >= 100 must be INTERNAL
try:
    guard.verify({"amount": 500, "destination": "EXTERNAL"})
except ChimeraError as e:
    print("Blocked:", e)
```    

---

## 2) Try a policy (Age Verification)

```python
from chimera_core import load_guard, ChimeraError

guard = load_guard("quickstart/02_age_verification.csl")

# Blocked: minor requesting MATURE
try:
    guard.verify({"user_age": 16, "category": "MATURE", "parent_verified": "NO"})
except ChimeraError as e:
    print("Blocked:", e)

# Allowed: adult requesting ALCOHOL with verification
print(
    guard.verify({"user_age": 25, "category": "ALCOHOL", "parent_verified": "YES"}).allowed
)
```    

---

## 3) LangChain quickstart (Tools Wrapping)
`03_langchain_template.py` is intentionally designed to be copied into your project.

** Run the template (as a starting point) **

From the repo root:
```bash
python quickstart/03_langchain_template.py
```

By default it loads the policy located next to the template:
- `quickstart/01_hello_world.csl`

** Environment variables **
The template supports:
- CHIMERA_DEBUG: Enables/disables the terminal dashboard (default: 1)

```bash
export CHIMERA_DEBUG=0
```

- USER_ROLE: Injected into policy context (default: USER)

```bash
export USER_ROLE=ADMIN
```
- ENV: Injected into policy context (default: DEV)

```bash
export ENV=PROD
```

** Integrate into your agent **

In your LangChain agent file:
```python
from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools

guard = load_guard("path/to/your_policy.csl")

safe_tools = guard_tools(
    tools=raw_tools,
    guard=guard,
    inject={
        "user_role": "USER",
        "environment": "DEV",
    },
    enable_dashboard=True,
    # tool_field="tool",  # optional: include tool name in policy context under your chosen key
)

agent = create_openai_tools_agent(llm, safe_tools, prompt)
executor = AgentExecutor(agent=agent, tools=safe_tools)
```
---

## Notes / Troubleshooting
**ImportError: langchain-core required **

Install LangChain core:
```bash
pip install langchain-core
```

** Policy file not found **
If you use relative paths, run scripts from the repo root, or use absolute paths.
The template already resolves the policy path relative to the template file location.

** Using tool-based rules **
If your policy has a variable that identifies which tool is being called (e.g. tool),
you can pass tool_field="tool" to guard_tools() so the wrapper injects the tool name into the context.

---

## Next steps
- Write your own policy in CSL language.
- Start with `01_hello_world.csl` (single rule), then expand like `02_age_verification.csl`
- Wrap your agent tools with guard_tools() for runtime enforcement.

Check out the `examples/` folder to more!