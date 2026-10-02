# Moving from CSL-Core 0.5.1 to 0.6

**Short version:** upgrading changes nothing in how your agents behave. Your code, your
policies and their `policy_hash` values stay exactly as they are. Everything new is opt-in,
and nothing is written outside the folder you run the new commands in.

## 1. Upgrade

```bash
pip install -U csl-core
```

`load_guard`, `ChimeraGuard.verify()`, `guard_tools`, the OpenClaw plugin, the CLI
(`verify`, `simulate`, `formal`, `repl`) and the MCP tools behave exactly as in 0.5.1. A
contract suite generated from the 0.5.1 release checks this on every change: public API
signatures, CLI output, MCP tool schemas and outputs, policy hashes and thousands of guard
decisions. Hashes in your existing audit records stay valid.

## 2. See what you have

```bash
cslcore venom
```

This is a read-only scan of the machine (or `--root /path/to/your/repo` for one folder). It
lists your agents, their tools and risk classes, which ones already have a guard in their call
path, and findings. Nothing is changed.

## 3. Bring your setup in

```bash
cslcore setup
```

Setup recognises an existing CSL-Core integration ("Existing CSL-Core setup found") and, at the
policies step, offers to **keep** the policies your agents already use. Kept (adopted) policies are
never copied or modified: your files stay the single source of truth. Agents that already enforce
a policy in your code stay in **block** mode.

Scripted: `cslcore setup --yes --strategy recommended`.

## 4. Test your own mapper

The mapping from a real tool call to policy variables is where bypasses hide (an unknown tool
name, a missing amount, a lowercase value). Test the mapper you already have:

```bash
cslcore map --agent payments-agent --test --mapping path/to/agent.py:agent_context_mapper
```

- LangChain `context_mapper(tool_input)`, plain `(tool_name, args)` functions, and OpenClaw
  (`--mapping openclaw`) are supported.
- Only the function itself is run, never the rest of your module. Literal constants, other
  functions in the same file and standard-library imports come along; for anything else you are
  told which names are missing, and `--import-module` loads the whole file when you want that.
- Every case that should block but ends in ALLOW is listed as fail-open. The generated mapping
  in `policies/<agent>_mapping.py` uses fail-closed helpers (`chimera_core.mapping`) and can
  replace or guide yours.
- If your mapper computes checks such as "is this path inside the workspace" or "is this command
  allowed", name them with `--classify` and give one value each accepts (`--allowed-root`,
  `--allowed-command`, `--allowed-destination`): the test then sends bypass tricks (traversal,
  command chaining, credentials in URLs, ...) at each check. Your red-team findings can be kept
  as regression cases with `--cases FILE --keep-cases`. Details and the hardened classifiers
  (`in_scope`, `command_allowed`, `destination_allowed`): [MAPPING.md](MAPPING.md).

## 5. One line for logs and live control (optional)

```python
from chimera_core.venom.observe import observe

guard = observe(load_guard("policies/payments.csl"), agent="payments-agent")
```

In block mode this behaves exactly like your current guard: same `RuntimeConfig`, same
`ChimeraError`, so `guard_tools(...)` and your error handling keep working. It adds:

- a decision log per agent (policy-variable values only, never the raw call),
- `cslcore watch`: live decisions, rules that block most, mode switch and kill switches,
- `cslcore mode --agent payments-agent log` to observe without blocking, and back.

It stays in block mode unless you switch it.

## What is written where

| Path | What |
|---|---|
| `.csl/venom/` | scan reports, state (modes, kill switches), decision logs, audit log of every change |
| `policies/` | policies and mappings you create or activate through Venom |

Nothing outside the folder you run `cslcore setup` in is ever written.
