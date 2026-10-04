# CLI Reference

CSL-Core provides a robust Command Line Interface (CLI) for compiling policies, verifying logic, and simulating runtime behavior, and (since 0.6) for discovering the agents on a host, setting them up, and running them.

## Commands at a glance

| Command | What it does |
|---|---|
| `cslcore setup` | Guided first install: discovery, findings, policies, mapping, mode, wiring, activation |
| `cslcore wire` | Put the guard in each agent's call path (a hook or a decorator), diff first; `--undo` |
| `cslcore venom` | Read-only discovery of the AI agents on this host |
| `cslcore venom report` | The latest report, or one agent in detail |
| `cslcore venom map` | The reach map, full screen: chains, dive into an agent, 3D globe |
| `cslcore studio` | Write and prove a policy in the terminal (Z3, TLA+), bind agents, go live |
| `cslcore watch` | Live management panel: decisions, modes, kill switches, exemptions, live reach map |
| `cslcore map` | Generate mappings and test them, including bypass tricks and regression cases |
| `cslcore policy` | Policy workbench: list, show, new, edit, extend, fix, verify, diff, activate, bind |
| `cslcore mode` | Log or block per agent, kill switches per agent or tool |
| `cslcore exempt` | Exemptions with a reason and an approver |
| `cslcore hook` | Claude Code `PreToolUse` hook backed by a verified policy |
| `cslcore verify`, `simulate`, `repl`, `formal` | Work on a single policy file (sections 1 to 3 below) |

The CLI is built with **fail-safe defaults** and rich visualization tools.

## Installation & Usage

If installed via pip/poetry:

```bash
cslcore --help
```

Or running directly as a module:

```bash
python -m cslcore --help
```

---

## 1. Verify Command

`cslcore verify <policy>`

Parses the CSL policy, validates syntax, runs the Z3 Formal Verifier, and generates the compiled artifact.

- **Success**: Prints policy metadata (Domain, Version, Hash) and confirms logical consistency.
- **Failure**: Prints the specific logic error (Contradiction, Unreachable Rule, etc.) or syntax error.

### Usage

```bash
# Standard verification (Recommended)
cslcore verify strict_policy.csl

# Debugging complex logic errors (Shows Z3 trace)
cslcore verify strict_policy.csl --debug-z3
```

### Options

| Flag | Description |
|------|-------------|
| `--debug-z3` | Recommended for debugging. On failure, prints the internal Z3 trace tail to help diagnose sort mismatches or encoding issues. |
| `--skip-verify` | Disables the Z3 logic check. Not recommended for production policies. |
| `--skip-validate` | Skips semantic validation steps. |

---

## 2. Simulate Command

`cslcore simulate <policy>`

Loads a compiled policy and runs the ChimeraGuard runtime against provided inputs. This mimics exactly how the policy will behave in your application code.

### Input Methods

You can provide input as a raw JSON string or a file. The file can contain a single JSON object or a list of objects (batch mode).

```bash
# Single input via string
cslcore simulate policy.csl --input '{"action": "TRANSFER", "amount": 10000}'

# Batch input from file
cslcore simulate policy.csl --input-file execution_logs.json
```

### Output Formats

By default, `simulate` prints human-readable tables using the Rich library.

```bash
# Visual Dashboard (Logic Gates & Audit Log)
cslcore simulate policy.csl --input-file test.json --dashboard

# Machine-Readable JSON (to stdout)
cslcore simulate policy.csl --input-file test.json --json --quiet

# JSON Lines (Append to file - good for pipelines)
cslcore simulate policy.csl --input-file test.json --json-out results.jsonl
```

### Runtime Behavior Flags

CSL-Core is fail-closed by default. You can adjust this behavior using flags.

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | `False` | Analyzes input but never blocks. Reports what would have been blocked. |
| `--fast-fail` | `False` | Stops evaluation at the first violation. By default, it collects all violations for a full audit. |
| `--no-raise` | `False` | Prevents the CLI from exiting with an error code on BLOCK. Useful for batch processing. |
| `--missing-key-behavior` | `block` | What to do if a rule references a missing key: `block`, `warn`, or `ignore`. |
| `--evaluation-error-behavior` | `block` | What to do on type mismatches (e.g. comparing string to int): `block`, `warn`, or `ignore`. |

---

## 3. REPL Command

`cslcore repl <policy>`

Starts an interactive Read-Eval-Print Loop. Useful for rapid prototyping and testing edge cases without reloading the policy file every time.

### Usage

```bash
cslcore repl my_policy.csl --dashboard
```

Once inside:

```
cslcore> {"action": "deploy", "env": "prod"}
BLOCKED: Violation 'prod_freeze': env='prod' must be 'dev'.

cslcore> {"action": "deploy", "env": "dev"}
ALLOWED
```

- **Exit**: Press `Ctrl+C` or enter an empty line.

---

## 4. Venom commands (0.6)

All Venom commands share these options:

| Option | Meaning |
|---|---|
| `--root PATH` | scan this folder only (default: this host) |
| `--since WINDOW` | run history window, e.g. `7d`, `30d` (default: `7d`) |
| `--probe` | ask configured MCP servers for their real tool lists (starts stdio servers; asks first) |
| `--workspace PATH` | Venom workspace folder (default: current folder) |
| `--no-color` | plain output (also honours `NO_COLOR`) |
| `--no-anim` | no discovery animation (also `CSL_NO_ANIM`, CI) |
| `--plan-only` | show what would be written, write nothing |

Everything Venom writes stays in the workspace: `.csl/venom/` (reports, state, decision logs,
audit log, drafts, kept regression cases) and `policies/`.

### `cslcore setup`

The guided, resumable first install. Read-only until you confirm.

| Option | Meaning |
|---|---|
| `--yes` | non-interactive: accept defaults (never approves exemptions or activates policies) |
| `--activate` | activate drafts that pass the gate (explicit; `--yes` alone never activates) |
| `--mode {log,block}` | default enforcement mode for all agents (default: ask; `log` with `--yes`) |
| `--strategy {recommended,choose,templates}` | how to get a policy per agent (default: ask; `templates` with `--yes`) |
| `--restart` | start the flow from step 1 |
| `--agent ID` | limit the policy and mapping steps to one agent |
| `--wire` | with `--yes`: also make the wiring change in each agent whose policy is active (interactively you are asked, with the diff) |

### `cslcore wire`

Puts the guard in each agent's call path, for every agent with an active policy (a guard without
its policy refuses every call, so agents without one are listed, not wired). Each change is shown as
a diff and confirmed; each file is copied into `.csl/venom/wire/` first; after a change the host is
scanned again so the map and the live panel show what is guarded.

| Agent | The change |
|---|---|
| Claude Code (a project) | a `PreToolUse` hook for every tool in `<project>/.claude/settings.local.json` |
| Claude Code (user level) | the same hook in `~/.claude/settings.json` |
| Python tool functions | the guard at the top of the file and `@_csl_guard.tool("name")` above each tool function |
| tools that exist only as schemas, JavaScript, other assistants | not wired automatically: the exact snippet, and why |

| Option | Meaning |
|---|---|
| `--agent ID` | one agent (key or name) |
| `--yes` | apply without asking (each diff is still printed) |
| `--undo` | put the files back as they were (a file changed since is left alone and reported) |
| `--root PATH` | the folder that was scanned (default: the one the last scan covered) |

### `cslcore venom`

Read-only discovery: agents, their tools and risk classes, guard coverage, findings V01 to V16,
and the strongest reach chain. At a terminal it then asks where to go: `m` the reach map, `s` the
guided setup, `q` back to the shell. Scripts, pipes, `--json`, `--check` and CI are never asked
(`CSL_NO_PROMPT=1` turns the question off). From the second scan in a workspace on, a SINCE section names every
path that opened or closed since the last scan (a plugin installed, a credential added, a new
agent), and the report and JSON carry the full list (`reach.since_last_scan`).

| Option | Meaning |
|---|---|
| `--json` | print the inventory as JSON |
| `--check` | CI mode: exit 3 on findings at `--fail-on` level or vocabulary drift |
| `--fail-on {high,medium,low}` | finding level that fails `--check` (default: `high`) |
| `--fail-on-new-reach` | with `--check`: also fail when a path opened since the last scan in this workspace |
| `--compact` | header, agent counts, coverage and finding counts only |
| `--no-save` | do not write the report into the workspace |
| `--budget SECONDS` | time budget; results are marked partial when exceeded |
| `--yes` | confirm `--probe` without asking |

`cslcore venom report [--agent ID] [--format screen|md|json] [--rescan]` shows the latest report,
or one agent in detail.

### `cslcore venom map`

The reach map full screen. `--rescan` scans again first; `--once` prints one frame (scripts, CI).

| Key | Action |
|---|---|
| arrows, Tab, `1` to `9` | select a node; the panel shows what reaches it and what it reaches |
| Enter | dive into the selected agent: its tools, and what each tool reaches |
| Esc | back out |
| `s` | the 3D globe; it turns the selected node to the front |
| `n` | names on the map on or off |
| `r` | replay the spread |
| `x` | freeze the selected agent: every action blocked in any mode until `x` again; cuts its veins on the map |
| `m` | switch the selected agent between log and block mode (block-mode agents wear a green ring) |
| `w` | the live panel, on its live decisions (`f` there comes back to the full map) |
| `q` | quit |

### `cslcore studio [policy]`

A full CSL editor in the terminal. `--agent ID` opens (or starts) that agent's policy, `--new`
starts a new one, `--mock` runs TLA+ with the Python model checker even when TLC is available.

| Key | Action |
|---|---|
| `F5` / `Ctrl+R` | Z3: contradictions, rules that can never trigger |
| `F8` / `Ctrl+T` | TLA+: which states each rule blocks, read as a guard |
| `Ctrl+S` | save the draft |
| `Ctrl+L` | go live (needs a current Z3 pass; the previous version is kept) |
| `Ctrl+B` | bind agents, one or many |
| `Ctrl+O` / `Ctrl+N` | open another policy / start a new one |
| `Ctrl+Q` | quit (unsaved edits are kept as a draft) |

### `cslcore watch`

The live management panel. `--refresh SECONDS` sets the table refresh, `--once` prints one frame.

| Key | Action |
|---|---|
| arrows, `/` | select and search agents |
| `m` / `M` | log or block for the agent / for every agent |
| `x` | freeze the agent: every action is blocked in any mode until `x` again (`d` still works) |
| `e` | exempt the agent, with a reason |
| Enter | the agent's tools: disable or exempt one |
| Tab | rules ranked by would-block: `e` exempts an agent from a rule, `o` opens it in the studio |
| `g` | the live reach map beside the decisions (decisions flow over it); `g` again for the stream |
| `f` | the full reach map; `w` there comes back |
| `?` / `q` | help / quit |

### `cslcore map`

Generate a fail-closed mapping for an agent and test it; also tests your own mapper. See
[MAPPING.md](venom/MAPPING.md).

| Option | Meaning |
|---|---|
| `--agent ID` | agent to map |
| `--policy PATH` | policy to map against (default: the agent's active policy) |
| `--mapping PATH[:FUNC]` | your own mapping: a module with `map_call`, `path.py:function`, or `openclaw` |
| `--test` | run the mapping test (exit 3 on any fail-open) |
| `--import-module` | with `path.py:function`, load the whole file instead of only the function |
| `--allowed-root PATH`, `--allowed-command CMD`, `--allowed-destination URL` | values your mapping accepts; the bypass tricks start from them (repeatable) |
| `--classify VAR=KIND[:PARAM]` | your own variable names: `KIND` is `scope`, `command` or `destination` |
| `--cases FILE` / `--keep-cases` | regression cases (JSON lines), and keep them for every later test |
| `--yes` | write the generated mapping without asking |

### `cslcore policy <action> [target]`

`list`, `show`, `new`, `edit`, `extend`, `fix`, `verify`, `diff`, `activate`, `bind`, `unbind`.
Every draft passes the same gate (parse, validate, Z3, diff, confirmation) before it becomes active.
`--agent ID` (repeatable for `bind`), `--match PATTERN`, `--unbound`, `--exec-mode {allowlist,block}`,
`--all` (draft for every agent that needs one), `--yes`.

### `cslcore mode [log|block]`

Without a mode it shows the current state. `--agent ID` or `--all` / `--match PATTERN` choose the
agents; `--disable` / `--enable` is the kill switch for an agent; `--disable-tool TOOL` /
`--enable-tool TOOL` for one tool. Changes reach running agents on their next call.

### `cslcore exempt <add|list|approve|remove> [target]`

Exemptions need `--reason` and `--approved-by`; `--scope {agent,tool}` with `--tool NAME`,
`--expires YYYY-MM-DD`, and `--propose` to record one for approval later.

### `cslcore hook`

The Claude Code `PreToolUse` hook: reads the event on stdin and answers allow or deny from a
verified policy. `--agent ID`, `--policy PATH` and `--mapping PATH` (default: the agent's binding),
`--mode {log,block}` to override the mode.

---

## Advanced Debugging

### Z3 Trace (`--debug-z3`)

When a policy fails verification with an internal error or complex contradiction, standard error messages might not be enough.

Adding `--debug-z3` enables the trace tail, which shows the last operations sent to the solver before the crash or contradiction.

**Example Output:**

```
Z3 Trace Tail
#   event            rule              data
1   register_var     limit             sort=Int
2   binop            GT                left=amount right=limit
3   INTERNAL_ERROR   ...               ...
```

---

## Validation Codes

The CLI returns standard exit codes for CI/CD integration:

| Code | Meaning |
|------|---------|
| `0` | Success / Allowed |
| `2` | Compilation/Verification Failed |
| `3` | Unexpected System Error |
| `10` | Runtime Blocked (Policy Violation) |

Venom commands (`setup`, `venom`, `map`, ...) use `0` for success, `2` for a usage error, and `3`
when a check fails: `cslcore venom --check` with findings or drift, `cslcore map --test` with a
fail-open case.
