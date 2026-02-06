# CLI Reference

CSL-Core provides a robust Command Line Interface (CLI) for compiling policies, verifying logic, and simulating runtime behavior.

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
