# Mapping: closing the layer between the agent and the policy

A CSL policy is proven over its own vocabulary: `tool == "Write"`, `path_ok == "YES"`. The agent
does not speak that vocabulary. Its calls look like `Write(file_path="/srv/app/../etc/passwd")`.
The **mapping** is the code that turns a real call into policy variables, and it is the one part
Z3 and TLA+ cannot see. If the mapping sorts a call into the wrong side, the policy is still
correct and the call still goes through.

Two kinds of mapping mistakes matter:

| Kind | Example | What catches it |
|---|---|---|
| Malformed input | an unknown tool name, a missing amount, `True` where `"YES"` is expected | the fail-closed helpers and the mapping test |
| Wrong classification | `"/srv/app/../etc/passwd"` counted as inside `/srv/app`, `"git status; rm -rf /"` counted as an allowlisted command | the bypass tricks and the hardened classifiers |

The second kind is where hand-written mappers usually fail: the check that decides "is this path
in scope", "is this command allowlisted", "is this destination allowed". This guide covers it.

## 1. Test your own mapper against bypass tricks

```bash
cslcore map --agent ops --test --mapping our_mapper.py:classify \
  --classify path_ok=scope:file_path \
  --classify cmd_ok=command:command \
  --classify dest_ok=destination:url \
  --allowed-root /srv/app \
  --allowed-command "git status" \
  --allowed-destination https://api.example.com/v1
```

- `--mapping FILE:FUNCTION` names your function. Only that function runs; what it needs from its
  file is brought along when that is safe without running the file (literal constants, other
  functions in the file, standard-library imports). `--import-module` loads the whole file instead.
- `--classify VAR=KIND:PARAM` tells the test which policy variables your mapper computes as checks.
  `KIND` is `scope`, `command` or `destination`; `PARAM` is the tool parameter it is computed from.
  The names `target_in_scope`, `command_allowlisted` and `destination_allowlisted` (the ones setup
  generates) are known without it.
- `--allowed-root`, `--allowed-command`, `--allowed-destination` give one value your mapper
  accepts. Every trick is built from it, so the test also knows the right answer for each trick.
  Without one, the check is reported as **not covered**, never as passing.

For every tool where the policy actually checks the variable, the test first confirms that the
accepted value is accepted, then sends the tricks. A trick that ends in ALLOW is **fail-open**:

```
  BYPASS TRICKS  inputs built to land outside what the mapping allows; each must end in BLOCK
  ✗ path_ok  scope · 11 families · 2 tools
      ✗ traversal       Edit  "/srv/app/../etc/passwd"  → YES  ALLOW
      ✗ prefix          Edit  "/srv/app-evil/file.txt"  → YES  ALLOW
  ✗ cmd_ok  command · 11 families · 1 tool
      ✗ chaining        Bash  "git status; rm -rf /"  → YES  ALLOW
  ✗ dest_ok  destination · 12 families · 1 tool
      ✗ credentials     WebFetch  "https://api.example.com@evil.example/v1"  → YES  ALLOW

  RESULT       39 of 119 pass · 80 fail-open · 81 bypass tricks
               fix: compute path_ok with chimera_core.mapping.in_scope (passes every trick family)
```

The command exits with 3 on any fail-open, so it can gate CI.

### The trick families

Every trick is outside the allowed region **by construction**, so the right answer is always NO
and no judgement is needed.

| Kind | Families |
|---|---|
| scope | traversal (`/root/../etc`), prefix (`/root-evil`), relative, home (`~`), encoded (`%2e%2e`), NUL byte, backslash, double slash, dot segments, sibling folder, wrong type |
| command | chaining (`;`), and / or (`&&`, `\|\|`), pipe, background (`&`), substitution (`$( )`, backticks, `${IFS}`), redirect, newline, wrapper (`sh -c`, `sudo`, `env`, `xargs`), extra arguments, glob, wrong type |
| destination: URL | credentials (`https://allowed@evil`), suffix (`allowed.evil`), lookalike, host in the path or query, scheme (`file:`, `gopher:`), parser confusion (`\@`, `#@`), control characters, numeric hosts, other port, homograph, lists, wrong type |
| destination: address | separators (`,` `;` space), header injection (`\nBcc:`), display name, suffix domain, other mailbox, double `@`, lists, wrong type |
| destination: name | prefix, separators, control characters, lists, wrong type |

Relative paths are a family on purpose: a mapping cannot know the agent's working directory, so a
relative path can never be shown to stay inside the folder.

## 2. Keep your red-team findings as regression cases

A case file holds calls with the decision they must get, one JSON object per line:

```json
{"id": "RT-07", "tool": "Write", "args": {"file_path": "/srv/app/../../root/.ssh/authorized_keys"}, "expect": "BLOCK", "note": "traversal out of the workspace"}
{"id": "OK-01", "tool": "Write", "args": {"file_path": "/srv/app/notes.md"}, "expect": "ALLOW", "note": "ordinary write"}
```

```bash
cslcore map --agent ops --test --mapping our_mapper.py:classify --cases redteam.jsonl --keep-cases
```

`--keep-cases` stores them in `.csl/venom/cases/<agent>.jsonl`; every later mapping test of that
agent runs them, including the one in `cslcore setup`. A case expected to BLOCK that ends in ALLOW
is fail-open; a case expected to ALLOW that ends in BLOCK is reported as a failure (over-blocking).
`context` (for example `{"approval": "YES"}`) can be given per case.

## 3. Replace a weak check with a hardened classifier

`chimera_core.mapping` has three classifiers that pass every family above. They return the policy
values (`"YES"` / `"NO"` by default; `yes=` and `no=` change them) and have no dependencies.

```python
from chimera_core.mapping import command_allowed, destination_allowed, in_scope, to_enum

def classify(tool_name, args):
    return {
        "tool": to_enum(tool_name, ["Bash", "Write", "Edit", "Read", "WebFetch"], name="tool"),
        "path_ok": in_scope(args.get("file_path"), ["/srv/app"]),
        "cmd_ok": command_allowed(args.get("command"), ["git status", "git log *"]),
        "dest_ok": destination_allowed(args.get("url"), ["api.example.com"]),
    }
```

- `in_scope(path, roots, resolve=False)`: absolute paths only; normalises `..`; matches whole path
  segments (`/srv/app-evil` is not inside `/srv/app`); refuses `~`, NUL and control characters,
  backslashes and percent-encoded dots or slashes. `resolve=True` follows symlinks on this machine.
- `command_allowed(command, allowlist)`: compares word by word after shell-style splitting. A
  command with any shell syntax (`;` `&&` `||` `|` `&` `$( )` backticks, redirects, globs, `~`,
  newlines) is never allowlisted, nor is one run through a wrapper unless that exact wrapped
  command is listed. An entry ending in ` *` (`"git log *"`) takes further plain arguments and can
  never start with a shell or wrapper. Lists (argv) are accepted as they are.
- `destination_allowed(destination, allowlist, schemes=("https",))`: URLs are parsed; the scheme
  must be allowed, credentials in the URL are refused, the host is compared exactly (case,
  trailing dot and IDNA normalised), `*.example.com` allows subdomains only, and a non-default port
  must be listed (`api.example.com:8443`). Addresses, channels and wallets are compared exactly
  (email case-insensitively), and separators or line breaks are refused, so one allowed recipient
  cannot carry another.

Mappings generated by `cslcore setup` and `cslcore map` use these classifiers.

## 4. What this does and does not prove

- It is a test, not a proof. It covers the known families completely and repeatably, on every
  run; a new technique found later becomes a regression case and stays covered.
- Business classifications ("is this one of our instruction files", "is this a live service") are
  specific to your setup. Write them as regression cases.
- Symlinks are only checked with `in_scope(..., resolve=True)` on the machine that runs the agent.
- The policy itself (which rules exist, how strict they are) is a separate question: tune it with
  `cslcore watch` and edit it in `cslcore studio`.
