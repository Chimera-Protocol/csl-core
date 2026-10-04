# Changelog

All notable changes to CSL-Core are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- The map controls what it shows, as the live panel does: `x` freezes the selected agent (its
  veins go cold, a chain through it is shown as held) and `m` switches it between log and block
  mode (block-mode agents wear a green ring). Both ask first, act on the next tool call without a
  restart, and say so when no policy guards the agent yet. The panel's own map shows the same marks.

## [0.6.6]

### Added
- One flow. After `cslcore venom`, at a terminal, one key leads on: `m` the reach map (it grows
  out of one point), `s` the guided setup, `q` the shell. Scripts, pipes, `--json`, `--check` and
  CI are never asked; `CSL_NO_PROMPT=1` turns it off.
- The map and the live panel lead into each other on one screen: `w` on the map opens the panel
  (always on its live decisions), `f` in the panel opens the full map; with the panel's own map
  open (`g`), it grows into the full map and settles back on the way in.
- Setup offers the reach map after the inventory and findings steps, and the panel or the map
  when it completes; its home screen has `m` for the map (modes moved to `o`).

### Changed
- Freezing an agent is `x` everywhere (the kill switch; `d` still works). In a rule's view in
  the panel, exempting an agent from it moved from `x` to `e`, as everywhere else.

## [0.6.5]

### Added
- Since the last scan: from the second scan in a workspace on, `cslcore venom` names every path
  that opened or closed since the previous scan of the same folder or host (a plugin installed,
  a credential added, a new or removed agent, a tool now decided by a rule). On screen as SINCE,
  in the report as "Since the last scan", in JSON as `reach.since_last_scan`.
- `cslcore venom --check --fail-on-new-reach` fails CI when a path opened since the last scan.
- Tests for what happens after the scan: a tool that appears later (a new plugin or MCP server)
  is denied by the guard and the Claude Code hook, in log mode it is recorded as WOULD_BLOCK,
  a tool an agent never had is denied, and the guard fails closed when its policy is gone.

## [0.6.4]

### Fixed
- Reach chains scale: one breadth-first search per input and impact instead of walking every path.
  A host where many agents share one account (each can reach every other) used to take minutes from
  about 30 agents; 600 agents now take about half a second.
- A chain only counts when the input does not already reach the agent at its end directly; the
  strongest direct exposure is shown next to the strongest chain (scan screen and share card).
- One input node per kind (`inbound HTTP · 111 routes`) instead of one per route, which flooded
  the map and the legend on hosts with real APIs.
- The reach map on large hosts: agents spread over the canvas, every way in ignites in turn, the
  strongest direct exposure is the hero stroke when there is no chain, long names are shortened.
- Python warnings from the operator's own files (for example an invalid escape sequence) are no
  longer printed while the scanner parses them.
- The discovery animation hands over to the reach map on hosts with direct exposures but no chain
  (it used to stop at the discovery web there).

## [0.6.3]

### Added
- `cslcore venom --share`: a card of the machine's reach map, counts and strongest chain, written as
  SVG (and PNG when cairosvg or rsvg-convert is installed). No host name, user names or file paths
  on it; `--anonymize` also replaces agent names with their kind.
- `csl-core` command, the same CLI under the package name: `uvx csl-core venom` runs without
  installing.
- Release workflow: a version tag runs lint and every test, checks the tag against the package
  version, and publishes to PyPI through Trusted Publishing.
- CLI reference for the 0.6 commands (`docs/cli-reference.md`).

### Changed
- Lint is clean and blocking in CI, with the rule set and the ruff version pinned; the files frozen
  since v0.5.1 are excluded.

## [0.6.2]

### Added
- Reach chains: a reach graph built from what discovery found (untrusted inputs, which agent can
  change which other agent through its declared file scope, its account or root, and what each
  agent can do: root, money, credentials, commands, publishing). Chains are routes from an input
  through at least two agents to something none of the earlier agents could do alone; a tool with an
  active rule breaks them. The strongest chain is shown step by step with its evidence on the scan
  screen and in the Markdown and JSON reports; a host without chains shows its most serious direct
  exposure.
- The discovery animation ends in the reach map: the web draws back into one point, a drop lands on
  the entry of the strongest chain, and the spread follows only real reach, with the chain as a
  purple artery.
- `cslcore venom map`: the reach map full screen. Arrows or `1` to `9` select, Enter dives into an
  agent (its tools and what each reaches), `s` turns the map into a 3D globe that brings the
  selected node to the front, `n` names, `r` replays the spread; `--once` prints one frame.
- `cslcore watch`: `g` shows the live reach map, with every decision flowing over it as it happens.

### Changed
- Canvas output merges neighbouring cells of the same style, so animations draw with far less output.

## [0.6.1]

### Fixed
- Studio, TLA+: a policy the TLC model checker cannot handle (for example decimal numbers: "TLC
  can't handle real numbers") is checked with the built-in model checker instead of ending as "not
  verified"; the screen says which engine ran and why.
- Studio, TLA+: no more stalls while the result is prepared. The blocked-state grid runs within a
  time budget, samples large state spaces at random, and pauses to let the screen draw.
- Studio, TLA+: the waiting screen shows the elapsed time, and a note once a large state space takes
  longer than a few seconds.

### Changed
- `cslcore venom`: the discovery web stays on screen when the scan finishes, above the report.
- README rewritten around the 0.6 flow: quick start with `cslcore setup`, recordings of each step,
  the policy language, integrations and commands.

## [0.6.0] - Venom

CSL-Core goes from passive to active: it finds the agents on a host, shows what they can reach,
helps write and verify their policies, tests the mapping from real tool calls to policy
variables, and gives the operator a live management panel. 0.5.1 behavior is unchanged.

### Added
- `cslcore venom`: read-only discovery of every AI agent on the host (or `--root` folder) across
  code (Python and TypeScript / JavaScript, parsed and never executed), assistant and MCP configs,
  cron / systemd / launchd, processes, containers, listening ports, run history and existing
  policies. Risk classes per tool, coverage, vocabulary drift and findings V01 to V16. JSON and
  Markdown reports (`schema_version: 1`) with a redaction pass; `--check` for CI;
  `cslcore venom report --agent ID` for one agent in detail. `--probe` lists the real tools of
  configured MCP servers (stdio servers start only after confirmation; HTTP only on loopback).
- `cslcore setup`: a resumable ten-step flow (scope, discover, inventory, findings, exemptions,
  policies, verify, map, wire, activate) with an enforcement mode per agent. `--yes` never approves
  exemptions and never activates; activation in scripted runs needs `--activate`.
- `cslcore policy`: workbench for first install and existing policies (list, show, new, edit,
  extend, fix, verify, diff, activate). Every draft passes one gate: parse, validate, Z3, diff,
  operator confirmation. Template drafts use only discovered tool names and encode exemptions as
  `agent_id != "..."`.
- `chimera_core.mapping`: fail-closed helpers `to_enum`, `to_flag`, `to_range` and
  `guarded_verify`. `cslcore map` generates mappings and runs the mapping test, which flags any
  malformed or unknown input that ends in ALLOW (finding V11); it also tests hand-written mappings.
- Log mode and decision logs (`chimera_core.venom.observe.venom_guard`), `cslcore hook` for Claude
  Code `PreToolUse`, and `cslcore watch`: a live management panel with agents, decision stream, rule
  tuning, per-agent log / block switch, kill switch per agent and per tool. Changes reach running
  agents on their next call and are recorded in `.csl/venom/audit.jsonl`. `cslcore mode` offers the
  same controls on the command line.
- `cslcore exempt`: exemptions with a required reason and approver, optional expiry (V13).
- Setup is one step per screen with plain-language findings (why it matters, what to do), a
  policy menu per agent (keep an existing policy, templates, write it in the studio or your own
  editor, or your
  assistant: Claude Code is run directly when installed, the MCP prompt `venom_draft` otherwise),
  one default mode for all agents with exceptions by pattern, and bulk activation. It fits
  narrow terminals. A completed workspace opens a home screen with status and next steps.
- `cslcore mode --all` / `--match`; a workspace default mode.
- 0.5.1 installations: existing policies are adopted in place (never copied or changed);
  `observe(guard, agent=...)` wraps an existing guard (block mode by default, same
  `ChimeraError`); own mappers are tested function-only (LangChain, plain, OpenClaw).
  `docs/venom/MIGRATION.md`.
- Panel navigation (breadcrumb, Enter, Esc, Tab, `/`), exemptions per agent, tool or rule with a
  reason, rule relaxing through a verified draft, policy history; running guards reload a changed
  policy and keep the last good one.
- Discovery animation driven by real scan progress; any key skips; off for CI, `--json`,
  `NO_COLOR`, non-terminals and `--no-anim`.
- MCP tools for AI-assisted authoring in the user's own assistant: `venom_inventory`,
  `venom_agent`, `venom_policy_context`, `venom_save_draft` (verifies, writes drafts only),
  `venom_propose_exemption` (proposals only).
- `ChimeraPlugin` and the LangChain `wrap_tool` / `guard_tools` accept an existing guard
  (`guard=` / `reuse_guard=True`); the default path is the 0.5.1 one.
- `cslcore studio`: a CSL editor in the terminal (Textual) with highlighting, separate Z3 (`F5`)
  and TLA+ (`F8`) checks with animations, suggestions from Z3, TLA+ and the discovered agents that
  apply as patches, a fit and replay view per bound agent (recorded decisions replayed against the
  edited text), agent binding one by one or in bulk (`Ctrl+B`) and going live (`Ctrl+L`, Z3 must
  pass; previous version kept). Opened from setup's policy step (`w`), the watch panel (`o`) and
  the scan's NEXT line.
- Explicit bindings between policies and agents (`.csl/venom/state.json`); one policy can guard
  many agents, each with its own mapping. `venom_guard(agent)` follows a rebinding live.
- Mapping bypass tests: `chimera_core.mapping.in_scope`, `command_allowed` and
  `destination_allowed` classify the derived checks policies rely on (path inside a folder,
  command on an allowlist, destination allowed) and answer NO to anything they cannot read
  unambiguously; generated mappings use them. `cslcore map --test` sends bypass tricks at every
  derived check, each built outside the allowed region so the right answer is known (scope:
  traversal, prefix, relative, home, encoded, NUL, backslash, double slash, dot segments, sibling;
  command: chaining, and / or, pipe, background, substitution, redirect, newline, wrapper, extra
  arguments, glob; destination: credentials, suffix, lookalike, path, scheme, parser confusion,
  control characters, numeric hosts, other port, homograph, separators, header injection, display
  name). Own mappers: `--classify VAR=scope|command|destination[:param]`, `--allowed-root`,
  `--allowed-command`, `--allowed-destination`; a check without an accepted value is reported as
  not covered, never as passing. Regression cases: `--cases FILE`, kept with `--keep-cases` in
  `.csl/venom/cases/<agent>.jsonl` for every later test. Testing an own mapper brings literal
  constants, other functions of the same file and standard-library imports along without running
  the file; setup asks which checks an own mapper computes. Guide: `docs/venom/MAPPING.md`.
- Decision logging checks the log path once per file (about 0.06 ms per call).
- `scripts/venom_demo_traffic.py --count N --seed S` for repeatable traffic.
- New required dependency: `textual` (the studio). Optional extra `csl-core[venom]` (psutil).

### Compatibility
- A contract suite generated from tag v0.5.1 (public API signatures, CLI help and outputs, MCP tool
  schemas and outputs, policy hashes and guard decisions across example and probe policies) runs
  first in CI together with the unmodified v0.5.1 test suite. `runtime.py`, `language/`,
  `engines/` and `factory.py` are untouched.
- The scope planned for 0.6.1 (decision logger, log mode, watch) and 0.6.2 (live MCP, TS / JS
  layer) shipped in this release.

## [0.5.1] - TLA+ Correctness

Found while assembling ground-truth data for an academic writeup of this project: running the real
TLA+ engine (not the mock fallback) against every example policy in the repo surfaced two serious
correctness bugs in `chimera_core/engines/tla_engine/tlc_runner.py` and `verifier.py`, on top of the
`context_mapper.py` and benchmark-methodology issues already queued from the 0.5.0 cycle.

### Fixed
- **TLC stopped at the first invariant violation.** The command line built in
  `TLCRunner._build_command` never passed TLC's `-continue` flag, which is required for TLC to keep
  checking after finding one violation (that flag is *not* TLC's default). A policy with N
  independently-violable rules would report exactly 1 and the other N-1 were never actually
  checked. Fixed by adding `-continue`; the output parser already collected multiple violations
  correctly, it just never received more than one from TLC.
- **A TLC run that failed to complete was silently reported as "verified, no violations."** TLC
  exits with code 151 (not 150) when it hits a construct it cannot evaluate — e.g. a real-number
  literal, since the generated spec only `EXTENDS Integers`. The old exit-code check only
  special-cased 150; every other non-(10,11,12) code, including 151, fell through to "no violations
  parsed => success". Concretely: a policy using `total_balance * 0.1` in a `WHEN` clause caused TLC
  to print `TLC can't handle real numbers.` and exit 151, and CSL-Core reported that policy as
  formally verified safe when TLA+ had never actually checked it at all. This is about as bad as a
  false result gets from a verification tool. Fixed in two places: `TLCRunner.run()` now treats any
  exit code other than 0/10/11/12 as a hard failure with the real TLC output attached, and
  `TLAVerifier.verify()` now checks for this "TLC did not complete" case explicitly and returns a
  `VERIFICATION_ERROR` issue (`all_valid=False`) instead of iterating per-constraint results that
  were never actually computed. Two regression tests added
  (`tests/test_tla_real_integration.py::TestTLCEndToEnd::test_all_violations_reported_not_just_the_first`,
  `::test_unsupported_construct_reported_as_failure_not_silent_success`).
- `chimera_core/plugins/openclaw/context_mapper.py`: `_extract_path_check` and
  `_extract_domain_check` now fail **closed** by default when a tool call's params contain no
  recognizable path-like or URL-like field (previously defaulted to "YES" / allowed). This was
  the exact failure mode an external adversarial test found: an unrecognized param shape silently
  passed as safe. New `OpenClawConfig.strict_unrecognized_fields` (default `True`, env override
  `CSL_STRICT_UNRECOGNIZED_FIELDS`) restores the old fail-open behavior if needed. Cost: an
  integration using nonstandard param names for paths/URLs will now get blocked by policies gating
  on `path_in_workspace`/`domain_allowlisted` rather than silently passing through — false
  positives instead of silent bypasses.
- README / `benchmarks/README.md`: replaced the undisclosed-outlier-filtered "~0.84ms avg" CSL
  latency figure with the unfiltered 22-attack median (~0.78ms), which needs no outlier-exclusion
  threshold. The 0.84 figure was real data (mean of 21/22 attack latencies after excluding one
  78ms outlier below a `<10ms` cutoff in the plotting script), not fabricated, but the cutoff was
  never disclosed next to the headline number; see `paper/PAPER_FACTS.md` §6.5 for the full
  derivation.

### Known limitations (found, not fixed, this release)
- Running real TLC (with the fixes above) against all 20 example policies in this repo: only 1/20
  completes cleanly (a deliberately tautological demo policy). 9/20 (45%) cannot be checked at all
  by TLC as currently generated — 8 because they use a float literal in arithmetic (e.g.
  `amount * 0.1`), which the generated spec's `EXTENDS Integers` cannot evaluate (TLC prints
  `TLC can't handle real numbers.` and exits 151 — now correctly reported as `VERIFICATION_ERROR`
  rather than silently as safe), and 1 (`examples/community/pediatric_safety_guard.csl`) because a
  bare `String`-typed variable declaration (as opposed to an explicit `{"A","B"}` enum set) maps to
  a bogus `{0}` (integer) domain in `tla_spec_builder.py`'s `_domain_to_tla_set`, then crashes when
  compared against a string literal — the equivalent Z3 code path
  (`z3_engine/verifier.py::_register_variables`) handles this correctly (falls back to an
  unconstrained `z3.String`), so this is specifically a TLA+ code-generation gap, not a language-level
  one. Neither of these is fixed in 0.5.1 — both need real feature work (Real-number support needs a
  different TLA+ module/representation than plain `Integers`, and there is no quick, safe patch to
  attempt in the time available for this release) rather than a quick patch, and are left as known,
  disclosed gaps rather than silently deferred. See `paper/PAPER_FACTS.md` §6.11 for the full
  per-policy breakdown.
- Separately (also found this release, not a code bug but an interpretation caveat): even where TLC
  *can* run, its `Init` picks every declared variable fully independently with no correlation
  between them, and `Next` is a stutter step (see PAPER_FACTS.md §4.3). This means a `SAFETY_VIOLATION`
  finding for a real, non-trivial policy typically means "this rule spans 2+ independent variables
  and TLC can freely combine their domains into a combination the rule forbids" — which is close to
  guaranteed for any interesting cross-variable `BLOCK` rule — not necessarily "this specific policy
  has a bug distinguishable from a correctly-authored one." Treat SAFETY_VIOLATION findings against
  real (non-toy) policies with that caveat until the Init/Next model is extended to represent what
  the runtime guard's own enforcement is supposed to prevent.

## [0.5.0] - Stability & Audit Trail

This release closes the audit-trail gap identified by an early production adopter running
CSL-Core as a pre-execution hook over live autonomous agent infrastructure: decisions could
not be tied back to a specific policy or policy version. It does not attempt the larger
"untrusted integration layer" problem (mapping a real tool call into CSL's variable vocabulary)
raised by the same report — that needs a design of its own and stays open post-0.5.0.
Not called 1.0: the language surface (this release adds new CONFIG keys) and the
integration-layer problem are still moving.

### Added
- `CONFIG` keys `POLICY_ID` and `POLICY_VERSION` — optional, stable policy identity independent
  of the domain name, stamped onto every `GuardResult` for audit trails.
- `GuardResult.violated_rule_ids`: rule names that actually produced a BLOCK violation, distinct
  from `triggered_rule_ids` (rules whose `WHEN` condition matched, violated or not). Use
  `violated_rule_ids` as the audit block-reason.
- `CompiledConstitution`/`GuardResult` now stamp a real `policy_hash` (SHA-256 of the compiled
  policy's source text) and `engine_version`. Previously always `None`.
- `--export-tla` flag on `cslcore formal` to export the auto-generated `.tla`/`.cfg` files (TLA+ Toolbox compatible).
- GitHub Actions CI (test matrix on Python 3.10–3.12, lint, package build check).
- `CHANGELOG.md` and `SECURITY.md`.
- README quick start now shows `AND`/`OR` compound `WHEN` conditions (previously only documented in `docs/syntax-spec.md`).

### Fixed
- `GuardResult.policy_hash`, `policy_name`, `policy_id`, `policy_version`, and `engine_version`
  were always `None` regardless of policy content — `CompiledConstitution` never populated them.
  Audit records can now be tied to the exact policy (and, if set, policy version) that produced them.
- `triggered_rule_ids` was previously the only per-decision rule list and was easy to mistake for
  "rules that were violated" (it lists every rule whose `WHEN` matched, including ones that
  passed). Its docstring now clarifies this; use the new `violated_rule_ids` for block-reason reporting.

### Known limitations
- CSL's variables are still enums and bounded ranges only. Mapping a real tool call or API
  request into that vocabulary is left entirely to the integrating application, and adversarial
  testing against a production deployment found that mapping layer to be the actual attack
  surface (verified core held; three bypasses found, all in the untrusted-to-trusted mapping).
  A reference pattern / conformance harness for that layer is on the roadmap, not in this release.

## [0.4.2] - 2026-04-08

### Added
- `--export-tla` flag on the `formal` command.

### Changed
- README contributor cache and download badge formatting.

## [0.4.1] - 2026-04-08

### Added
- `tla_verify` and `universe_info` MCP tools, plus new TLA+ example policies exposed over MCP.

## [0.4.0] - 2026-04-06

### Added
- TLA+ formal verification engine (`chimera_core.engines.tla_engine`): CSL-to-TLA+ translation, real TLC model-checker integration with auto-download, mock BFS fallback, proof certificates, and terminal animations.
- `cslcore formal` CLI command for TLA+ verification with full terminal output.
- VS Code extension with CSL syntax highlighting.
- OpenClaw plugin and deterministic gatekeeper example policy.
- Numerous community policy examples (DevOps deploy guard, DeFi trading/slippage guards, pediatric dosage safety, supply chain provenance, construction site safety, IP whitelist, ecommerce margin guard, API budget circuit breaker, PII output guard).
- Dockerfile for containerized MCP server deployment.

## [0.3.0] - 2026-02-20

Production-ready stable release.

### Added
- Benchmark suite and results.

## [0.3.0-alpha] - 2026-02-17

### Added
- MCP Server (`csl-core-mcp`) exposing `verify_policy`, `simulate_policy`, `explain_policy` for Claude Desktop, Cursor, and VS Code integration.
- Quick Start guide and expanded README documentation.

## [0.2.0-alpha] - 2026-02-09

### Changed
- Refactored verifier logic into `chimera_core.engines.z3_engine`.

## [0.1.0-alpha] - 2026-02-07

### Added
- Initial public release: CSL compiler (parser, AST, validator), Z3-based `LogicVerifier`, deterministic runtime guard (`ChimeraGuard`), and LangChain plugin integration.

[Unreleased]: https://github.com/Chimera-Protocol/csl-core/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/Chimera-Protocol/csl-core/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.0...v0.4.2
[0.4.1]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.0...v0.4.2
[0.4.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.3.0-alpha...v0.3.0
[0.3.0-alpha]: https://github.com/Chimera-Protocol/csl-core/compare/v0.2.0-alpha...v0.3.0-alpha
[0.2.0-alpha]: https://github.com/Chimera-Protocol/csl-core/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0-alpha]: https://github.com/Chimera-Protocol/csl-core/releases/tag/v0.1.0-alpha
