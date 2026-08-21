# Changelog

All notable changes to CSL-Core are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Chimera-Protocol/csl-core/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.0...v0.4.2
[0.4.1]: https://github.com/Chimera-Protocol/csl-core/compare/v0.4.0...v0.4.2
[0.4.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Chimera-Protocol/csl-core/compare/v0.3.0-alpha...v0.3.0
[0.3.0-alpha]: https://github.com/Chimera-Protocol/csl-core/compare/v0.2.0-alpha...v0.3.0-alpha
[0.2.0-alpha]: https://github.com/Chimera-Protocol/csl-core/compare/v0.1.0-alpha...v0.2.0-alpha
[0.1.0-alpha]: https://github.com/Chimera-Protocol/csl-core/releases/tag/v0.1.0-alpha
