# Security Policy

CSL-Core is a deterministic safety/enforcement layer for AI agents. Because policies compiled
with CSL-Core are meant to be a hard boundary an LLM cannot talk its way past, a verification
bypass or an enforcement gap here has outsized impact — please report it privately.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately via one of:

- [GitHub Security Advisories](https://github.com/Chimera-Protocol/csl-core/security/advisories/new) (preferred)
- Email: akarlaraytu@gmail.com

Please include:

- A description of the vulnerability and its impact (e.g. "a crafted `.csl` policy compiles
  successfully but the runtime guard allows an action the policy should block").
- Steps to reproduce, ideally a minimal `.csl` policy and input context.
- The CSL-Core version (`cslcore --version`) and, if relevant, whether Z3 and/or the TLA+/TLC
  engine is involved.

We aim to acknowledge new reports within 5 business days.

## Scope

In scope:
- Logic that lets a runtime `ChimeraGuard.verify()` call return `allowed=True` for a context the
  compiled policy should block (or vice versa causing incorrect denial of safe actions).
- Z3 or TLA+ verification producing a false "VERIFIED"/"HOLDS" result for an inconsistent or
  unsafe policy.
- Parser/compiler issues that allow policy semantics to be altered unexpectedly (e.g. injection
  through untrusted `.csl` source).
- Dependency or supply-chain issues in the published `csl-core` PyPI package.

Out of scope:
- Vulnerabilities requiring the ability to edit the `.csl` policy file itself, or the Python
  process embedding CSL-Core, with a trust level equal to the policy author (CSL-Core does not
  claim to sandbox untrusted policy authors — the LLM/agent is the untrusted party, not the
  developer defining the policy).
- Findings from the community example policies under `examples/community/`, which are
  illustrative and not audited for production use.

## Supported Versions

Security fixes are made against the latest released version on PyPI. Given the project's current
release pace, older minor versions are not backported.
