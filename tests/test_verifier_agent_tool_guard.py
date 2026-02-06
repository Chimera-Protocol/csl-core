from __future__ import annotations

from chimera_core.language.parser import parse_csl_file
from chimera_core.verification.verifier import LogicVerifier


def test_verifier_passes_on_agent_tool_guard(agent_tool_guard_policy_path):
    constitution = parse_csl_file(str(agent_tool_guard_policy_path))
    ok, issues = LogicVerifier().verify(constitution)

    assert ok is True, f"Expected verifier OK. Issues: {issues}"


def test_verifier_fails_on_int_string_sort_mismatch(tmp_path):
    # Minimal CSL that creates a sort mismatch:
    # recipient_domain is an enum set, should become String, but we compare it to an int domain or vice versa.
    # We intentionally craft a policy that causes Z3/type conflict or an UNSUPPORTED/INTERNAL_ERROR that must fail-closed.

    bad = tmp_path / "bad_sort.csl"
    bad.write_text(
        """
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  ENABLE_FORMAL_VERIFICATION: FALSE
  ENABLE_CAUSAL_INFERENCE: FALSE
  INTEGRATION: "native"
}

DOMAIN BadSort {
  VARIABLES {
    x: 0..10
    y: {"A", "B"}
  }

  STATE_CONSTRAINT bad_rule {
    WHEN x == "A"
    THEN y == "B"
  }
}
""".strip(),
        encoding="utf-8",
    )

    constitution = parse_csl_file(str(bad))
    ok, issues = LogicVerifier().verify(constitution)

    assert ok is False, "Verifier must fail-closed on type/sort mismatch"
    assert isinstance(issues, list) and len(issues) > 0
