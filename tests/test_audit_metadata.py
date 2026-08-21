from __future__ import annotations

from chimera_core.language.parser import parse_csl, parse_csl_file
from chimera_core.language.compiler import CSLCompiler
from chimera_core.runtime import ChimeraGuard, RuntimeConfig


_POLICY_WITH_IDENTITY = """
CONFIG {
  ENFORCEMENT_MODE: BLOCK
  CHECK_LOGICAL_CONSISTENCY: TRUE
  POLICY_ID: "payments.transfer-guard"
  POLICY_VERSION: "1.2.0"
}

DOMAIN IdentityGuard {
  VARIABLES {
    amount: 0..100000
    tier: {"BASIC", "ADMIN"}
  }

  STATE_CONSTRAINT large_amount_requires_admin {
    WHEN amount > 1000
    THEN tier MUST BE "ADMIN"
  }
}
"""


def _compile(policy_path):
    constitution = parse_csl_file(str(policy_path))
    return CSLCompiler().compile(constitution)


def _compile_text(text):
    constitution = parse_csl(text)
    return CSLCompiler().compile(constitution)


def test_policy_hash_is_stamped_and_stable(agent_tool_guard_policy_path):
    compiled_a = _compile(agent_tool_guard_policy_path)
    compiled_b = _compile(agent_tool_guard_policy_path)

    assert compiled_a.policy_hash is not None
    assert len(compiled_a.policy_hash) == 64  # sha256 hex digest
    assert compiled_a.policy_hash == compiled_b.policy_hash


def test_policy_hash_differs_for_different_source(agent_tool_guard_policy_path, banking_policy_path):
    compiled_a = _compile(agent_tool_guard_policy_path)
    compiled_b = _compile(banking_policy_path)
    assert compiled_a.policy_hash != compiled_b.policy_hash


def test_policy_name_defaults_to_domain_name(agent_tool_guard_policy_path):
    compiled = _compile(agent_tool_guard_policy_path)
    assert compiled.policy_name == compiled.domain_name


def test_engine_version_is_stamped(agent_tool_guard_policy_path):
    from chimera_core import __version__

    compiled = _compile(agent_tool_guard_policy_path)
    assert compiled.engine_version == __version__


def test_guard_result_carries_policy_hash(agent_tool_guard_policy_path):
    compiled = _compile(agent_tool_guard_policy_path)
    guard = ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))
    result = guard.verify({
        "user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES",
    })
    assert result.policy_hash == compiled.policy_hash
    assert result.policy_name == compiled.domain_name


def test_triggered_vs_violated_rule_ids_differ_on_compliant_match(agent_tool_guard_policy_path):
    """A rule can trigger (WHEN matched) without being violated (THEN satisfied)."""
    compiled = _compile(agent_tool_guard_policy_path)
    guard = ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))

    result = guard.verify({
        "user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES",
    })

    assert result.allowed is True
    assert result.violated_rule_ids == []
    # At least one rule's WHEN condition matched even though nothing was violated.
    assert len(result.triggered_rule_ids) >= 1


def test_violated_rule_ids_populated_on_block(agent_tool_guard_policy_path):
    compiled = _compile(agent_tool_guard_policy_path)
    guard = ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))

    result = guard.verify({
        "user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES",
    })

    assert result.allowed is False
    assert len(result.violated_rule_ids) >= 1
    # Every violated rule must also have triggered (it can't violate without matching WHEN).
    assert set(result.violated_rule_ids).issubset(set(result.triggered_rule_ids))


def test_policy_id_and_version_default_to_none_when_unset(agent_tool_guard_policy_path):
    """Backward compatibility: policies without CONFIG.POLICY_ID/POLICY_VERSION still compile fine."""
    compiled = _compile(agent_tool_guard_policy_path)
    assert compiled.policy_id is None
    assert compiled.policy_version is None


def test_policy_id_and_version_parsed_from_config():
    compiled = _compile_text(_POLICY_WITH_IDENTITY)
    assert compiled.policy_id == "payments.transfer-guard"
    assert compiled.policy_version == "1.2.0"


def test_guard_result_carries_policy_id_and_version():
    compiled = _compile_text(_POLICY_WITH_IDENTITY)
    guard = ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))
    result = guard.verify({"amount": 500})
    assert result.policy_id == "payments.transfer-guard"
    assert result.policy_version == "1.2.0"
    # policy_hash/policy_name/engine_version still populate alongside the new fields.
    assert result.policy_hash == compiled.policy_hash
    assert result.policy_name == "IdentityGuard"
