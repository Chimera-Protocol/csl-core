from __future__ import annotations

from tests.conftest import compile_policy


def test_compile_agent_tool_guard_smoke(agent_tool_guard_policy_path):
    compiled = compile_policy(agent_tool_guard_policy_path)

    # minimum metadata expectations
    assert compiled is not None
    assert getattr(compiled, "domain_name", None) in ("AgentToolGuard", "BankingGuard", None)
    
    # IR or constraints existence (attribute names may vary; keep defensive)
    assert hasattr(compiled, "__dict__"), "Compiled artifact should be a structured object"


def test_compile_banking_case_study_smoke(banking_policy_path):
    compiled = compile_policy(banking_policy_path)
    assert compiled is not None
    assert getattr(compiled, "domain_name", None) in ("BankingGuard", None)
