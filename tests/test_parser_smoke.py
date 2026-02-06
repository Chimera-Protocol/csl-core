from __future__ import annotations

from chimera_core.language.parser import parse_csl_file


def test_parse_agent_tool_guard_smoke(agent_tool_guard_policy_path):
    constitution = parse_csl_file(str(agent_tool_guard_policy_path))

    assert constitution is not None
    assert constitution.domain is not None
    assert getattr(constitution.domain, "name", None) in ("AgentToolGuard", None)  # name field may vary

    constraints = constitution.constraints or []
    assert len(constraints) >= 3, "Expected at least a few constraints in the example policy"

    # sanity: each constraint should have name + condition + action
    for c in constraints:
        assert getattr(c, "name", None), "Constraint missing name"
        assert getattr(c, "condition", None) is not None, "Constraint missing condition"
        assert getattr(c, "action", None) is not None, "Constraint missing action"
