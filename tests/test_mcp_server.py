from __future__ import annotations

import json

import pytest

from chimera_core.mcp.server import (
    verify_policy,
    simulate_policy,
    explain_policy,
    scaffold_policy,
    universe_info,
    example_hello_world,
    example_agent_tool_guard,
    example_banking_guard,
    example_tla_demo,
    csl_expert,
)


@pytest.fixture(scope="module")
def policy_text(agent_tool_guard_policy_path):
    return agent_tool_guard_policy_path.read_text(encoding="utf-8")


def test_verify_policy_success(policy_text):
    result = verify_policy(policy_text)
    assert "VERIFIED" in result
    assert "Domain" in result


def test_verify_policy_parse_error():
    result = verify_policy("this is not valid CSL {{{")
    assert "PARSE ERROR" in result


def test_simulate_policy_single_allow(policy_text):
    ctx = {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    result = simulate_policy(policy_text, json.dumps(ctx))
    assert "ALLOWED" in result or "allowed" in result.lower()


def test_simulate_policy_single_block(policy_text):
    ctx = {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"}
    result = simulate_policy(policy_text, json.dumps(ctx))
    assert "BLOCKED" in result or "blocked" in result.lower()


def test_simulate_policy_batch(policy_text):
    ctx = [
        {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"},
        {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"},
    ]
    result = simulate_policy(policy_text, json.dumps(ctx))
    assert "Batch Summary" in result


def test_simulate_policy_invalid_json(policy_text):
    result = simulate_policy(policy_text, "{not json")
    assert "INVALID JSON" in result


def test_simulate_policy_compile_error():
    result = simulate_policy("not valid csl", "{}")
    assert "COMPILATION FAILED" in result


def test_explain_policy(policy_text):
    result = explain_policy(policy_text)
    assert isinstance(result, str) and len(result) > 0


def test_explain_policy_parse_error():
    result = explain_policy("not valid csl {{{")
    assert "PARSE ERROR" in result


def test_scaffold_policy():
    result = scaffold_policy(
        domain_name="SpendGuard",
        description="limit spending per user tier",
        variables="amount, role",
    )
    assert "SpendGuard" in result
    assert "amount" in result


def test_scaffold_policy_no_variable_hints():
    result = scaffold_policy(domain_name="EmptyGuard", description="no hints given")
    assert "EmptyGuard" in result


def test_universe_info(policy_text):
    result = universe_info(policy_text)
    assert "Universe Analysis" in result
    assert "Total state space size" in result


def test_universe_info_parse_error():
    result = universe_info("not valid csl {{{")
    assert "PARSE ERROR" in result


def test_example_resources():
    assert "DOMAIN" in example_hello_world()
    assert "DOMAIN" in example_agent_tool_guard()
    assert "DOMAIN" in example_banking_guard()
    assert "DOMAIN" in example_tla_demo()


def test_csl_expert_prompt():
    result = csl_expert()
    assert "CSL" in result
