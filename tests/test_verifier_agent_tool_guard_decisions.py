from __future__ import annotations

from chimera_core.runtime import ChimeraGuard, RuntimeConfig
from tests.conftest import compile_policy


def test_runtime_blocks_and_allows_expected_cases(agent_tool_guard_policy_path):
    compiled = compile_policy(agent_tool_guard_policy_path)

    guard = ChimeraGuard(
        compiled,
        config=RuntimeConfig(
            dry_run=False,
            missing_key_behavior="block",
            evaluation_error_behavior="block",
            collect_all_violations=True,
            raise_on_block=False,  # tests: don't throw, just return result
        ),
    )

    # --- BLOCK cases ---
    block_cases = [
        # Non-admin cannot transfer
        {"user_role": "USER", "tool": "TRANSFER_FUNDS", "amount": 100, "approval_token": "YES"},
        # PII present -> cannot email EXTERNAL
        {"tool": "SEND_EMAIL", "pii_present": "YES", "recipient_domain": "EXTERNAL"},
        # Hard-ban delete record
        {"tool": "DELETE_RECORD", "user_role": "ADMIN"},
        # Secrets table forbidden
        {"tool": "QUERY_DB", "db_table": "SECRETS"},
    ]

    for ctx in block_cases:
        res = guard.verify(ctx)
        assert res.allowed is False, f"Expected BLOCK for ctx={ctx}"
        assert len(res.violations) >= 1, f"Expected violations for ctx={ctx}"

    # --- ALLOW cases ---
    allow_cases = [
        # Admin transfer <= 5000 with approval token
        {"user_role": "ADMIN", "tool": "TRANSFER_FUNDS", "amount": 5000, "approval_token": "YES"},
        # Email internal with PII
        {"tool": "SEND_EMAIL", "pii_present": "YES", "recipient_domain": "INTERNAL"},
        # Normal query db allowed
        {"tool": "QUERY_DB", "db_table": "CUSTOMERS"},
    ]

    for ctx in allow_cases:
        res = guard.verify(ctx)
        assert res.allowed is True, f"Expected ALLOW for ctx={ctx}"
