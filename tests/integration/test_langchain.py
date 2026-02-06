from __future__ import annotations

import pytest
from chimera_core.runtime import ChimeraGuard, ChimeraError
from chimera_core.plugins.langchain import ChimeraRunnableGate, wrap_tool

# --- 1. Bağımlılık Kontrolü ---
def _has_langchain_core() -> bool:
    try:
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        return False

# Eğer LangChain yoksa tüm dosyayı atla (Skip)
pytestmark = pytest.mark.skipif(
    not _has_langchain_core(),
    reason="langchain-core not installed; integration test skipped.",
)

# --- 2. Testler (Conftest Fixture'larını Kullanarak) ---

def test_lcel_runnable_gate_pass_through(compiled_agent_tool_guard):
    """
    Test ChimeraRunnableGate using the shared compiled policy from conftest.py
    """
    guard = ChimeraGuard(compiled_agent_tool_guard)

    gate = ChimeraRunnableGate(
        guard,
        # policy-agnostic mapper
        context_mapper=lambda x: x if isinstance(x, dict) else {"content": str(x)},
        # Inject policy-compliant defaults
        inject={
            "tool": "NOOP",
            "user_role": "ADMIN",
            "recipient_domain": "INTERNAL", # Safe
            "target_table": "CUSTOMERS",
            "pii_present": "NO",
            "approval_token": "NO",
        },
    )

    x = {"hello": "world"}
    y = gate.invoke(x)
    assert y is x  # Verify pass-through identity


def test_tool_wrapper_blocks_external_pii(compiled_agent_tool_guard):
    """
    Test Tool Wrapper enforcing DLP rules using the shared policy.
    """
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    guard = ChimeraGuard(compiled_agent_tool_guard)

    # --- Setup Dummy Tool ---
    class EmailArgs(BaseModel):
        recipient_domain: str = Field(...)
        pii_present: str = Field(...)
        user_role: str = Field(default="ADMIN")

    class SendEmailTool(BaseTool):
        name: str = "SEND_EMAIL"
        description: str = "Test tool: send email"
        args_schema: type[BaseModel] = EmailArgs

        def _run(self, recipient_domain: str, pii_present: str, user_role: str = "ADMIN") -> str:
            return "sent"

    # --- Setup Mapper ---
    def mapper(tool_kwargs: dict) -> dict:
        return {
            "tool": "SEND_EMAIL",
            "user_role": tool_kwargs.get("user_role", "ADMIN"),
            "pii_present": tool_kwargs.get("pii_present"),
            "recipient_domain": tool_kwargs.get("recipient_domain"),
        }

    tool = SendEmailTool()
    guarded = wrap_tool(tool, guard, context_mapper=mapper)

    # --- Case A: ALLOW (Internal + PII is OK) ---
    out = guarded.invoke({
        "recipient_domain": "INTERNAL", 
        "pii_present": "YES", 
        "user_role": "ADMIN"
    })
    assert out == "sent"

    # --- Case B: BLOCK (External + PII is Forbidden) ---
    with pytest.raises(ChimeraError) as excinfo:
        guarded.invoke({
            "recipient_domain": "EXTERNAL", 
            "pii_present": "YES", 
            "user_role": "ADMIN"
        })
    
    assert "no_external_email_with_pii" in str(excinfo.value) or "Violation" in str(excinfo.value)