"""A payments agent integrated with CSL-Core 0.5.1 (the pattern of examples/integrations/langchain_agent_demo.py)."""
from typing import Any, Dict

from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


class TransferFundsTool(BaseTool):
    name: str = "TRANSFER_FUNDS"
    description: str = "Transfer money to a recipient."

    def _run(self, amount: int, recipient: str, approval_token: str = "NO") -> str:
        return f"sent {amount} to {recipient}"


class SendEmailTool(BaseTool):
    name: str = "SEND_EMAIL"
    description: str = "Send an email."

    def _run(self, recipient_domain: str, pii_present: str, body: str = "") -> str:
        return "sent"


class QueryDBTool(BaseTool):
    name: str = "QUERY_DB"
    description: str = "Query a database table."

    def _run(self, table_name: str, query: str = "") -> str:
        return "rows"


def agent_context_mapper(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Bridges LangChain tool input to CSL variables (hand-written in 0.5.1)."""
    return {
        "amount": tool_input.get("amount", 0),
        "approval_token": tool_input.get("approval_token", "NO"),
        "recipient_domain": tool_input.get("recipient_domain", "INTERNAL"),
        "pii_present": tool_input.get("pii_present", "NO"),
        "db_table": tool_input.get("table_name", "CUSTOMERS"),
    }


guard = load_guard("policies/agent_tool_guard.csl")
tools = guard_tools([TransferFundsTool(), SendEmailTool(), QueryDBTool()], guard,
                    context_mapper=agent_context_mapper, inject={"user_role": "ADMIN"}, tool_field="tool")
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

if __name__ == "__main__":
    print(llm.invoke("pay the invoice"))
