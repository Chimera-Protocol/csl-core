import anthropic
from chimera_core import load_guard

client = anthropic.Anthropic()
guard = load_guard("policies/membership.csl")

TOOLS = [
    {
        "name": "transfer_funds",
        "description": "Send stablecoins from the treasury to a member wallet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "minimum": 0, "maximum": 100000},
                "to_wallet": {"type": "string"},
                "requires_dual_approval": {"type": "boolean"},
            },
            "required": ["amount", "to_wallet"],
        },
    },
    {
        "name": "check_balance",
        "description": "Read the treasury balance.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def handle(message: str):
    return client.messages.create(model="claude-sonnet-4-5", max_tokens=1024, tools=TOOLS,
                                  system="You manage membership payouts for the DAO. Never pay twice.",
                                  messages=[{"role": "user", "content": message}])


if __name__ == "__main__":
    handle("status")
