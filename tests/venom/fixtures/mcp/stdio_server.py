"""Minimal stdio MCP server for the --probe test (no dependencies). Writes its pid next to itself."""
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid())) if len(sys.argv) > 1 else None
TOOLS = [
    {"name": "query_orders", "description": "Read orders from the shop database.",
     "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "integer"}, "status": {"type": "string", "enum": ["open", "paid"]}}}},
    {"name": "issue_refund", "description": "Refund a payment to the customer.",
     "inputSchema": {"type": "object", "properties": {"amount": {"type": "integer", "minimum": 0, "maximum": 500}}, "required": ["amount"]}},
]
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    if msg["method"] == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "shop", "version": "1"}}
    elif msg["method"] == "tools/list":
        result = {"tools": TOOLS}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
