"""B21 TypeScript / JavaScript code layer."""

from __future__ import annotations

import pytest

from chimera_core.venom.layers.code_js import analyze_js

SHAPES = {
    "mcp_tool": ('''
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
const server = new McpServer({ name: "shop", version: "1.0.0" });
// server.tool("commented_out", {}, async () => ({}))
server.tool("refund_order", "Refund an order", { order_id: z.string(), amount: z.number().int().min(0).max(500) }, async (args) => ({ content: [] }));
''', {"refund_order": [("order_id", "string", None, None), ("amount", "integer", 0, 500)]}),
    "mcp_registerTool": ('''
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
server.registerTool("send_sms", { title: "SMS", description: "Send an SMS", inputSchema: { to: z.string(), body: z.string().max(160) } }, handler);
''', {"send_sms": [("to", "string", None, None), ("body", "string", None, None)]}),
    "vercel_ai": ('''
import { generateText, tool } from "ai";
import { z } from "zod";
const result = await generateText({
  model: openai("gpt-4o"),
  tools: {
    deleteFile: tool({ description: "Delete a file", parameters: z.object({ path: z.string(), force: z.boolean().optional() }), execute: async () => {} }),
    weather: tool({ description: "Get weather", inputSchema: z.object({ city: z.enum(["Paris", "Rome"]) }), execute }),
  },
});
''', {"deleteFile": [("path", "string", None, None), ("force", "boolean", None, None)], "weather": [("city", "string", None, None)]}),
    "langchain_js": ('''
import { DynamicStructuredTool, tool } from "@langchain/core/tools";
const t1 = new DynamicStructuredTool({ name: "run_sql", description: "Run SQL", schema: z.object({ sql: z.string() }), func: async () => "" });
const t2 = tool(async ({ id }) => id, { name: "get_user", description: "Get a user", schema: z.object({ id: z.number() }) });
''', {"run_sql": [("sql", "string", None, None)], "get_user": [("id", "number", None, None)]}),
    "openclaw": ('''
export default function register(api) {
  api.registerTool({ name: "bash_exec", description: "Run a shell command", parameters: Type.Object({ command: Type.String(), timeout: Type.Optional(Type.Number({ minimum: 1, maximum: 600 })) }) });
}
''', {"bash_exec": [("command", "string", None, None), ("timeout", "number", 1, 600)]}),
    "function_schema": ('''
import OpenAI from "openai";
const tools = [{ type: "function", function: { name: "post_tweet", description: "Post publicly", parameters: { type: "object", properties: { text: { type: "string" }, visibility: { type: "string", enum: ["public", "private"] } }, required: ["text"] } } }];
''', {"post_tweet": [("text", "string", None, None), ("visibility", "string", None, None)]}),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_b21_tool_shapes(shape):
    src, expected = SHAPES[shape]
    cf = analyze_js(f"/x/{shape}.ts", src)
    got = {t.name: [(p.name, p.type, p.minimum, p.maximum) for p in t.params] for t in cf.tools}
    assert got == expected


def test_b21_details():
    cf = analyze_js("/x/a.ts", SHAPES["vercel_ai"][0] + '''
const app = express();
app.post("/webhooks/twilio", handler);
const key = process.env.OPENAI_API_KEY;
const agent = new Agent({ instructions: "You are a careful refund assistant for the shop." });
''')
    assert "vercel-ai" in cf.frameworks and cf.model_ids == ["gpt-4o"]
    assert [(r.path, r.handler) for r in cf.routes] == [("/webhooks/twilio", "post")]
    assert cf.env_names == ["OPENAI_API_KEY"] and cf.prompt.present
    weather = next(t for t in cf.tools if t.name == "weather")
    assert weather.params[0].enum == ["Paris", "Rome"]
    assert next(t for t in cf.tools if t.name == "deleteFile").params[1].required is False


def test_b21_scan_js_project(tmp_path):
    proj = tmp_path / "shop-agent"
    proj.mkdir()
    (proj / "package.json").write_text('{"name": "shop-agent"}')
    (proj / "server.ts").write_text(SHAPES["mcp_tool"][0] + "\nawait server.connect(new StdioServerTransport());\n")
    nm = proj / "node_modules" / "x"
    nm.mkdir(parents=True)
    (nm / "decoy.js").write_text(SHAPES["openclaw"][0])
    (proj / "types.d.ts").write_text(SHAPES["openclaw"][0])
    from chimera_core.venom.probe import probe_for
    from chimera_core.venom.scanner import Scanner
    from chimera_core.venom.workspace import Workspace
    probe, roots = probe_for(str(tmp_path))
    inv = Scanner(probe, roots, Workspace(tmp_path / "ws")).run().inventory
    a = inv.agent("shop-agent")
    assert a is not None and [t.name for t in a.tools] == ["refund_order"]
    assert a.tools[0].risk_class == "SPEND" and a.entrypoint.endswith("server.ts")
    assert not (proj / "IMPORTED").exists()
