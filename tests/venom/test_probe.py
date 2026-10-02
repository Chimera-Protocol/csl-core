"""B20 live MCP enumeration (--probe)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .conftest import HOST_OPS, run_cli, scan_fixture

SERVER = Path(__file__).parent / "fixtures" / "mcp" / "stdio_server.py"


def _project(tmp_path, servers) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".mcp.json").write_text(json.dumps({"mcpServers": servers}))
    return proj


def _scan(root, ws, live):
    from chimera_core.venom.probe import probe_for
    from chimera_core.venom.scanner import Scanner
    from chimera_core.venom.workspace import Workspace

    probe, roots = probe_for(str(root))
    return Scanner(probe, roots, Workspace(ws), live_mcp=live).run(), probe


def test_b20_no_probe_starts_nothing(tmp_path):
    result = scan_fixture(HOST_OPS, tmp_path)
    assert result.probe.spawned == [] and result.probe.http_calls == []


def test_b20_stdio_server_listed_and_stopped(tmp_path, capsys):
    pidfile = tmp_path / "server.pid"
    proj = _project(tmp_path, {"shop": {"command": sys.executable, "args": [str(SERVER), str(pidfile)]}})
    # declined without --yes (stdin is not a terminal): nothing starts
    rc, out, _ = run_cli(["venom", "--root", str(proj), "--probe", "--no-save", "--workspace", str(tmp_path)], capsys)
    assert "Start / query them now?" in out and not pidfile.exists()
    res, probe = _scan(proj, tmp_path, live=True)
    agent = next(a for a in res.inventory.agents if a.kind == "assistant")
    tools = {t.name: t for t in agent.tools}
    assert {"query_orders", "issue_refund"} <= set(tools) and "shop/*" not in tools
    assert tools["issue_refund"].risk_class == "SPEND" and tools["query_orders"].risk_class == "READ"
    assert [(p.name, p.maximum) for p in tools["issue_refund"].params] == [("amount", 500)]
    assert len(probe.spawned) == 1 and "mcp" in res.inventory.host.layers_run
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except OSError:
            break
    else:
        raise AssertionError("the probed MCP server is still running")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "id" not in msg:
            self.send_response(202)
            self.end_headers()
            return
        if msg["method"] == "tools/list":
            result = {"tools": [{"name": "send_newsletter", "description": "Email all subscribers.",
                                 "inputSchema": {"type": "object", "properties": {"subject": {"type": "string"}}}}]}
        else:
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
        body = ("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}) + "\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Mcp-Session-Id", "s1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_b20_loopback_http_server(tmp_path):
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
        proj = _project(tmp_path, {"news": {"url": url}})
        res, probe = _scan(proj, tmp_path, live=True)
        agent = next(a for a in res.inventory.agents if a.kind == "assistant")
        t = next(t for t in agent.tools if t.name == "send_newsletter")
        assert t.risk_class == "EXTERNAL" and t.params[0].name == "subject" and t.source == "mcp_live"
        assert probe.http_calls == [url]
    finally:
        httpd.shutdown()


def test_b20_non_loopback_never_queried(tmp_path):
    proj = _project(tmp_path, {"remote": {"url": "http://203.0.113.5/mcp"}})
    res, probe = _scan(proj, tmp_path, live=True)
    assert probe.http_calls == []
    assert "no answer from: remote" in res.inventory.host.layers_unavailable.get("mcp", "")
