"""
CSL-Core OpenClaw Server — Bridge Daemon

Two modes for TypeScript ↔ Python communication:

1. STDIO mode (default): JSON line protocol over stdin/stdout.
   Launched by the TypeScript plugin as a child process.

2. HTTP mode: Lightweight HTTP server on localhost.
   For long-running daemon deployments.

Usage:
    # STDIO mode (TypeScript spawns this)
    python -m chimera_core.plugins.openclaw.server --policy openclaw_guard.csl

    # HTTP mode
    python -m chimera_core.plugins.openclaw.server --policy openclaw_guard.csl --http --port 9100

Protocol (STDIO):
    Input:  {"tool": "bash", "params": {...}, "metadata": {...}}
    Output: {"allowed": false, "violations": ["untrusted_no_bash"], "latency_us": 52.3}
"""

import argparse
import json
import sys

from .guard import OpenClawGuard
from .config import OpenClawConfig


def run_stdio(guard: OpenClawGuard) -> None:
    """Run in STDIO mode — read JSON lines from stdin, write results to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _write_response({"error": f"Invalid JSON: {e}"})
            continue

        tool_name = request.get("tool", "")
        tool_params = request.get("params", {})
        metadata = request.get("metadata", {})

        if not tool_name:
            _write_response({"error": "Missing 'tool' field"})
            continue

        result = guard.evaluate(tool_name, tool_params, metadata)
        _write_response({
            "allowed": result.allowed,
            "violations": result.violations,
            "latency_us": result.latency_us,
        })


def run_http(guard: OpenClawGuard, host: str = "127.0.0.1", port: int = 9100) -> None:
    """Run in HTTP mode — lightweight server on localhost."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/evaluate":
                self.send_error(404, "Use POST /evaluate")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                request = json.loads(body)
            except json.JSONDecodeError as e:
                self._respond(400, {"error": f"Invalid JSON: {e}"})
                return

            tool_name = request.get("tool", "")
            tool_params = request.get("params", {})
            metadata = request.get("metadata", {})

            if not tool_name:
                self._respond(400, {"error": "Missing 'tool' field"})
                return

            result = guard.evaluate(tool_name, tool_params, metadata)
            self._respond(200, {
                "allowed": result.allowed,
                "violations": result.violations,
                "latency_us": result.latency_us,
            })

        def do_GET(self):
            if self.path == "/health":
                self._respond(200, {"status": "ok", **guard.stats})
                return
            self.send_error(404)

        def _respond(self, code: int, data: dict):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            # Suppress default access log, use stderr for blocks only
            pass

    server = HTTPServer((host, port), Handler)
    print(f"[CSL-Guard] HTTP server listening on {host}:{port}", file=sys.stderr)
    print("[CSL-Guard] POST /evaluate — GET /health", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CSL-Guard] Shutting down.", file=sys.stderr)
        server.server_close()


def _write_response(data: dict) -> None:
    """Write JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description="CSL-Core OpenClaw Guard Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Path to .csl policy file",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP mode (default: STDIO)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9100,
        help="HTTP port (default: 9100)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--deployment-mode",
        default="DESKTOP",
        choices=["DESKTOP", "SERVER", "EMBEDDED", "UNATTENDED"],
        help="Deployment mode (default: DESKTOP)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable sandbox mode",
    )

    args = parser.parse_args()

    config = OpenClawConfig(
        deployment_mode=args.deployment_mode,
        sandbox_active=args.sandbox,
    )

    print(f"[CSL-Guard] Loading policy: {args.policy}", file=sys.stderr)
    guard = OpenClawGuard(args.policy, config=config)
    print(f"[CSL-Guard] Policy compiled. Mode: {args.deployment_mode}", file=sys.stderr)

    if args.http:
        run_http(guard, host=args.host, port=args.port)
    else:
        run_stdio(guard)


if __name__ == "__main__":
    main()
