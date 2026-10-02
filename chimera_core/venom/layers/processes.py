"""
L4 runtime layer: running processes, containers, listening ports.

Process table via optional psutil, falling back to `ps`. Only process metadata is used
(pid, uid, user, command line); command lines are shown with credential-looking
arguments masked by the redaction pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProcessInfo:
    pid: int
    uid: Optional[int]
    user: str
    etime: str
    args: str
    cwd: Optional[str] = None


@dataclass
class Container:
    name: str
    image: str
    command: str
    ports: str
    state: str


@dataclass
class RuntimeScan:
    processes: List[ProcessInfo] = field(default_factory=list)
    containers: List[Container] = field(default_factory=list)
    listening: Dict[int, List[str]] = field(default_factory=dict)  # pid -> ["127.0.0.1:8080", ...]
    unavailable: Dict[str, str] = field(default_factory=dict)
    sources_ok: List[str] = field(default_factory=list)


# Known assistant and agent runtimes, matched against the command line.
ASSISTANT_BINARIES = [
    ("claude-code", re.compile(r"(^|/)(claude)(\s|$)|@anthropic-ai/claude-code|claude-code/cli")),
    ("claude-desktop", re.compile(r"Claude\.app/Contents/MacOS/Claude(\s|$)")),
    ("cursor", re.compile(r"Cursor\.app/Contents/MacOS/Cursor(\s|$)|(^|/)cursor-agent(\s|$)")),
    ("codex", re.compile(r"(^|/)codex(\s|$)|@openai/codex")),
    ("aider", re.compile(r"(^|/)aider(\s|$)")),
    ("gemini-cli", re.compile(r"(^|/)gemini(\s|$)|@google/gemini-cli")),
    ("openclaw", re.compile(r"(^|/)openclaw(\s|$)|openclaw/")),
    ("goose", re.compile(r"(^|/)goose(\s|$)")),
]
AGENT_CLI = re.compile(r"\b(crewai\s+run|langgraph\s+(dev|up)|autogen|llama_index|agent\.py|agents?/main\.py)\b")
MCP_SERVER = re.compile(r"(mcp-server|server-[a-z\-]+|[\w\-]+-mcp\b|mcp\s+run|fastmcp|csl-core-mcp)")
PERMISSION_SKIP = re.compile(r"--dangerously-skip-permissions|--permission-mode[= ]bypassPermissions|--yolo\b|--full-auto\b")


def assistant_product(args: str) -> Optional[str]:
    # helper processes (renderers, crash handlers) are not the assistant itself
    if "Helper" in args or "crashpad" in args.lower() or "--type=" in args:
        return None
    for product, rx in ASSISTANT_BINARIES:
        if rx.search(args):
            return product
    return None


def parse_ps(text: str) -> List[ProcessInfo]:
    out = []
    for line in text.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        pid, uid, user, etime, args = parts
        out.append(ProcessInfo(int(pid), int(uid) if uid.lstrip("-").isdigit() else None, user, etime, args))
    return out


def parse_lsof_listen(text: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9 or not parts[1].isdigit():
            continue
        addr = parts[8]
        out.setdefault(int(parts[1]), [])
        if addr not in out[int(parts[1])]:
            out[int(parts[1])].append(addr)
    return out


def parse_ss_listen(text: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for line in text.splitlines():
        m = re.search(r"\s(\S+:\d+)\s+\S+\s+users:\(\(\"[^\"]*\",pid=(\d+)", line)
        if m:
            out.setdefault(int(m.group(2)), []).append(m.group(1))
    return out


def scan_runtime(probe, use_psutil: bool = True) -> RuntimeScan:
    scan = RuntimeScan()
    rows = probe.process_table() if use_psutil else None
    procs = [ProcessInfo(*row) for row in rows] if rows is not None else None
    if procs is None:
        r = probe.run(["ps", "-eo", "pid=,uid=,user=,etime=,args="])
        if r is None or r.returncode != 0:
            scan.unavailable["processes"] = "ps not available"
        else:
            procs = parse_ps(r.stdout)
    if procs is not None:
        scan.processes = procs
        scan.sources_ok.append("processes")

    r = probe.run(["docker", "ps", "--all", "--format", "{{json .}}"], timeout=5)
    if r is None or r.returncode != 0:
        scan.unavailable["docker"] = "docker not available"
    else:
        for line in r.stdout.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            scan.containers.append(Container(d.get("Names", ""), d.get("Image", ""), d.get("Command", ""),
                                             d.get("Ports", ""), d.get("State", "")))
        scan.sources_ok.append("docker")

    r = probe.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], timeout=10)
    if r is not None and r.returncode in (0, 1):
        scan.listening = parse_lsof_listen(r.stdout)
        scan.sources_ok.append("sockets")
    else:
        r = probe.run(["ss", "-ltnpH"])
        if r is not None and r.returncode == 0:
            scan.listening = parse_ss_listen(r.stdout)
            scan.sources_ok.append("sockets")
        else:
            scan.unavailable["sockets"] = "lsof and ss not available"
    return scan


def process_cwd(probe, pid: int) -> Optional[str]:
    r = probe.run(["lsof", "-a", "-d", "cwd", "-p", str(pid), "-Fn"], timeout=5)
    if r is None or r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("n/"):
            return line[1:]
    return None


def is_loopback(addr: str) -> bool:
    host = addr.rsplit(":", 1)[0].strip("[]")
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")
