"""
Redaction pass: runs before anything is written or printed.

Venom never stores credential values or prompt text in the first place; this pass is the
second line of defense for strings that come from command lines (process arguments,
cron and unit commands), where a value can appear inline.
"""

from __future__ import annotations

import re
from typing import Any

MASK = "***"

_PATTERNS = [
    # NAME=value where NAME looks like a secret
    (re.compile(r"\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL|AUTH|DSN)[A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)", re.I), r"\1=" + MASK),
    # --token value / --api-key=value
    (re.compile(r"(--?(?:api[-_]?key|token|password|passwd|secret|auth|access[-_]key|client[-_]secret)(?:=|\s+))(\"[^\"]*\"|'[^']*'|\S+)", re.I), r"\1" + MASK),
    # Authorization headers
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-~+/]+=*", re.I), r"\1" + MASK),
    # well-known token shapes
    (re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{12,})"), MASK),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,})"), MASK),
    (re.compile(r"\b(xox[abposr]-[A-Za-z0-9\-]{10,})"), MASK),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})"), MASK),
    (re.compile(r"\b(AIza[0-9A-Za-z_\-]{30,})"), MASK),
    # credentials inside URLs: scheme://user:pass@host
    (re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+(@)", re.I), r"\1" + MASK + r"\2"),
]


def text(s: str) -> str:
    for rx, repl in _PATTERNS:
        s = rx.sub(repl, s)
    return s


def deep(obj: Any) -> Any:
    if isinstance(obj, str):
        return text(obj)
    if isinstance(obj, list):
        return [deep(x) for x in obj]
    if isinstance(obj, dict):
        return {k: deep(v) for k, v in obj.items()}
    return obj
