"""
Approvals: a call that needs a person's approval waits here until someone approves it in cslcore watch.

    .csl/venom/approvals.json   {id: {agent, tool, key, shown, rules, at, status, by, decided_at}}

A wired Python agent (the decorator, or guard.check) that makes a call in the approval band does
not run it: a request is recorded and the agent gets an ApprovalPending result. Once approved,
the same call (the same agent, tool and arguments, matched by a hash: the arguments themselves are
never written) runs once within TTL seconds. A request that was denied, used, expired or cannot
be read approves nothing (fail closed).

Claude Code agents do not wait here: their hook answers "ask" and Claude Code asks the person.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .workspace import LockTimeout

TTL = 600  # seconds an approval stays usable
CALL_WAIT = 0.5  # seconds a tool call waits for the state lock at most (then: not approved, stopped)
KEEP = timedelta(days=1)  # decided or stale requests are forgotten after this


def _now() -> datetime:
    return datetime.now(timezone.utc)


def call_key(agent: str, tool: str, args: Dict[str, Any]) -> str:
    """The same agent, tool and arguments; nothing of the arguments can be read back from it."""
    try:
        canon = json.dumps(args or {}, sort_keys=True, default=repr, separators=(",", ":"))
    except (TypeError, ValueError):
        canon = repr(sorted((args or {}).items()))
    return hashlib.sha256(f"{agent}\x00{tool}\x00{canon}".encode()).hexdigest()


class Approvals:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.path = ws.venom / "approvals.json"

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.ws.read(self.path) or "{}")
        except ValueError:
            return {}  # unreadable: nothing is approved
        return data if isinstance(data, dict) else {}

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        cutoff = _now() - KEEP
        keep = {k: v for k, v in data.items() if _when(v.get("decided_at") or v.get("at")) and
                _when(v.get("decided_at") or v.get("at")) > cutoff}
        self.ws.write_text(self.path, json.dumps(keep, indent=1, sort_keys=True) + "\n")

    # the agent's side ------------------------------------------------------------------
    def request(self, agent: str, tool: str, args: Dict[str, Any], shown: Dict[str, Any], rules: List[str]) -> str:
        """Record a request (or find the one already waiting for the same call); its id, or "" when
        the workspace was busy for longer than a tool call waits (the call still does not run)."""
        try:
            lock = self.ws.state_lock(timeout=CALL_WAIT)
            lock.__enter__()
        except LockTimeout:
            return ""
        try:
            key = call_key(agent, tool, args)
            data = self._load()
            for rid, r in data.items():
                if r.get("key") == key and r.get("status") == "pending":
                    return rid
            rid = secrets.token_hex(3)
            data[rid] = {"agent": agent, "tool": tool, "key": key, "shown": shown, "rules": rules,
                         "at": _now().isoformat(timespec="seconds"), "status": "pending"}
            self._save(data)
            return rid
        finally:
            lock.__exit__(None, None, None)

    def consume(self, agent: str, tool: str, args: Dict[str, Any]) -> Optional[str]:
        """An approval for exactly this call, approved less than TTL seconds ago and not used yet:
        mark it used and return its id. Anything else: None."""
        try:
            lock = self.ws.state_lock(timeout=CALL_WAIT)  # an approval is used once, across processes
            lock.__enter__()
        except LockTimeout:
            return None  # busy: not approved this time (the call stops, it never waits)
        try:
            key = call_key(agent, tool, args)
            data = self._load()
            for rid, r in data.items():
                if r.get("key") != key or r.get("status") != "approved":
                    continue
                at = _when(r.get("decided_at"))
                if at is None or _now() - at > timedelta(seconds=TTL):
                    r["status"] = "expired"
                    continue
                r["status"], r["used_at"] = "used", _now().isoformat(timespec="seconds")
                self._save(data)
                return rid
            return None
        finally:
            lock.__exit__(None, None, None)

    # the operator's side ---------------------------------------------------------------
    def pending(self) -> List[Dict[str, Any]]:
        out = []
        for rid, r in self._load().items():
            if r.get("status") == "pending":
                out.append({"id": rid, **r})
        return sorted(out, key=lambda r: r.get("at", ""))

    def decide(self, rid: str, approve: bool, by: str = "") -> bool:
        with self.ws.state_lock():  # one at a time, across processes: an approval is used once
            data = self._load()
            r = data.get(rid)
            if not r or r.get("status") != "pending":
                return False
            r["status"] = "approved" if approve else "denied"
            r["decided_at"], r["by"] = _now().isoformat(timespec="seconds"), by
            self._save(data)
            return True


def _when(iso: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
