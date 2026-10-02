"""
`cslcore hook`: a Claude Code PreToolUse hook backed by a verified policy.

settings.json:

    "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
      "command": "cslcore hook --agent claude-code-ops --policy policies/claude-code-ops.csl --mapping policies/claude_code_ops_mapping.py --workspace /srv/ops"}]}]}

Reads the hook event (JSON) on stdin. In block mode a violation denies the call with the
violated rule names; in log mode the call proceeds and the decision is recorded. Any
internal failure denies in block mode (fail closed) and is logged in log mode.
"""

from __future__ import annotations

import json
import sys

from .observe import venom_guard


def _deny(reason: str) -> int:
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                  "permissionDecisionReason": f"CSL-Core: {reason}"}}
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


def cmd_hook(args) -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        event = {}
    tool = event.get("tool_name") or ""
    tool_input = event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {}
    try:
        if bool(args.policy) != bool(args.mapping):
            raise ValueError("pass both --policy and --mapping, or neither")
        guard = venom_guard(args.agent, policy=args.policy or None, mapping=args.mapping or None,
                            workspace=args.workspace or ".", mode=args.mode)
    except Exception as e:
        if args.mode == "log":
            sys.stderr.write(f"cslcore hook: guard unavailable ({type(e).__name__}); log mode, call proceeds\n")
            return 0
        return _deny(f"guard unavailable ({type(e).__name__}); failing closed")
    result = guard.verify(tool, tool_input, {"session": event.get("session_id")})
    if not result.allowed:  # block mode violation, or an agent / tool disabled by the operator
        return _deny("blocked by " + (", ".join(result.violated_rule_ids) or "policy"))
    return 0
