"""Integration snippets per framework (setup step 9)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from .model import Agent


@dataclass
class Snippet:
    agent: str
    title: str
    language: str
    code: str
    notes: List[str]


def snippet_for(agent: Agent, key: str, policy_rel: str, mapping_rel: str, workspace: str) -> Snippet:
    product = agent.framework[0] if agent.framework else ""
    notes = [
        "Starts in log mode: nothing is blocked, every decision is recorded. Review with `cslcore watch`,",
        f"then switch: `cslcore mode --agent {key} block`.",
    ]
    if agent.kind == "assistant" and product == "claude-code":
        hook = {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command":
                f"cslcore hook --agent {key} --workspace {workspace}"}]}]}}
        target = f"{agent.project}/.claude/settings.json" if agent.project else "~/.claude/settings.json"
        return Snippet(agent.display_name, f"Claude Code hook: merge into {target}", "json", json.dumps(hook, indent=2), notes)
    head = (
        "from chimera_core.venom.observe import venom_guard\n\n"
        "# the policy and mapping come from the agent's binding and follow changes live\n"
        f"guard = venom_guard({json.dumps(key)}, workspace={json.dumps(workspace)})\n"
    )
    if any(f in agent.framework for f in ("langchain", "langgraph")):
        code = head + (
            "\n\ndef guarded(tool):\n"
            '    """Run the policy before every call of a LangChain tool."""\n'
            "    original = tool.func\n\n"
            "    def run(*args, **kwargs):\n"
            "        guard.check(tool.name, kwargs)\n"
            "        return original(*args, **kwargs)\n\n"
            "    tool.func = run\n"
            "    return tool\n\n\n"
            "tools = [guarded(t) for t in tools]\n"
        )
        return Snippet(agent.display_name, "LangChain / LangGraph: wrap the tools", "python", code, notes)
    code = head + (
        "\n# before running a tool call requested by the model:\n"
        "guard.check(tool_name, arguments)  # block mode raises PermissionError; log mode records only\n"
    )
    label = ", ".join(agent.framework) or agent.kind
    return Snippet(agent.display_name, f"{label}: check before each tool call", "python", code, notes)


def observe_snippet(agent: Agent, key: str, workspace: str) -> Snippet:
    """For agents that already enforce a policy in their own code (0.5.1 integrations)."""
    import re

    # the agent's own guard line, as found in its code: load_guard("policies/x.csl") and friends
    call = next((e.detail for e in agent.guard.evidence if e.detail and "(" in e.detail), None)
    m = re.match(r"(\w+)\((.*)\)$", call or "")
    original = f'{m.group(1)}("{m.group(2)}")' if m and m.group(2) else 'load_guard("policies/your_policy.csl")'
    from pathlib import Path as _P
    where = next((f"{_P(e.path).name}:{e.line}" for e in agent.guard.evidence if e.line), None)
    code = (
        "from chimera_core.venom.observe import observe\n\n"
        + (f"# {where}\n" if where else "")
        + f"# before:  guard = {original}\n"
        + "guard = observe(\n"
        + f"    {original},\n"
        + f"    agent={json.dumps(key)},\n"
        + f"    workspace={json.dumps(workspace)},\n"
        + ")\n"
        + "\n# guard.verify(context), guard_tools(...) and ChimeraError keep working as before\n"
    )
    notes = [
        "Behaves exactly like your current guard in block mode (same RuntimeConfig, same ChimeraError),",
        "and adds decision logs, log mode and kill switches from `cslcore watch`.",
        "It stays in block mode unless you switch it.",
    ]
    return Snippet(agent.display_name, "already enforcing: add one line for logs and live control", "python", code, notes)
