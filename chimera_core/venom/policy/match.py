"""Rank existing policies for an agent (the "use an existing policy" option)."""

from __future__ import annotations

from typing import List, Tuple

from ..analysis.coverage import link_policies, suggest, tool_variable
from ..model import Agent, PolicyRef


def fit(agent: Agent, policy: PolicyRef) -> Tuple[float, str]:
    """0..1 score and a short reason: how well the policy's vocabulary matches the agent's tools."""
    tools = [t for t in agent.tools if not t.name.endswith("/*")]
    if policy.error or not tools:
        return 0.0, "no tools to compare"
    if policy in link_policies(agent, [policy]) and agent.guard.status != "none":
        return 1.0, "already wired to this agent"
    tv = tool_variable(policy, tools)
    vocab = policy.vocabulary.get(tv, []) if tv else []
    exact = sum(1 for t in tools if t.name in vocab)
    close = sum(1 for t in tools if t.name not in vocab and suggest(t.name, vocab))
    params = {p.name for t in tools for p in t.params}
    shared = len(params & set(policy.variables))
    score = (exact + 0.7 * close) / len(tools) * 0.8 + min(shared, 5) / 5 * 0.2
    parts = []
    if exact or close:
        parts.append(f"{exact + close} of {len(tools)} tools named")
    if close:
        parts.append(f"{close} with spelling differences (fixable)")
    if shared:
        parts.append(f"{shared} parameters shared")
    return round(min(score, 0.99), 2), ", ".join(parts) or "little overlap"


def candidates(agent: Agent, policies: List[PolicyRef], limit: int = 3, minimum: float = 0.2) -> List[Tuple[PolicyRef, float, str]]:
    ranked = []
    for p in policies:
        if p.status == "draft":
            continue
        score, why = fit(agent, p)
        if score >= minimum:
            ranked.append((p, score, why))
    ranked.sort(key=lambda x: (-x[1], x[0].path))
    return ranked[:limit]
