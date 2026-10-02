"""
L6 governance layer: policies, their vocabulary, and hooks.

Policies are parsed with the existing CSL parser only (no compile, no Z3), so reading
them has no side effects.
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Dict, List, Optional, Set, Tuple

from ..model import PolicyRef


def _enum_values(domain: str) -> Optional[List[str]]:
    d = domain.strip()
    if d.startswith("{"):
        return re.findall(r'"([^"]*)"', d)
    return None


def _walk_expr(expr, pairs: Set[Tuple[str, str]], names: Set[str]) -> None:
    from ...language import ast as A

    if expr is None:
        return
    if isinstance(expr, A.Variable):
        names.add(expr.name)
        return
    if isinstance(expr, A.BinaryOp):
        l, r = expr.left, expr.right
        if isinstance(l, A.Variable) and isinstance(r, A.Literal) and isinstance(r.value, str):
            pairs.add((l.name, r.value))
        if isinstance(r, A.Variable) and isinstance(l, A.Literal) and isinstance(l.value, str):
            pairs.add((r.name, l.value))
        _walk_expr(l, pairs, names)
        _walk_expr(r, pairs, names)
        return
    for attr in ("operand", "object", "array", "index", "condition", "then_branch", "else_branch", "body"):
        sub = getattr(expr, attr, None)
        if sub is not None and hasattr(sub, "__dataclass_fields__"):
            _walk_expr(sub, pairs, names)
    for a in getattr(expr, "args", []) or []:
        _walk_expr(a, pairs, names)


def read_policy(path: str, text: str, status: str) -> PolicyRef:
    from ...language.parser import parse_csl
    from ...language import ast as A

    ref = PolicyRef(path=path, status=status)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c = parse_csl(text)
    except Exception as e:
        ref.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        return ref
    ref.domain = c.domain.name if c.domain else None
    ref.policy_hash = getattr(c, "source_hash", None)
    if c.config is not None:
        ref.policy_id = getattr(c.config, "policy_id", None)
        ref.policy_version = getattr(c.config, "policy_version", None)
    for d in (c.domain.variable_declarations if c.domain else []):
        ref.variables[d.name] = d.domain
        vals = _enum_values(d.domain)
        if vals is not None:
            ref.vocabulary[d.name] = vals
    for con in c.constraints or []:
        ref.rules.append(con.name)
        pairs: Set[Tuple[str, str]] = set()
        names: Set[str] = set()
        _walk_expr(con.condition.condition, pairs, names)
        act = con.action
        names.add(act.variable)
        if isinstance(act.value, A.Literal) and isinstance(act.value.value, str):
            pairs.add((act.variable, act.value.value))
        _walk_expr(act.value, pairs, names)
        ref.rule_values[con.name] = sorted(f"{v}={val}" for v, val in pairs) + sorted(f"{n}" for n in names)
    return ref


def policy_label(ref: PolicyRef) -> str:
    return ref.policy_id or ref.domain or ref.path.rsplit("/", 1)[-1]


def scan_policies(probe, roots: List[str], workspace_items: List[Tuple[str, str, str]]) -> List[PolicyRef]:
    """Policies found under the scan roots (status found) plus the workspace's own
    policies/ (active) and .csl/venom/drafts/ (draft), given as (path, text, status)."""
    seen: Dict[str, PolicyRef] = {}
    for path, text, status in workspace_items:
        seen[path] = read_policy(path, text, status)
    for root in roots:
        for path in probe.walk(root, suffixes=(".csl",)):
            if path in seen:
                continue
            text = probe.read_text(path)
            if text is not None:
                seen[path] = read_policy(path, text, "found")
    order = {"active": 0, "draft": 1, "found": 2}
    return sorted(seen.values(), key=lambda p: (order.get(p.status, 3), p.path))
