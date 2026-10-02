"""
The one verification gate every draft passes, whoever wrote it:
parse, validate, Z3 verification with the existing explanations. A draft that fails
the gate can never be activated.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Issue:
    kind: str
    message: str
    rules: List[str] = field(default_factory=list)
    model: Optional[Dict[str, Any]] = None


@dataclass
class GateResult:
    ok: bool
    stage: str  # parse | validate | verify | compile | ok
    policy_hash: Optional[str] = None
    policy_id: Optional[str] = None
    domain: Optional[str] = None
    rules: int = 0
    variables: int = 0
    issues: List[Issue] = field(default_factory=list)


def verify_text(text: str) -> GateResult:
    from ...language.compiler import CSLCompiler
    from ...language.parser import parse_csl

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            ast = parse_csl(text)
    except Exception as e:
        return GateResult(False, "parse", issues=[Issue("PARSE_ERROR", _first_line(e))])
    try:
        with contextlib.redirect_stdout(sink):
            compiled = CSLCompiler().compile(ast)
    except Exception as e:
        issues = _z3_issues(ast) or [Issue(type(e).__name__.upper(), _first_line(e))]
        stage = "verify" if any(i.kind in ("CONTRADICTION", "UNREACHABLE", "UNSUPPORTED", "INTERNAL_ERROR") for i in issues) else "validate"
        return GateResult(False, stage, issues=issues, domain=getattr(ast.domain, "name", None))
    return GateResult(
        True, "ok", policy_hash=compiled.policy_hash, policy_id=compiled.policy_id, domain=compiled.domain_name,
        rules=len(compiled.constraints), variables=len(compiled.variable_domains),
    )


def _first_line(e: Exception) -> str:
    s = str(e).strip()
    return s.splitlines()[0][:300] if s else type(e).__name__


def _z3_issues(ast) -> List[Issue]:
    try:
        from ...engines.z3_engine.verifier import LogicVerifier

        with contextlib.redirect_stdout(io.StringIO()):
            _ok, raw = LogicVerifier().verify(ast)
    except Exception:
        return []
    out = []
    for it in raw or []:
        if it.get("kind") == "COVERAGE" or it.get("severity") not in (None, "error"):
            continue
        out.append(Issue(it.get("kind", "ERROR"), str(it.get("message", "")), list(it.get("rules") or []), it.get("model")))
    return out
