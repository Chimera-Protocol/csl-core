"""
Mapping test: a conformance run over generated cases per
variable, pushed through the mapping and the guard.

Valid inputs must map to the intended value. Malformed or unknown inputs (case variants,
unknown values, missing keys, wrong types, range edges plus and minus one) must end in
BLOCK; any that ends in ALLOW is a fail-open case (finding V11). Works on generated and
hand-written mappings alike: the cases come from the discovered tools and the policy.

Derived values (is the path in scope, is the command allowlisted, is the destination allowed)
get the bypass tricks of tricks.py, built from one value the mapping accepts. A trick that
ends in ALLOW where the policy checks that value is fail-open, reported with its family.
Regression cases (a JSON list of calls with the decision they must get) run last.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...mapping import MappingError
from . import tricks as T
from .spec import PATH_PARAMS, MappingSpec, VarSpec, param_for

TEST_VALUE = "venom-test-value"
TEST_ROOT = "/venom-scope"
TEST_URL = "https://allowed.venom.test/report"
TEST_ADDRESS = "ops@allowed.venom.test"
URL_PARAMS = ("url", "webhook", "endpoint", "link", "href", "uri")
ADDRESS_PARAMS = ("email", "to", "recipient", "recipients", "cc", "bcc", "from")


@dataclass
class Case:
    variable: str
    tool: str
    input: str
    mapped: str
    decision: str  # ALLOW | BLOCK
    expected: str  # ALLOW | BLOCK | "=" (valid input: any decision, value must map)
    kind: str  # valid | malformed | bypass | regression
    ok: bool
    note: str = ""
    family: str = ""  # bypass: the trick family


@dataclass
class Uncovered:
    """A derived value the tricks could not exercise, and why."""
    variable: str
    tool: str
    kind: str
    reason: str


@dataclass
class HarnessResult:
    cases: List[Case] = field(default_factory=list)
    inconclusive: List[str] = field(default_factory=list)  # tools whose valid base call blocks
    uncovered: List[Uncovered] = field(default_factory=list)  # derived values without bypass coverage

    @property
    def fail_open(self) -> List[Case]:
        return [c for c in self.cases if c.expected == "BLOCK" and c.decision == "ALLOW"]

    @property
    def bypass(self) -> List[Case]:
        return [c for c in self.cases if c.kind == "bypass"]

    def families(self) -> Dict[str, Dict[str, int]]:
        """variable -> {family: tricks that got through}; 0 means the family was tested and held."""
        out: Dict[str, Dict[str, int]] = {}
        for c in self.bypass:
            fam = out.setdefault(c.variable, {})
            fam[c.family] = fam.get(c.family, 0) + (1 if c.decision == "ALLOW" else 0)
        return out

    @property
    def failed(self) -> List[Case]:
        return [c for c in self.cases if not c.ok]


def _fmt(v: Any, width: int = 24) -> str:
    if v is _MISSING:
        return "missing"
    if isinstance(v, str):
        v = v.encode("unicode_escape").decode("ascii") if any(ord(ch) < 32 or ord(ch) > 126 for ch in v) else v
        return f'"{v}"' if len(v) <= width else f'"{v[:width - 3]}..."'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, 18) for x in v[:3]) + ("]" if len(v) <= 3 else ", ...]")
    return f"{v!r} {type(v).__name__}"


class _Missing:
    pass


_MISSING = _Missing()


def _variant(s: str) -> str:
    """A spelling the agent never produces: Title_Case, swapped case, or a suffix."""
    cands = ["_".join(w.capitalize() for w in s.split("_")), s.swapcase(), s.upper(), s.lower(), s + "_x"]
    return next(c for c in cands if c != s)


def _base_value(spec: MappingSpec, tool: str, p) -> Any:
    for v in spec.variables:
        if v.source == "param" and tool in v.params and v.params[tool].name == p.name:
            if v.kind == "range":
                return v.low
            if v.kind == "flag":
                return True if (p.type or "").lower() in ("bool", "boolean") else "YES"
            if v.kind == "enum":
                return (p.enum or v.values or [TEST_VALUE])[0]
    t = (p.type or "").lower()
    if t in ("int", "integer"):
        return int(p.minimum) if isinstance(p.minimum, (int, float)) else 0
    if t in ("float", "number"):
        return float(p.minimum) if isinstance(p.minimum, (int, float)) else 0.0
    if t in ("bool", "boolean"):
        return False
    if p.enum:
        return p.enum[0]
    if PATH_PARAMS.match(p.name):
        return f"{TEST_ROOT}/file.txt"
    return TEST_VALUE


def _base_context(spec: MappingSpec) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"approval": "YES"}
    for v in spec.variables:
        if v.source == "context":
            ctx[v.name] = v.low if v.kind == "range" else ("YES" if v.kind == "flag" else (v.values[0] if v.values else TEST_VALUE))
    return ctx


def run(spec: MappingSpec, map_call: Callable, guard, module: Any = None,
        allowed: Optional[Dict[str, List[str]]] = None, cases: Optional[List[Dict[str, Any]]] = None) -> HarnessResult:
    """`allowed`: {"scope": [folder], "command": [cmd], "destination": [url or address]}, values the
    mapping accepts; the bypass tricks are built from the first one of each. Without them the
    test values are used, which generated mappings accept (their allowlists get the test values
    for the run)."""
    res = HarnessResult()
    patched: Dict[str, Any] = {}
    if module is not None:
        for attr, val in (("COMMAND_ALLOWLIST", [TEST_VALUE]),
                          ("DESTINATION_ALLOWLIST", [TEST_VALUE, "allowed.venom.test", TEST_ADDRESS]),
                          ("SCOPE_ROOTS", [TEST_ROOT])):
            if hasattr(module, attr):
                patched[attr] = getattr(module, attr)
                setattr(module, attr, list(getattr(module, attr)) + val)
    try:
        _run(spec, map_call, guard, res, allowed or {})
        _run_bypass(spec, map_call, guard, res, allowed or {})
        for case in cases or []:
            _run_case(map_call, guard, res, case)
    finally:
        for attr, val in patched.items():
            setattr(module, attr, val)
    return res


def _base_for(kind: str, param: str, allowed: Dict[str, List[str]]) -> str:
    given = [v for v in allowed.get(kind) or [] if isinstance(v, str) and v]
    if given:
        if kind != "destination" or len(given) == 1:
            return given[0]
        p = param.lower()
        want = "url" if p in URL_PARAMS else ("address" if p in ADDRESS_PARAMS else "opaque")
        shape = {"url": lambda v: "://" in v, "address": lambda v: "@" in v and "://" not in v,
                 "opaque": lambda v: "@" not in v and "://" not in v}[want]
        return next((v for v in given if shape(v)), given[0])
    if kind == "scope":
        return TEST_ROOT
    if kind == "command":
        return TEST_VALUE
    p = param.lower()
    return TEST_URL if p in URL_PARAMS else (TEST_ADDRESS if p in ADDRESS_PARAMS else TEST_VALUE)


def _relevant(spec: MappingSpec, guard, var: str, ctx: Dict[str, Any]) -> bool:
    """Does the policy look at `var` for this call: do two values of it get different decisions?"""
    v = spec.var(var)
    values = list(v.values) if v is not None and v.values else ["YES", "NO"]
    seen = set()
    for x in values:
        try:
            seen.add(bool(guard.verify({**ctx, var: x}).allowed))
        except Exception:
            seen.add(False)
        if len(seen) > 1:
            return True
    return False


def _run_bypass(spec: MappingSpec, map_call: Callable, guard, res: HarnessResult, allowed: Dict[str, List[str]]) -> None:
    base_ctx = _base_context(spec)
    for var, (kind, explicit) in spec.classify.items():
        for tool, params in spec.tool_params.items():
            if tool not in spec.tool_table:
                continue
            param = param_for(params, kind, explicit)
            base_args = _accepted_args(spec, tool, allowed)
            if param is None:
                _d, ctx, _w = _decide(map_call, guard, tool, base_args, base_ctx)
                if ctx is not None and var in ctx and _relevant(spec, guard, var, ctx):
                    res.uncovered.append(Uncovered(var, tool, kind, "no parameter known for this tool, so every call maps "
                                                                    "to the stricter value (cslcore venom --probe reads "
                                                                    "the parameters; --classify names one)"))
                continue
            base = _base_for(kind, param, allowed)
            controls, ref = [], None
            for good in T.valid(kind, base):
                decision, ctx, why = _decide(map_call, guard, tool, {**base_args, param: good}, base_ctx)
                ref = ref or ctx
                controls.append(Case(var, tool, _fmt(good, 40), why or str((ctx or {}).get(var, "(not set)")), decision,
                                     "ALLOW", "valid", decision == "ALLOW", f"accepted {kind} input", "valid"))
            _d, baseline, _w = _decide(map_call, guard, tool, base_args, base_ctx)
            probe = baseline if baseline is not None and var in baseline else ref
            if probe is None or var not in probe:
                if explicit:  # named on the command line: say why nothing ran
                    res.uncovered.append(Uncovered(var, tool, kind, f"the mapping does not set {var} for this tool"))
                continue
            if not _relevant(spec, guard, var, probe):
                continue  # the policy does not check this value for this tool: nothing to bypass
            if not all(c.ok for c in controls):
                flag = "--allowed-root" if kind == "scope" else f"--allowed-{kind}"
                res.uncovered.append(Uncovered(var, tool, kind, f"the mapping does not accept {_fmt(base, 40)} on {param}; "
                                                                f"pass one it accepts with {flag}"))
                continue
            res.cases += controls
            for trick in T.tricks(kind, base):
                decision, ctx, why = _decide(map_call, guard, tool, {**base_args, param: trick.value}, base_ctx)
                mapped = why or str((ctx or {}).get(var, "(not set)"))
                res.cases.append(Case(var, tool, _fmt(trick.value, 40), mapped, decision, "BLOCK", "bypass",
                                      decision == "BLOCK", f"{trick.family} on {param}", trick.family))


def _computed_by_mapper(spec: MappingSpec, map_call: Callable, v: VarSpec, tool: str, args: Dict[str, Any],
                        base_ctx: Dict[str, Any]) -> bool:
    """A "context" variable the mapping sets on its own: two different supplied values give the same result."""
    choices = list(v.values) if v.values else ([v.low, v.high] if v.kind == "range" else [])
    if len(choices) < 2:
        return False
    seen = []
    for x in choices[:2]:
        try:
            ctx = map_call(tool, dict(args), {**base_ctx, v.name: x})
        except Exception:
            return False
        if not isinstance(ctx, dict) or v.name not in ctx:
            return False
        seen.append(ctx[v.name])
    return seen[0] == seen[1]


def _run_case(map_call: Callable, guard, res: HarnessResult, case: Dict[str, Any]) -> None:
    tool = str(case.get("tool", ""))
    expect = str(case.get("expect", "BLOCK")).upper()
    decision, _ctx, why = _decide(map_call, guard, tool, dict(case.get("args") or {}), dict(case.get("context") or {}))
    label = str(case.get("note") or case.get("id") or "")
    shown = label or _fmt(json.dumps(case.get("args") or {}, sort_keys=True), 40)
    res.cases.append(Case("(regression)", tool, shown, why or "", decision, expect, "regression", decision == expect,
                          str(case.get("id") or ""), "regression"))


def _decide(map_call, guard, tool, args, context) -> Tuple[str, Optional[Dict[str, Any]], str]:
    try:
        ctx = map_call(tool, args, context)
    except MappingError as e:
        return "BLOCK", None, f"(unmapped: {e.reason})"
    except Exception as e:
        return "BLOCK", None, f"(mapping error: {type(e).__name__})"
    if not isinstance(ctx, dict):
        return "BLOCK", None, "(mapping returned no dict)"
    try:
        r = guard.verify(ctx)
        allowed = bool(r.allowed)
    except Exception:
        allowed = False
    return ("ALLOW" if allowed else "BLOCK"), ctx, ""


def _accepted_args(spec: MappingSpec, tool: str, allowed: Dict[str, List[str]]) -> Dict[str, Any]:
    """Base arguments of a valid call: test values, or the operator's accepted values where given."""
    params = spec.tool_params.get(tool, [])
    args = {p.name: _base_value(spec, tool, p) for p in params}
    for var, (kind, explicit) in spec.classify.items():
        if not allowed.get(kind):
            continue
        param = param_for(params, kind, explicit)
        if param is not None and (explicit or param in args):
            args[param] = T.valid(kind, _base_for(kind, param, allowed))[0]
    return args


def _run(spec: MappingSpec, map_call: Callable, guard, res: HarnessResult, allowed: Optional[Dict[str, List[str]]] = None) -> None:
    base_ctx = _base_context(spec)
    tools = [t for t in spec.tool_params]
    base_args = {t: _accepted_args(spec, t, allowed or {}) for t in tools}
    base_decision: Dict[str, str] = {}

    def shown(var, val):
        """Only values inside the declared domain are printed (policy-variable values only)."""
        v = spec.var(var)
        if v is None:
            return "(not a policy variable)"
        if v.kind in ("enum", "flag"):
            return _fmt(val) if isinstance(val, str) and val in v.values else "(outside domain)"
        if v.kind == "range":
            ok = isinstance(val, (int, float)) and not isinstance(val, bool) and v.low <= val <= v.high
            return _fmt(val) if ok else "(outside domain)"
        return "(not shown)"

    def add(var, tool, value, args, context, kind, intended=None, note="", call_tool=None):
        decision, ctx, why = _decide(map_call, guard, tool if call_tool is None else call_tool, args, context)
        mapped = why or (shown(var, ctx.get(var)) if ctx is not None and var in ctx else "(not set)")
        if kind == "valid":
            ok = ctx is not None and (intended is None or ctx.get(var) == intended)
            expected = "="
        else:
            ok = decision == "BLOCK"
            expected = "BLOCK"
        res.cases.append(Case(var, tool, _fmt(value), mapped, decision, expected, kind, ok, note))
        return decision

    tv = spec.tool_var
    for t in tools:
        if t in spec.tool_table:
            base_decision[t] = add(tv or "tool", t, t, base_args[t], base_ctx, "valid", spec.tool_table.get(t) if tv else None)
        else:
            add(tv or "tool", t, t, base_args[t], base_ctx, "malformed", note="not in the policy vocabulary")
    if tools:
        first = next((t for t in tools if base_decision.get(t) == "ALLOW"), tools[0])
        for bad, note in ((_variant(first), "case variant"), ("__unknown_tool__", "unknown tool"), (123, "wrong type")):
            add(tv or "tool", first, bad, base_args[first], base_ctx, "malformed", note=note, call_tool=bad)
    res.inconclusive = [t for t, d in base_decision.items() if d == "BLOCK"]

    for v in spec.variables:
        if v.source == "param":
            for t, p in v.params.items():
                if t not in spec.tool_table:
                    continue
                for value, kind, intended in _param_cases(v, p):
                    args = dict(base_args[t])
                    if value is _MISSING:
                        args.pop(p.name, None)
                    else:
                        args[p.name] = value
                    add(v.name, t, value, args, base_ctx, kind, intended)
        elif v.source == "context":
            t = next((x for x in tools if base_decision.get(x) == "ALLOW"), tools[0] if tools else "")
            if _computed_by_mapper(spec, map_call, v, t, base_args.get(t, {}), base_ctx):
                res.uncovered.append(Uncovered(v.name, t, "context", "the mapping computes it itself instead of taking it "
                                                                     "from the caller; --classify "
                                                                     f"{v.name}=scope|command|destination[:param] adds "
                                                                     "bypass tests for it"))
                continue
            for value, kind, intended in _context_cases(v):
                c = dict(base_ctx)
                if value is _MISSING:
                    c.pop(v.name, None)
                else:
                    c[v.name] = value
                add(v.name, t, value, base_args.get(t, {}), c, kind, intended)


def _param_cases(v: VarSpec, p) -> List[Tuple[Any, str, Any]]:
    out: List[Tuple[Any, str, Any]] = []
    ptype = (p.type or "").lower()
    if v.kind == "range":
        lo, hi = v.low, v.high
        out += [(lo, "valid", lo), (hi, "valid", hi), (str(hi), "valid", hi)]
        out += [(lo - 1, "malformed", None), (hi + 1, "malformed", None), ("not_a_number", "malformed", None),
                (True, "malformed", None), (_MISSING, "malformed", None)]
    elif v.kind == "flag":
        if ptype in ("bool", "boolean"):
            out += [(True, "valid", "YES"), (False, "valid", "NO")]
        else:
            out += [("YES", "valid", "YES"), ("NO", "valid", "NO")]
        out += [("maybe", "malformed", None), (2, "malformed", None), (_MISSING, "malformed", None)]
    elif v.kind == "enum":
        raw = p.enum or v.values
        for val in raw:
            out.append((val, "valid", None))
        if raw:
            out.append((_variant(raw[0]), "malformed", None))
        out += [("__unknown_value__", "malformed", None), (7, "malformed", None), (_MISSING, "malformed", None)]
    return out


def _context_cases(v: VarSpec) -> List[Tuple[Any, str, Any]]:
    if v.kind == "enum":
        return [(x, "valid", x) for x in v.values] + [("__unknown_value__", "malformed", None), (_MISSING, "malformed", None)]
    if v.kind == "range":
        return [(v.low, "valid", v.low), (v.high + 1, "malformed", None), (_MISSING, "malformed", None)]
    if v.kind == "flag":
        return [("YES", "valid", "YES"), ("maybe", "malformed", None), (_MISSING, "malformed", None)]
    return []
