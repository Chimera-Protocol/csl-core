"""
Suggestions the studio can apply with one key: every suggestion carries a patch (a function
from the current text to the new text). After a patch the policy is verified again.

Sources:
  z3     contradictions (separate the overlapping rules), unreachable rules (widen the domain
         or remove the rule)
  tla    guard reading of TLA+: strict policies the compiler would refuse, rules that never fire
  agent  vocabulary drift (real tool names), risky tools without a rule, tools missing from
         the tool enum
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..policy import edit as E
from .engines import TLARun, Z3Run

Patch = Callable[[str], Optional[str]]


@dataclass
class Suggestion:
    source: str  # z3 | tla | agent
    title: str
    explanation: str
    confidence: str = "MEDIUM"  # HIGH | MEDIUM | LOW
    rule: Optional[str] = None
    patch: Optional[Patch] = field(default=None, repr=False)

    @property
    def applicable(self) -> bool:
        return self.patch is not None


RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def rule_line(text: str, rule: str) -> Optional[int]:
    m = re.search(rf"(?m)^[ \t]*STATE_CONSTRAINT\s+{re.escape(rule)}\b", text)
    return text.count("\n", 0, m.start()) + 1 if m else None


def _when(text: str, rule: str) -> Optional[str]:
    span = E._block(text, rule)
    if not span:
        return None
    block = text[span[0]:span[1]]
    m = re.search(r"(?m)^\s*WHEN\s+(.+?)\s*$", block)
    if m:
        return m.group(1)
    if re.search(r"(?m)^\s*ALWAYS\s+True\s*$", block):
        return "True"
    return None


def _add_condition(rule: str, cond: str) -> Patch:
    def apply(text: str) -> Optional[str]:
        span = E._block(text, rule)
        if not span:
            return None
        block = text[span[0]:span[1]]
        if re.search(r"(?m)^\s*ALWAYS\s+True\s*$", block):
            new = re.sub(r"(?m)^(\s*)ALWAYS\s+True\s*$", lambda m: f"{m.group(1)}WHEN {cond}", block, count=1)
        else:
            new = re.sub(r"(?m)^(\s*WHEN\s+)(.+?)\s*$", lambda m: f"{m.group(1)}({m.group(2)}) AND {cond}", block, count=1)
        return text[:span[0]] + new + text[span[1]:]
    return apply


def _remove(rule: str, why: str) -> Patch:
    return lambda text: E.remove_rules(text, [rule], why) if E._block(text, rule) else None


def _replace(before: str, after: str) -> Patch:
    def apply(text: str) -> Optional[str]:
        if before in text:
            return text.replace(before, after, 1)
        # tolerate indentation differences
        b = "\n".join(l.strip() for l in before.strip().splitlines())
        lines = text.splitlines()
        stripped = [l.strip() for l in lines]
        n = len(before.strip().splitlines())
        for i in range(len(lines) - n + 1):
            if "\n".join(stripped[i:i + n]) == b:
                indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                block = "\n".join(indent + l.strip() for l in after.strip().splitlines())
                return "\n".join(lines[:i] + [block] + lines[i + n:]) + ("\n" if text.endswith("\n") else "")
        return None
    return apply


def from_z3(run: Z3Run, text: str) -> List[Suggestion]:
    out: List[Suggestion] = []
    for a, b in run.conflicts:
        wb, wa = _when(text, b), _when(text, a)
        if wb and wb != "True":
            out.append(Suggestion("z3", f"Keep {a} out of {b}'s situation",
                                  f"{a} and {b} can trigger together and then demand opposite things. Adding "
                                  f"NOT ({wb}) to {a} makes them mutually exclusive; {b} decides in the overlap.",
                                  "HIGH", a, _add_condition(a, f"NOT ({wb})")))
        if wa and wa != "True":
            out.append(Suggestion("z3", f"Keep {b} out of {a}'s situation",
                                  f"The same, the other way round: {a} decides in the overlap.",
                                  "MEDIUM", b, _add_condition(b, f"NOT ({wa})")))
        out.append(Suggestion("z3", f"Disable {b}", f"Last resort: comment {b} out (kept in the file).", "LOW", b,
                              _remove(b, f"disabled in the studio: contradicted {a}")))
    for r in run.unreachable:
        w = _when(text, r) or ""
        widened = _widen_for(text, w)
        if widened:
            var, title, patch = widened
            out.append(Suggestion("z3", title, f"{r} can never trigger: its WHEN needs a value of {var} that the "
                                  "declared domain does not allow.", "HIGH", r, patch))
        out.append(Suggestion("z3", f"Remove {r}", f"{r} can never trigger, so it protects nothing.", "MEDIUM", r,
                              _remove(r, "removed in the studio: unreachable")))
    for issue in run.issues:
        if issue.kind in ("PARSE_ERROR", "VALIDATION_ERROR"):
            out.append(Suggestion("z3", "Fix the syntax first", _parse_hint(issue.message), "HIGH"))
    return out


def _widen_for(text: str, when: str):
    """An enum literal or a number outside the declared domain, and a patch that widens it."""
    for var, val in re.findall(r'\b(\w+)\s*==\s*"([^"]+)"', when):
        m = re.search(rf'(?m)^(\s*{re.escape(var)}\s*:\s*)\{{([^}}]*)\}}', text)
        if m and f'"{val}"' not in m.group(2):
            def patch(t: str, var=var, val=val) -> Optional[str]:
                mm = re.search(rf'(?m)^(\s*{re.escape(var)}\s*:\s*)\{{([^}}]*)\}}', t)
                return t[: mm.end(2)] + f', "{val}"' + t[mm.end(2):] if mm else None
            return var, f'Add "{val}" to {var}', patch
    for var, op, num in re.findall(r"\b(\w+)\s*(>=|>|<=|<)\s*(-?\d+)", when):
        m = re.search(rf"(?m)^(\s*{re.escape(var)}\s*:\s*)(-?\d+)\.\.(-?\d+)", text)
        if not m:
            continue
        lo, hi, n = int(m.group(2)), int(m.group(3)), int(num)
        need_hi = n + 1 if op == ">" else n
        need_lo = n - 1 if op == "<" else n
        if op in (">", ">=") and need_hi > hi:
            def patch(t: str, var=var, lo=lo, new=need_hi) -> Optional[str]:
                return re.sub(rf"(?m)^(\s*{re.escape(var)}\s*:\s*)-?\d+\.\.-?\d+", lambda x: f"{x.group(1)}{lo}..{new}", t, count=1)
            return var, f"Widen {var} to {lo}..{need_hi}", patch
        if op in ("<", "<=") and need_lo < lo:
            def patch(t: str, var=var, hi=hi, new=need_lo) -> Optional[str]:
                return re.sub(rf"(?m)^(\s*{re.escape(var)}\s*:\s*)-?\d+\.\.-?\d+", lambda x: f"{x.group(1)}{new}..{hi}", t, count=1)
            return var, f"Widen {var} to {need_lo}..{hi}", patch
    return None


def _parse_hint(message: str) -> str:
    low = message.lower()
    if "expected" in low and "}" in message:
        return f"{message}. A block is probably not closed: every '{{' needs a '}}'."
    if "expected" in low and "string" in low:
        return f"{message}. Enum values are written in double quotes: {{\"A\", \"B\"}}."
    return message


def _loosen(text: str) -> Optional[str]:
    new = re.sub(r"ENABLE_FORMAL_VERIFICATION\s*:\s*TRUE", "ENABLE_FORMAL_VERIFICATION: FALSE", text, count=1)
    return new if new != text else None


def from_tla(run: TLARun) -> List[Suggestion]:
    """Guard reading of TLA+: a rule reachable states break is a rule that blocks (that is the
    point of a guard), so it needs no fix. Two things do deserve a suggestion."""
    out: List[Suggestion] = []
    if run.refuses_to_load:
        out.append(Suggestion(
            "tla", "Let the guard enforce: ENABLE_FORMAL_VERIFICATION: FALSE",
            f"With TRUE the compiler requires every rule to hold over the whole domain and refuses to load "
            f"this policy, because {', '.join(run.enforced[:3])} block real states. The studio already runs "
            f"TLA+ for you (F8); FALSE keeps the rules as runtime checks.", "HIGH", None, _loosen))
    for name in run.never_fires:
        out.append(Suggestion(
            "tla", f"{name} never blocks anything",
            "No reachable state breaks this rule: the domain already keeps it true. It is harmless, but it "
            "guards nothing; tighten its THEN or remove it.", "LOW", name,
            _remove(name, "never fires: no reachable state breaks it (TLA+)")))
    return out


def from_fit(fit, text: str) -> List[Suggestion]:
    """Suggestions from the agent fit (see fit.py)."""
    out: List[Suggestion] = []
    if fit is None:
        return out
    if fit.renames:
        pairs = ", ".join(f"{a} → {b}" for a, b in list(fit.renames.items())[:3])
        out.append(Suggestion("agent", "Use the agent's real tool names",
                              f"The policy names tools the agent never sends ({pairs}); those rules would never match.",
                              "HIGH", None, lambda t, r=dict(fit.renames): _fix(t, r)))
    if fit.missing_from_enum and fit.tool_var:
        names = ", ".join(fit.missing_from_enum[:4])
        out.append(Suggestion("agent", "Add the agent's other tools to the policy",
                              f"Not in {fit.tool_var}: {names}. Tools outside the enum are blocked by the mapping.",
                              "MEDIUM", None, _extend_enum(fit.tool_var, fit.missing_from_enum)))
    for tool, cls in fit.uncovered_risky[:3]:
        out.append(Suggestion("agent", f"Add a rule for {tool} ({cls})",
                              f"{tool} is {cls} and no rule speaks about it.", "MEDIUM", None, fit.rule_patch(tool)))
    return out


def _fix(text: str, renames) -> str:
    from ..policy.draft import fix_text
    return fix_text(text, renames, {})


def _extend_enum(var: str, names: List[str]) -> Patch:
    def apply(text: str) -> Optional[str]:
        m = re.search(rf'(?m)^(\s*{re.escape(var)}\s*:\s*)\{{([^}}]*)\}}', text)
        if not m:
            return None
        vals = re.findall(r'"([^"]*)"', m.group(2))
        vals += [n for n in names if n not in vals]
        return text[: m.start(2)] + ", ".join(f'"{v}"' for v in vals) + text[m.end(2):]
    return apply


def ranked(items: List[Suggestion]) -> List[Suggestion]:
    return sorted(items, key=lambda s: (RANK.get(s.confidence, 3), {"z3": 0, "tla": 1, "agent": 2}.get(s.source, 3)))
