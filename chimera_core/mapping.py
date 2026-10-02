"""
Fail-closed mapping helpers (CSL-Core 0.6).

A mapping turns an agent's real tool call into policy variables. Z3 proves the policy
consistent over its declared domain, but it cannot see what the integration actually
sends: an unknown tool name that maps to nothing, a missing amount, a boolean where the
policy expects "YES"/"NO". If such a value slips through as-is, rules silently stop
matching and the call is allowed.

These helpers never let that happen. A value that cannot be mapped raises MappingError;
the integration treats that as a block (see `guarded_verify`).

    from chimera_core.mapping import to_enum, to_flag, to_range

    ctx = {
        "tool": to_enum(tool_name, {"transfer_funds": "TRANSFER_FUNDS"}, name="tool"),
        "amount": to_range(args.get("amount"), 0, 100_000, name="amount"),
        "dual_approval": to_flag(args.get("requires_dual_approval"), name="dual_approval"),
    }

This module has no dependencies and is safe to import anywhere.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

__all__ = ["MappingError", "to_enum", "to_flag", "to_range", "guarded_verify", "MISSING",
           "in_scope", "command_allowed", "destination_allowed"]


class MappingError(ValueError):
    """A value could not be mapped into the policy domain. Treat as a block."""

    def __init__(self, name: str, value: Any, reason: str) -> None:
        self.name = name
        self.value = value
        self.reason = reason
        super().__init__(f"cannot map {name}: {reason}")


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

_TRUE = {"true", "yes", "y", "1", "on"}
_FALSE = {"false", "no", "n", "0", "off"}
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def _absent(value: Any) -> bool:
    return value is None or value is MISSING


def to_enum(value: Any, allowed: Union[Mapping[str, str], Iterable[str]], *, name: str = "value",
            required: bool = True, default: Optional[str] = None) -> str:
    """
    Map a raw string onto a policy enum value.

    `allowed` is either a dict {raw value: policy value} or an iterable of policy values
    that must match exactly. Matching is exact and case-sensitive on purpose: a variant
    the agent never produces is treated as unknown, not guessed.
    """
    if _absent(value):
        if required or default is None:
            raise MappingError(name, value, "missing")
        return default
    if not isinstance(value, str):
        raise MappingError(name, value, f"expected a string, got {type(value).__name__}")
    table = dict(allowed) if isinstance(allowed, Mapping) else {v: v for v in allowed}
    if value in table:
        return table[value]
    raise MappingError(name, value, "unknown value")


def to_flag(value: Any, *, name: str = "flag", yes: str = "YES", no: str = "NO",
            required: bool = True, default: Optional[str] = None) -> str:
    """Map a boolean-like value onto a two-valued enum such as {"YES", "NO"}."""
    if _absent(value):
        if required or default is None:
            raise MappingError(name, value, "missing")
        if default not in (yes, no):
            raise MappingError(name, value, f"default {default!r} is not {yes!r} or {no!r}")
        return default
    if isinstance(value, bool):
        return yes if value else no
    if isinstance(value, int) and value in (0, 1):
        return yes if value else no
    if isinstance(value, str):
        v = value.strip()
        if v in (yes, no):
            return v
        if v.lower() in _TRUE:
            return yes
        if v.lower() in _FALSE:
            return no
    raise MappingError(name, value, "not a recognised boolean")


def to_range(value: Any, low: float, high: float, *, name: str = "value", integer: bool = True,
             required: bool = True, default: Optional[float] = None) -> Union[int, float]:
    """
    Map a number (or a plain numeric string) into [low, high].

    Booleans, non-numeric strings, NaN and values outside the range raise; nothing is
    clamped, because clamping would turn an out-of-range request into a valid one.
    """
    if _absent(value):
        if required or default is None:
            raise MappingError(name, value, "missing")
        value = default
    if isinstance(value, bool):
        raise MappingError(name, value, "expected a number, got bool")
    if isinstance(value, str):
        s = value.strip()
        if not _NUMBER.match(s):
            raise MappingError(name, value, "not a number")
        value = float(s) if ("." in s) else int(s)
    if not isinstance(value, (int, float)) or value != value:  # NaN
        raise MappingError(name, value, "not a number")
    if integer:
        if isinstance(value, float):
            if not value.is_integer():
                raise MappingError(name, value, "expected a whole number")
            value = int(value)
    if value < low or value > high:
        raise MappingError(name, value, f"outside {low}..{high}")
    return value


# ---------------------------------------------------------------------------
# Classifiers: the derived values a policy relies on (is this path in scope, is this command
# allowlisted, is this destination allowed). They answer NO to anything they cannot read
# unambiguously; every trick family in `cslcore map --test` must come out NO.
# ---------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ENCODED = re.compile(r"%(2e|2f|5c|00)", re.I)


def in_scope(path: Any, roots: Iterable[str], *, yes: str = "YES", no: str = "NO", resolve: bool = False) -> str:
    """
    YES only when `path` is an absolute path inside one of `roots` after normalisation.

    NO for: anything that is not a string, relative paths, `~`, control characters (NUL,
    newline), backslashes, percent-encoded dots or slashes, and every path that leaves the
    root after `..` is resolved. A root matches on whole path segments: `/srv/app-evil` is not
    inside `/srv/app`. With `resolve=True` symlinks are followed on this machine (both sides).
    """
    import os
    import posixpath

    if not isinstance(path, str) or not path or not path.startswith("/"):
        return no
    if _CONTROL.search(path) or "\\" in path or _ENCODED.search(path):
        return no
    p = posixpath.normpath(path)
    if p.startswith("//"):
        return no
    if resolve:
        p = os.path.realpath(p)
    for r in roots or []:
        if not isinstance(r, str) or not r.startswith("/"):
            continue
        root = posixpath.normpath(os.path.realpath(r) if resolve else r)
        if root == "/" or p == root or p.startswith(root.rstrip("/") + "/"):
            return yes
    return no


_SHELL_META = set(";&|`$<>(){}\\*?[]!~#\n\r")
_WRAPPERS = {"sh", "bash", "zsh", "dash", "ksh", "fish", "env", "sudo", "su", "doas", "xargs", "eval", "exec",
             "nohup", "time", "timeout", "nice", "stdbuf", "command", "builtin", "busybox", "chroot", "watch"}


def _argv(command: Any) -> Optional[list]:
    import shlex

    if isinstance(command, (list, tuple)):
        if not command or not all(isinstance(a, str) for a in command):
            return None
        return [a for a in command]
    if not isinstance(command, str) or not command.strip():
        return None
    if _CONTROL.search(command) or any(ch in _SHELL_META for ch in command):
        return None
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:  # unbalanced quotes
        return None
    return argv or None


def command_allowed(command: Any, allowlist: Iterable[Union[str, Iterable[str]]], *, yes: str = "YES",
                    no: str = "NO") -> str:
    """
    YES only when `command` is exactly one allowlisted command (compared word by word).

    A command string with any shell syntax is never allowlisted: `;`, `&&`, `||`, `|`, `&`,
    `$( )`, backticks, redirections, globs, `~`, newlines. Running through a wrapper (`sh -c`, `env`, `sudo`,
    `xargs`, ...) is never allowlisted either, unless the exact wrapped command is listed.

    An allowlist entry ending in ` *` accepts further plain arguments, for example
    `"git log *"`; such entries cannot start with a shell or wrapper.
    """
    argv = _argv(command)
    if argv is None:
        return no
    for entry in allowlist or []:
        want = entry if isinstance(entry, (list, tuple)) else _entry_argv(entry)
        if not want:
            continue
        if want[-1] == "*":
            head = list(want[:-1])
            if head and head[0] not in _WRAPPERS and argv[:len(head)] == head and len(argv) >= len(head):
                return yes
        elif list(want) == argv:
            return yes
    return no


def _entry_argv(entry: Any) -> Optional[list]:
    import shlex

    if not isinstance(entry, str):
        return None
    try:
        return shlex.split(entry, posix=True)
    except ValueError:
        return None


def destination_allowed(destination: Any, allowlist: Iterable[str], *, yes: str = "YES", no: str = "NO",
                        schemes: Iterable[str] = ("https",)) -> str:
    """
    YES only when `destination` names one allowlisted place.

    URLs (`scheme://...`): the scheme must be in `schemes`, the host is compared exactly
    (lower case, trailing dot removed, IDNA) against host entries; `*.example.com` allows
    its subdomains only (list `example.com` itself separately). NO for credentials in the URL (`https://allowed@evil`), backslashes,
    whitespace or control characters, ports other than the default unless the entry names
    the port (`api.example.com:8443`), and hosts that are not exactly listed (numeric IP forms
    included).

    Other destinations (email addresses, channels, wallets, phone numbers) are compared
    exactly (email addresses case-insensitively); lists, separators (`,` `;` whitespace) and
    control characters are NO, so one allowed recipient cannot carry another.
    """
    from urllib.parse import urlsplit

    if isinstance(destination, (list, tuple)):
        return yes if destination and all(destination_allowed(d, allowlist, yes=yes, no=no, schemes=schemes) == yes
                                          for d in destination) else no
    if not isinstance(destination, str) or not destination or destination != destination.strip():
        return no
    if _CONTROL.search(destination) or "\\" in destination or any(ch.isspace() for ch in destination):
        return no
    entries = [e.strip() for e in (allowlist or []) if isinstance(e, str) and e.strip()]
    if "://" not in destination:
        if any(ch in destination for ch in ",;<>\"'()"):
            return no
        if destination.count("@") > 1:
            return no
        d = destination.lower() if "@" in destination else destination
        return yes if any((e.lower() if "@" in e else e) == d for e in entries if "://" not in e) else no
    try:
        u = urlsplit(destination)
        port = u.port
    except ValueError:
        return no
    if u.scheme.lower() not in {s.lower() for s in schemes} or "@" in u.netloc or not u.hostname:
        return no
    host = _host(u.hostname)
    if host is None:
        return no
    default = {"https": 443, "http": 80}.get(u.scheme.lower())
    for e in entries:
        if "@" in e:
            continue
        if "://" in e:
            try:
                eu = urlsplit(e)
                e_host, e_port = eu.hostname or "", eu.port
            except ValueError:
                continue
        else:
            e_host, _, p = e.partition(":")
            e_port = int(p) if p.isdigit() else None
        eh = _host(e_host.lstrip("*.")) if e_host.startswith("*.") else _host(e_host)
        if eh is None:
            continue
        if (port if port is not None else default) != (e_port if e_port is not None else default):
            continue
        if (host.endswith("." + eh) if e_host.startswith("*.") else host == eh):
            return yes
    return no


def _host(h: str) -> Optional[str]:
    h = (h or "").strip().lower().rstrip(".")
    if not h or any(ch in h for ch in "%\\/@ "):
        return None
    try:
        return h.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def guarded_verify(guard: Any, map_call: Callable[..., Dict[str, Any]], tool_name: str,
                   args: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None):
    """
    Map a call and verify it; a mapping failure is a block, never a pass.

    Returns the guard's GuardResult, or a blocked GuardResult naming the mapping error.
    """
    from .runtime import GuardResult

    try:
        ctx = map_call(tool_name, args or {}, context or {})
    except MappingError as e:
        return GuardResult(allowed=False, violations=[str(e)], violated_rule_ids=["__mapping__"])
    except Exception as e:  # a broken mapping must not open the gate
        return GuardResult(allowed=False, violations=[f"mapping failed: {type(e).__name__}"], violated_rule_ids=["__mapping__"])
    return guard.verify(ctx)
