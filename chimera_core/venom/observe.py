"""
Decision logger and log mode.

`venom_guard(...)` is what the generated integration snippets use:

    from chimera_core.venom.observe import venom_guard

    guard = venom_guard("membership-bot", policy="policies/membership.csl",
                        mapping="policies/membership_bot_mapping.py")
    guard.check(tool_name, args, context)   # raises PermissionError when blocked (block mode)

or, on the tool function itself (what `cslcore wire` writes into an agent's code):

    @guard.tool("transfer_funds")
    def transfer_funds(amount: int, to_wallet: str): ...

The enforcement mode comes from `cslcore mode --agent ID log|block` (state.json) at start.
Log mode uses the existing RuntimeConfig(dry_run=True): nothing is blocked, every decision
is recorded as ALLOW or WOULD_BLOCK. Records carry policy-variable values only, and only
values inside the declared domain; the raw call payload is never written.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ..mapping import MappingError
from ..runtime import ChimeraError, ChimeraGuard, GuardResult, RuntimeConfig
from .workspace import Workspace

OUTSIDE = "(outside domain)"


def _domain_value(domain: str, value: Any) -> Any:
    d = str(domain).strip()
    if d.startswith("{"):
        return value if isinstance(value, str) and value in re.findall(r'"([^"]*)"', d) else OUTSIDE
    m = re.fullmatch(r"\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)\s*", d)
    if m:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and float(m.group(1)) <= value <= float(m.group(2))
        return value if ok else OUTSIDE
    return OUTSIDE


class DecisionLogger:
    def __init__(self, workspace: Workspace, agent_id: str, guard: ChimeraGuard, mode: str) -> None:
        self.ws = workspace
        self.agent_id = agent_id
        self.mode = mode
        self.path = workspace.decision_log(agent_id)
        self.domains = dict(getattr(guard.constitution, "variable_domains", {}) or {})
        self.policy_hash = getattr(guard.constitution, "policy_hash", None)

    def values(self, ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ctx:
            return {}
        return {k: _domain_value(dom, ctx[k]) for k, dom in self.domains.items() if k in ctx}

    def record(self, tool: str, decision: str, rules, latency_ms: float, ctx: Optional[Dict[str, Any]] = None,
               error: Optional[str] = None, mode: Optional[str] = None, raw_ok: bool = False) -> Dict[str, Any]:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            # the raw tool name is logged only when the mapping (or the operator's control) accepted it
            "agent": self.agent_id, "tool": tool if (ctx is not None or raw_ok) else OUTSIDE,
            "decision": decision, "rules": list(rules or []), "latency_ms": round(latency_ms, 4),
            "policy_hash": self.policy_hash, "mode": mode or self.mode, "values": self.values(ctx),
        }
        if error:
            rec["error"] = error
        self.ws.append_line(self.path, json.dumps(rec, sort_keys=True))
        return rec


class VenomGuard:
    """A guard, a mapping and a decision logger, following the live control plane."""

    def __init__(self, agent_id: str, compiled, map_call: Callable, logger: DecisionLogger, controls,
                 policy_path=None, workspace: Optional[Workspace] = None, follow_binding: bool = False,
                 mapping_path=None) -> None:
        self.agent_id = agent_id
        self.map_call = map_call
        self.logger = logger
        self.controls = controls
        self.policy_path = policy_path
        self.mapping_path = mapping_path
        self.ws = workspace
        self.follow_binding = follow_binding
        self._state_mtime = workspace.state_mtime() if workspace is not None else None
        self._policy_mtime = workspace.mtime(policy_path) if workspace is not None and policy_path is not None else None
        self._mapping_mtime = workspace.mtime(mapping_path) if workspace is not None and mapping_path is not None else None
        self._install(compiled)

    def _install(self, compiled) -> None:
        self.log_guard = ChimeraGuard(compiled, RuntimeConfig(dry_run=True, raise_on_block=False))
        self.block_guard = ChimeraGuard(compiled, RuntimeConfig(raise_on_block=False))
        self.logger.domains = dict(getattr(compiled, "variable_domains", {}) or {})
        self.logger.policy_hash = getattr(compiled, "policy_hash", None)

    def _rebind_if_changed(self) -> None:
        """Follow `cslcore policy bind` / studio "go live": switch to the newly bound policy and mapping."""
        if not self.follow_binding or self.ws is None:
            return
        sm = self.ws.state_mtime()
        if sm == self._state_mtime:
            return
        self._state_mtime = sm
        from .bindings import Bindings
        b = Bindings(self.ws).get(self.agent_id)
        if b is None or not b.mapping:
            return
        policy, mapping = Bindings(self.ws).abs(b.policy), Bindings(self.ws).abs(b.mapping)
        if policy == self.policy_path and mapping == self.mapping_path:
            return
        try:
            compiled = _compile_quiet(self.ws.read(policy) or "")
            module = self.ws.load_module(mapping)
        except Exception:
            return  # keep the last good policy and mapping
        self.policy_path, self.mapping_path, self.map_call = policy, mapping, module.map_call
        self._policy_mtime, self._mapping_mtime = self.ws.mtime(policy), self.ws.mtime(mapping)
        self._install(compiled)

    def _reload_if_changed(self) -> None:
        """Pick up a policy changed by `cslcore policy activate` or the panel; keep the old one if the new one fails.

        Venom's own changes bump state.json, so one stat per call notices them at once; files
        edited outside Venom are checked at most once a second."""
        if self.ws is None:
            return
        sm = self.ws.state_mtime()
        now = time.monotonic()
        if sm == getattr(self, "_seen_state", None) and now - getattr(self, "_last_file_check", 0.0) < 1.0:
            return
        self._seen_state = sm
        self._last_file_check = now
        self._rebind_if_changed()
        if self.ws is not None and self.mapping_path is not None:
            mm = self.ws.mtime(self.mapping_path)
            if mm is not None and mm != self._mapping_mtime:
                self._mapping_mtime = mm
                try:
                    self.map_call = self.ws.load_module(self.mapping_path).map_call
                except Exception:
                    pass
        if self.ws is None or self.policy_path is None:
            return
        m = self.ws.mtime(self.policy_path)
        if m is None or m == self._policy_mtime:
            return
        self._policy_mtime = m
        text = self.ws.read(self.policy_path)
        if text is None:
            return
        try:
            compiled = _compile_quiet(text)
        except Exception:
            return  # the active guard keeps working with the last good policy
        self._install(compiled)

    @property
    def mode(self) -> str:
        return self.controls.refresh().mode

    @property
    def guard(self) -> ChimeraGuard:
        return self.log_guard if self.mode == "log" else self.block_guard

    def verify(self, tool_name: str, args: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> GuardResult:
        """Decide one tool call. `allowed` is False only when the call must not run."""
        self._reload_if_changed()
        ctl = self.controls.refresh()
        mode = ctl.mode
        t0 = time.perf_counter()
        if ctl.exempt and not ctl.disabled:
            self.logger.record(tool_name, "ALLOW", ["__exempt__"], 0.0, None, mode=mode, raw_ok=True)
            return GuardResult(allowed=True, warnings=["exempted by operator"], triggered_rule_ids=["__exempt__"])
        # assistants call MCP tools as mcp__<server>__<tool>: a disabled tool covers both spellings
        base = tool_name.rsplit("__", 1)[-1] if tool_name.startswith("mcp__") else tool_name
        if ctl.disabled or tool_name in ctl.disabled_tools or base in ctl.disabled_tools:
            rule = "__agent_disabled__" if ctl.disabled else "__tool_disabled__"
            ms = (time.perf_counter() - t0) * 1000
            known = tool_name in ctl.disabled_tools or base in ctl.disabled_tools
            self.logger.record(tool_name if known else OUTSIDE, "BLOCK", [rule], ms, None, mode=mode, raw_ok=True)
            return GuardResult(allowed=False, violations=[f"{rule.strip('_').replace('_', ' ')} by operator"],
                               violated_rule_ids=[rule])
        try:
            ctx = self.map_call(tool_name, args or {}, context or {})
        except MappingError as e:
            ms = (time.perf_counter() - t0) * 1000
            decision = "WOULD_BLOCK" if mode == "log" else "BLOCK"
            self.logger.record(tool_name, decision, ["__mapping__"], ms, None, error=e.reason, mode=mode)
            return GuardResult(allowed=mode == "log", violations=[str(e)], violated_rule_ids=["__mapping__"],
                               enforcement="DRY_RUN" if mode == "log" else "ACTIVE")
        guard = self.log_guard if mode == "log" else self.block_guard
        try:
            result = guard.verify(ctx)
        except ChimeraError as e:
            result = e.result or GuardResult(allowed=False, violations=[str(e)], violated_rule_ids=[e.constraint_name])
        ms = (time.perf_counter() - t0) * 1000
        violated = list(result.violated_rule_ids)
        if not violated and not result.allowed:
            violated = ["__blocked__"]
        if mode == "log":
            decision = "WOULD_BLOCK" if violated else "ALLOW"
        else:
            decision = "ALLOW" if result.allowed else "BLOCK"
        self.logger.record(tool_name, decision, violated, ms, ctx, mode=mode)
        return result

    def check(self, tool_name: str, args: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> GuardResult:
        """verify(), raising PermissionError when the call must not run."""
        result = self.verify(tool_name, args, context)
        if not result.allowed:
            raise PermissionError(f"blocked by policy: {', '.join(result.violated_rule_ids) or 'violation'}")
        return result

    def tool(self, name: str) -> Callable:
        """Decorator for a tool function: the policy decides every call before the function runs.
        It keeps the function's name, docstring and signature, so frameworks that read them
        (LangChain's @tool, OpenAI Agents' @function_tool) see the same tool. Put it directly
        above `def`, under the framework's own decorator."""
        import functools
        import inspect

        def wrap(fn: Callable) -> Callable:
            sig = inspect.signature(fn)

            def arguments(args, kwargs) -> Dict[str, Any]:
                try:
                    bound = sig.bind_partial(*args, **kwargs).arguments
                except TypeError:
                    return dict(kwargs)
                out: Dict[str, Any] = {}
                for k, v in bound.items():
                    kind = sig.parameters[k].kind
                    if kind is inspect.Parameter.VAR_KEYWORD and isinstance(v, dict):
                        out.update(v)
                    elif k not in ("self", "cls") and kind is not inspect.Parameter.VAR_POSITIONAL:
                        out[k] = v
                return out

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def guarded_async(*args, **kwargs):
                    self.check(name, arguments(args, kwargs))
                    return await fn(*args, **kwargs)
                return guarded_async

            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                self.check(name, arguments(args, kwargs))
                return fn(*args, **kwargs)
            return guarded
        return wrap


def _compile_quiet(text: str):
    import contextlib
    import io

    from ..language.compiler import CSLCompiler
    from ..language.parser import parse_csl

    with contextlib.redirect_stdout(io.StringIO()):
        return CSLCompiler().compile(parse_csl(text))


def current_mode(workspace: Workspace, agent_id: str, default: str = "log") -> str:
    from .controls import Controls
    return Controls(workspace).get(agent_id, default).mode


def venom_guard(agent_id: str, *, policy: Optional[str] = None, mapping: Optional[str] = None, workspace: str = ".",
                mode: Optional[str] = None) -> VenomGuard:
    """Build the guard an agent uses at start.

    Without `policy` / `mapping` the agent's binding is used (`cslcore policy bind`, studio "go
    live") and followed live. Paths are relative to the workspace. `mode` pins the enforcement
    mode; otherwise it follows `cslcore mode` and the watch panel live."""
    import contextlib
    import io

    from ..factory import load_guard
    from .controls import AgentControl, LiveControls

    ws = Workspace(workspace)
    follow = policy is None
    if follow:
        from .bindings import Bindings
        b = Bindings(ws).get(agent_id)
        if b is None or not b.mapping:
            raise LookupError(f"{agent_id} is not bound to a policy in {ws.root}: cslcore policy bind <policy> --agent {agent_id}")
        policy, mapping = b.policy, b.mapping
    elif mapping is None:
        raise ValueError("pass both policy and mapping, or neither to use the agent's binding")
    policy_path = (ws.root / policy) if not Path(policy).is_absolute() else Path(policy)
    mapping_path = (ws.root / mapping) if not Path(mapping).is_absolute() else Path(mapping)
    with contextlib.redirect_stdout(io.StringIO()):  # the compiler reports progress on stdout
        compiled = load_guard(str(policy_path)).constitution
    module = ws.load_module(mapping_path)
    controls = LiveControls(ws, agent_id, mode or "log")
    if mode:
        pinned = controls.refresh

        def refresh():
            c = pinned()
            return AgentControl(mode, c.disabled, c.disabled_tools, c.exempt)
        controls.refresh = refresh  # type: ignore[method-assign]
    logger = DecisionLogger(ws, agent_id, ChimeraGuard(compiled), mode or "log")
    return VenomGuard(agent_id, compiled, module.map_call, logger, controls, policy_path=policy_path.resolve(), workspace=ws,
                      follow_binding=follow, mapping_path=mapping_path.resolve())


# ---------------------------------------------------------------------------
# observe(): one line for 0.5.1 integrations
# ---------------------------------------------------------------------------

class ObservedGuard(ChimeraGuard):
    """
    A drop-in for an existing ChimeraGuard (0.5.1 integrations):

        guard = observe(load_guard("policies/payments.csl"), agent="payments-bot")

    Same `verify(context)` API and, in block mode, exactly the wrapped guard's behaviour
    (its RuntimeConfig, its ChimeraError). Adds decision logs, log mode and the operator's
    kill switches from `cslcore watch` / `cslcore mode`. Without an explicit mode it stays
    in BLOCK: adding observation never weakens an integration that already enforces.
    """

    def __init__(self, guard: ChimeraGuard, agent: str, workspace: str = ".", tool_field: str = "tool",
                 mode: Optional[str] = None) -> None:
        from .controls import AgentControl, LiveControls

        super().__init__(guard.constitution, guard.config)
        self.inner = guard
        self.agent = agent
        self.tool_field = tool_field
        c = guard.config
        self.log_guard = ChimeraGuard(guard.constitution, RuntimeConfig(
            raise_on_block=False, collect_all_violations=True, missing_key_behavior=c.missing_key_behavior,
            evaluation_error_behavior=c.evaluation_error_behavior, dry_run=True))
        ws = Workspace(workspace)
        self.controls = LiveControls(ws, agent, "block")
        if mode:
            inner_refresh = self.controls.refresh

            def pinned():
                cur = inner_refresh()
                return AgentControl(mode, cur.disabled, cur.disabled_tools, cur.exempt)
            self.controls.refresh = pinned  # type: ignore[method-assign]
        self.logger = DecisionLogger(ws, agent, guard, "block")

    @property
    def mode(self) -> str:
        return self.controls.refresh().mode

    def _tool(self, context: Any) -> str:
        if not isinstance(context, dict):
            return OUTSIDE
        raw = context.get(self.tool_field)
        if raw is None:
            return OUTSIDE
        dom = self.logger.domains.get(self.tool_field)
        return str(raw) if dom is None else str(_domain_value(dom, raw))

    def verify(self, context: Dict[str, Any]) -> GuardResult:
        ctl = self.controls.refresh()
        tool = self._tool(context)
        t0 = time.perf_counter()
        if ctl.exempt and not ctl.disabled:
            self.logger.record(tool, "ALLOW", ["__exempt__"], 0.0, context, mode=ctl.mode, raw_ok=True)
            return GuardResult(allowed=True, warnings=["exempted by operator"], triggered_rule_ids=["__exempt__"])
        if ctl.disabled or (tool != OUTSIDE and tool in ctl.disabled_tools):
            rule = "__agent_disabled__" if ctl.disabled else "__tool_disabled__"
            self.logger.record(tool, "BLOCK", [rule], (time.perf_counter() - t0) * 1000, context, mode=ctl.mode, raw_ok=True)
            result = GuardResult(allowed=False, violations=[f"{rule.strip('_').replace('_', ' ')} by operator"], violated_rule_ids=[rule])
            if self.inner.config.raise_on_block and not self.inner.config.dry_run:
                raise ChimeraError(result.violations[0], rule, context, result=result)
            return result
        if ctl.mode == "log":
            r = self.log_guard.verify(context)
            self.logger.record(tool, "WOULD_BLOCK" if r.violated_rule_ids else "ALLOW", r.violated_rule_ids,
                               (time.perf_counter() - t0) * 1000, context, mode="log", raw_ok=True)
            return r
        try:
            r = self.inner.verify(context)
        except ChimeraError as e:
            rules = list(e.result.violated_rule_ids) if e.result is not None else [e.constraint_name]
            self.logger.record(tool, "BLOCK", rules, (time.perf_counter() - t0) * 1000, context, mode="block", raw_ok=True)
            raise
        decision = "ALLOW" if r.allowed else "BLOCK"
        self.logger.record(tool, decision, r.violated_rule_ids, (time.perf_counter() - t0) * 1000, context, mode="block", raw_ok=True)
        return r


def observe(guard: ChimeraGuard, agent: str, workspace: str = ".", tool_field: str = "tool",
            mode: Optional[str] = None) -> ObservedGuard:
    """Wrap an existing guard with decision logs, log mode and kill switches (see ObservedGuard)."""
    return ObservedGuard(guard, agent, workspace=workspace, tool_field=tool_field, mode=mode)
