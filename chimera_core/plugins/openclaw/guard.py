"""
OpenClaw Guard — ChimeraPlugin Integration

Inherits from ChimeraPlugin base class. Does NOT reinvent:
  - Z3 verification lifecycle (base handles it)
  - Rich visualization (base handles it)
  - Fail-closed error handling (base handles it)
  - Context normalization pipeline (base handles it)

Only adds:
  - OpenClaw-specific context mapping (tool_name + params + metadata → CSL vars)
  - EvalResult wrapper for the TypeScript bridge
  - Stats tracking for /health endpoint

Usage:
    from chimera_core.plugins.openclaw import OpenClawGuard

    guard = OpenClawGuard("path/to/openclaw_guard.csl")
    result = guard.evaluate("bash", {"command": "rm -rf /"}, {"sender_role": "UNKNOWN"})
    # result.allowed = False
    # result.violations = ["untrusted_no_bash"]
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from chimera_core.language.compiler import CSLCompiler
from chimera_core.runtime import ChimeraError

from ..base import ChimeraPlugin
from .config import OpenClawConfig
from .context_mapper import map_context


@dataclass
class EvalResult:
    """Result of a tool call evaluation.

    Thin wrapper over GuardResult, shaped for the TypeScript bridge
    (JSON serializable, latency in microseconds).

    Attributes:
        allowed: True if the tool call is permitted.
        violations: Constraint names that triggered the block.
        context: The derived CSL context dict (for audit/debug).
        latency_us: Evaluation time in microseconds.
    """

    allowed: bool
    violations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    latency_us: float = 0.0


class OpenClawGuard(ChimeraPlugin):
    """Deterministic gatekeeper for OpenClaw tool execution.

    Inherits ChimeraPlugin's full pipeline:
      normalize → guard.verify → visualize → pass/raise

    Adds OpenClaw-specific context mapping that converts
    before_tool_call event data into CSL policy variables
    using 100% generic extraction (zero tool names in Python).
    """

    def __init__(
        self,
        policy_path: str,
        config: Optional[OpenClawConfig] = None,
        enable_dashboard: bool = False,
    ):
        """Initialize the guard with a CSL policy.

        Args:
            policy_path: Path to a .csl policy file.
            config: Plugin configuration. Uses defaults if None.
            enable_dashboard: Enable Rich terminal visualization.

        Raises:
            FileNotFoundError: If policy file doesn't exist.
            ChimeraError: If policy fails to compile.
        """
        # Validate policy file exists
        policy = Path(policy_path)
        if not policy.exists():
            raise FileNotFoundError(f"CSL policy not found: {policy_path}")

        # Store config
        self._config = config or OpenClawConfig()
        self._config.policy_path = policy_path

        # Compile policy and init base class
        constitution = CSLCompiler.load(str(policy))

        super().__init__(
            constitution=constitution,
            enable_dashboard=enable_dashboard,
            title="CSL-Guard::OpenClaw",
            context_mapper=None,  # We override normalize_input directly
        )

        # Stats
        self._eval_count = 0
        self._block_count = 0

    def normalize_input(self, input_data: Any) -> Dict[str, Any]:
        """Override base normalization with OpenClaw context mapping.

        input_data is expected to be a dict with:
          {"tool_name": str, "tool_params": dict, "metadata": dict}

        The context_mapper extracts 10 CSL variables using pure
        data-type inspection — zero tool names in Python.
        """
        if isinstance(input_data, dict):
            return map_context(
                tool_name=input_data.get("tool_name", ""),
                tool_params=input_data.get("tool_params", {}),
                metadata=input_data.get("metadata", {}),
                config=self._config,
            )
        # Fallback: let base class handle it
        return super().normalize_input(input_data)

    def process(self, input_data: Any) -> EvalResult:
        """ChimeraPlugin abstract method implementation.

        Calls run_guard() from base class, wraps result in EvalResult.
        """
        start = time.perf_counter_ns()

        try:
            guard_result = self.run_guard(input_data)
            elapsed_us = (time.perf_counter_ns() - start) / 1000

            self._eval_count += 1
            return EvalResult(
                allowed=guard_result.allowed,
                violations=list(guard_result.violations) if not guard_result.allowed else [],
                context=self.normalize_input(input_data),
                latency_us=round(elapsed_us, 1),
            )

        except ChimeraError as e:
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            self._eval_count += 1
            self._block_count += 1

            result = EvalResult(
                allowed=False,
                violations=[str(e)],
                context=self.normalize_input(input_data),
                latency_us=round(elapsed_us, 1),
            )

            if self._config.log_blocks:
                tool = input_data.get("tool_name", "?") if isinstance(input_data, dict) else "?"
                self._log_block(tool, result)

            return result

    def evaluate(
        self,
        tool_name: str,
        tool_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Convenience method: evaluate a tool call.

        Wraps process() with a friendlier signature for direct Python usage
        and the TypeScript bridge (server.py).

        Args:
            tool_name: OpenClaw tool name (e.g. "bash", "gmail_delete").
            tool_params: Tool arguments dict. Defaults to empty dict.
            metadata: Event metadata (sender_role, deployment_mode, etc.).

        Returns:
            EvalResult with allowed/blocked status, violations, and timing.
        """
        result = self.process({
            "tool_name": tool_name,
            "tool_params": tool_params or {},
            "metadata": metadata or {},
        })


        return result

    @property
    def stats(self) -> Dict[str, Any]:
        """Return evaluation statistics for /health endpoint."""
        return {
            "total_evaluations": self._eval_count,
            "total_blocks": self._block_count,
            "block_rate": (
                round(self._block_count / self._eval_count, 3)
                if self._eval_count > 0
                else 0.0
            ),
            "policy_path": self._config.policy_path,
            "deployment_mode": self._config.deployment_mode,
        }

    def _log_block(self, tool_name: str, result: EvalResult) -> None:
        """Log blocked actions to stderr."""
        violations_str = ", ".join(result.violations)
        print(
            f"[CSL-Guard] BLOCKED tool={tool_name} "
            f"violations=[{violations_str}] "
            f"latency={result.latency_us}μs",
            file=sys.stderr,
        )
