"""
CSL Core - Runtime Visualizer (Elite Grade)

Terminal dashboard for runtime decisions.
Deterministic, compact, and forward-compatible.
"""

from typing import Dict, Any, List, Iterable, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..runtime import GuardResult


class RuntimeVisualizer:
    def __init__(
        self,
        width: int = 92,
        max_value_len: int = 80,
        max_items: int = 50,
        max_depth: int = 3,
    ):
        self.console = Console()
        self.width = width
        self.max_value_len = max_value_len
        self.max_items = max_items
        self.max_depth = max_depth

    def visualize(self, result: GuardResult, context: Dict[str, Any], title: str = "Chimera Gatekeeper"):
        self.console.print()

        allowed = bool(getattr(result, "allowed", False))
        warnings = list(getattr(result, "warnings", []) or [])
        violations = list(getattr(result, "violations", []) or [])
        latency_ms = float(getattr(result, "latency_ms", 0.0) or 0.0)

        # --- Header status ---
        if allowed:
            if warnings:
                header_style = "bold yellow"
                status_text = "⚠️  ALLOWED WITH WARNINGS"
                border_style = "yellow"
            else:
                header_style = "bold green"
                status_text = "✅ REQUEST ALLOWED"
                border_style = "green"
        else:
            header_style = "bold red"
            status_text = "⛔ REQUEST BLOCKED"
            border_style = "red"

        self.console.print(
            Panel(
                Text(status_text, justify="center", style=header_style),
                title=f"[white]{title}[/]",
                border_style=border_style,
                width=self.width,
            )
        )

        # --- Policy meta (optional, forward compatible) ---
        meta = self._extract_meta(result)
        if meta:
            meta_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", width=self.width)
            meta_table.add_column("Meta", style="cyan", width=26)
            meta_table.add_column("Value", style="white")
            for k, v in meta:
                meta_table.add_row(k, v)
            self.console.print(meta_table)

        # --- Context (flatten + deterministic order) ---
        ctx_table = Table(
            title="Input Context",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            width=self.width,
        )
        ctx_table.add_column("Key", style="cyan", width=34)
        ctx_table.add_column("Value", style="white")

        flat_items = self._flatten_context(context)
        shown = 0
        for key, val in flat_items:
            if shown >= self.max_items:
                ctx_table.add_row("…", f"(truncated after {self.max_items} items)")
                break
            ctx_table.add_row(key, self._format_value(val))
            shown += 1

        self.console.print(ctx_table)

        # --- Triggered rules (optional) ---
        triggered = getattr(result, "triggered_rules", None) or getattr(result, "triggered_rule_ids", None)
        if triggered:
            self.console.print()
            t = Table(title="Triggered Rules", box=box.ROUNDED, style="cyan", width=self.width, show_header=True)
            t.add_column("#", style="dim", width=4)
            t.add_column("Rule", style="cyan")
            for i, r in enumerate(list(triggered), 1):
                t.add_row(str(i), str(r))
            self.console.print(t)

        # --- Violations / Warnings ---
        if not allowed:
            self.console.print()
            err_table = Table(title="Violation Details", box=box.ROUNDED, style="red", width=self.width, show_header=True)
            err_table.add_column("#", style="dim", width=4)
            err_table.add_column("Violation", style="red")
            for i, msg in enumerate(violations, 1):
                err_table.add_row(str(i), self._format_value(msg))
            self.console.print(err_table)

        if warnings:
            self.console.print()
            warn_table = Table(title="Policy Warnings (Non-Blocking)", box=box.ROUNDED, style="yellow", width=self.width)
            warn_table.add_column("Warning", style="yellow")
            for msg in warnings:
                warn_table.add_row(self._format_value(msg))
            self.console.print(warn_table)

        # --- Footer metrics ---
        self.console.print()
        latency_color = "green" if latency_ms < 10 else ("yellow" if latency_ms < 50 else "red")
        enforcement = getattr(result, "enforcement", None) or "ACTIVE"
        footer = Text.assemble(
            ("Latency: ", "dim"),
            (f"{latency_ms:.3f}ms", f"bold {latency_color}"),
            (" | ", "dim"),
            ("Enforcement: ", "dim"),
            (str(enforcement), "bold white"),
        )
        self.console.print(footer, justify="right", width=self.width)
        self.console.print()

    def _extract_meta(self, result: GuardResult) -> List[Tuple[str, str]]:
        """Try to show policy versioning / hash / engine info if available."""
        keys = [
            ("Policy", "policy_name"),
            ("Policy ID", "policy_id"),
            ("Policy Version", "policy_version"),
            ("Policy Hash", "policy_hash"),
            ("Domain", "domain_name"),
            ("Engine", "engine_version"),
        ]
        out = []
        for label, attr in keys:
            v = getattr(result, attr, None)
            if v is not None:
                out.append((label, self._format_value(v)))
        return out

    def _flatten_context(self, ctx: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Flatten nested dicts/lists deterministically with depth limit."""
        items: List[Tuple[str, Any]] = []

        def walk(prefix: str, val: Any, depth: int):
            if depth > self.max_depth:
                items.append((prefix, "(max depth reached)"))
                return

            if isinstance(val, dict):
                for k in sorted(val.keys(), key=lambda x: str(x)):
                    walk(f"{prefix}.{k}" if prefix else str(k), val[k], depth + 1)
            elif isinstance(val, list):
                # Keep deterministic order; cap list expansion
                cap = min(len(val), 10)
                for i in range(cap):
                    walk(f"{prefix}[{i}]", val[i], depth + 1)
                if len(val) > cap:
                    items.append((f"{prefix}[…]", f"(truncated list: {len(val)} items)"))
            else:
                items.append((prefix, val))

        walk("", ctx, 0)
        return items

    def _format_value(self, v: Any) -> str:
        s = str(v)
        if len(s) > self.max_value_len:
            s = s[: self.max_value_len - 3] + "..."
        return s