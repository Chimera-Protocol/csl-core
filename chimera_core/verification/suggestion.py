from typing import List, Union, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

class SuggestionEngine:
    def __init__(self):
        self.console = Console()

    def report_issues(self, issues: List[Union[str, Dict[str, Any]]]):
        if not issues:
            self._print_success()
            return

        # Filter out pure coverage if it's the only thing (optional)
        self._print_header()

        for i, issue in enumerate(issues, 1):
            self._render_issue(i, issue)

        self._print_footer()

    def _render_issue(self, index: int, issue: Union[str, Dict[str, Any]]):
        if isinstance(issue, str):
            payload = {"kind": "UNKNOWN", "message": issue, "rules": [], "model": None, "unsat_core": None, "meta": None}
        else:
            payload = issue

        kind = payload.get("kind", "UNKNOWN")
        msg = payload.get("message", "")
        rules = payload.get("rules", []) or []
        model = payload.get("model", None)
        unsat_core = payload.get("unsat_core", None)
        meta = payload.get("meta", None)
        severity = payload.get("severity", "error")

        title_color = "red" if severity == "error" else "yellow"
        border = "red" if kind in ("CONTRADICTION", "INTERNAL_ERROR") and severity == "error" else ("yellow" if kind in ("UNSUPPORTED", "UNREACHABLE", "COVERAGE") else "magenta")

        title = f"[bold {title_color}]{kind} #{index}[/bold {title_color}]"

        header = msg
        if rules:
            header = f"[bold]Rules:[/bold] " + ", ".join(rules) + "\n\n" + msg

        self.console.print(Panel(Text(header, style="white"), title=title, border_style=border, width=96))

        if unsat_core:
            self._print_unsat_core(unsat_core)

        if model:
            self._print_model_table(model)

        if meta and kind == "COVERAGE":
            self._print_coverage(meta)

        self._print_suggestions(kind, rules)

    def _print_unsat_core(self, core: List[str]):
        table = Table(title="🧩 UNSAT Core (Assumptions)", show_header=True, header_style="bold cyan", width=96)
        table.add_column("#", style="cyan", width=6)
        table.add_column("Core Literal", style="white")
        for idx, lit in enumerate(core, 1):
            table.add_row(str(idx), str(lit))
        self.console.print(table)
        self.console.print()

    def _print_model_table(self, model: Dict[str, Any]):
        table = Table(title="🧪 Example Assignment (Model)", show_header=True, header_style="bold cyan", width=96)
        table.add_column("Symbol", style="cyan", width=36)
        table.add_column("Value", style="white")
        for k in sorted(model.keys()):
            table.add_row(str(k), str(model[k]))
        self.console.print(table)
        self.console.print()

    def _print_coverage(self, meta: Dict[str, Any]):
        table = Table(title="📊 Verification Coverage", show_header=True, header_style="bold yellow", width=96)
        table.add_column("Metric", style="cyan", width=40)
        table.add_column("Value", style="white")
        for k in [
            "total_constraints",
            "analyzed_pairs",
            "policywide_checks",
            "skipped_pairs_temporal",
            "skipped_pairs_unsupported",
            "unreachable_rules",
            "internally_inconsistent_rules",
        ]:
            if k in meta:
                table.add_row(k, str(meta[k]))
        self.console.print(table)
        self.console.print()

    def _print_suggestions(self, kind: str, rules: List[str]):
        table = Table(title="💡 Suggestions", show_header=True, header_style="bold yellow", width=96)
        table.add_column("Strategy", style="cyan", width=26)
        table.add_column("What to do", style="white")

        if kind == "CONTRADICTION":
            table.add_row(
                "Refine overlap",
                "If two rules can co-trigger, make them mutually exclusive by adding a discriminating predicate."
            )
            table.add_row(
                "Align actions",
                "If they may co-trigger, their THEN constraints on the same variable must be jointly satisfiable."
            )
            table.add_row(
                "Use UNSAT core",
                "Read the UNSAT core literals: they tell you which parts of the rule interaction caused the conflict."
            )

        elif kind == "UNREACHABLE":
            table.add_row(
                "Check domains",
                "Rule condition is unsat under your VARIABLES domains. Relax bounds or fix typos."
            )
            table.add_row(
                "Simplify condition",
                "Remove impossible conjunctions (x > 5 AND x < 3) or mismatched types."
            )

        elif kind == "UNSUPPORTED":
            table.add_row(
                "Rewrite for Core",
                "CSL-Core supports simple boolean/arithmetic comparisons with WHEN/ALWAYS. Remove unsupported features."
            )
            table.add_row(
                "Enterprise path",
                "If you intended temporal/model-checking semantics (TLA+/TLC), keep it for Enterprise."
            )

        elif kind == "COVERAGE":
            table.add_row(
                "Reduce skips",
                "If skipped_pairs_temporal/unsupported is high, the verifier can't fully guarantee consistency."
            )
            table.add_row(
                "Add declarations",
                "Declare key variables with domains to improve Z3 precision and model readability."
            )

        else:
            table.add_row(
                "Review logic",
                "Check for contradictions, missing declarations, or unsupported functions/operators."
            )

        self.console.print(table)
        self.console.print()

    def _print_header(self):
        self.console.print()
        self.console.print(Panel(
            "[bold white]Chimera Logic Verification Report[/bold white]",
            style="bold red",
            subtitle="[red]Verification Findings[/red]",
            width=96
        ))
        self.console.print()

    def _print_success(self):
        self.console.print()
        self.console.print(Panel(
            "[bold green]No issues found.[/bold green]\n"
            "No logical contradictions detected in CSL constraints.",
            style="bold green",
            title="✅ Verification Passed",
            width=96
        ))
        self.console.print()

    def _print_footer(self):
        self.console.print(
            "[dim]Tip: You can skip checks using config.check_logical_consistency = false "
            "but it is [bold red]NOT RECOMMENDED[/bold red] for safety-critical policies.[/dim]"
        )
        self.console.print()
