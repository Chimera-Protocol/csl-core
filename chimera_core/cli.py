from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import asdict, is_dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from . import __version__
from .language.compiler import CSLCompiler, CompilationError
from .language.parser import parse_csl_file
from .engines.z3_engine import LogicVerifier
from .runtime import ChimeraGuard, ChimeraError, RuntimeConfig
from .audit.visualizer import RuntimeVisualizer


console = Console()


# -----------------------
# Utilities
# -----------------------
def _run_z3_debug_trace(policy_path: str):
    """
    Best-effort Z3 debug trace:
    We re-run parse + verifier in debug mode to surface sort mismatches and encoding issues.
    This is intentionally separate from compiler.compile() to avoid changing the core pipeline.
    """
    console.print()
    console.print(Panel("[bold yellow]Z3 Debug Trace Enabled[/bold yellow]\n(Showing tail events only)", border_style="yellow", width=92))

    try:
        constitution = parse_csl_file(policy_path)
    except Exception as e:
        _print_error("Debug trace failed", f"Could not parse policy for debug: {_safe_to_str(e, 2000)}")
        return

    try:
        verifier = LogicVerifier()
        ok, issues = verifier.verify(constitution, debug=True)
    except Exception as e:
        _print_error("Debug trace failed", f"Verifier crashed while tracing: {_safe_to_str(e, 2000)}")
        return

    # Print INTERNAL_ERROR trace tails if present
    printed_any = False
    for it in issues:
        if it.get("kind") == "INTERNAL_ERROR":
            meta = it.get("meta") or {}
            tail = meta.get("trace_tail") or []
            if tail:
                printed_any = True
                t = Table(title="Z3 Trace Tail", box=box.SIMPLE, show_header=True, header_style="bold yellow", width=92)
                t.add_column("#", style="dim", width=6)
                t.add_column("event", style="yellow", width=16)
                t.add_column("rule", style="cyan", width=22)
                t.add_column("data", style="white")
                for idx, row in enumerate(tail, 1):
                    ev = _safe_to_str(row.get("event", ""), 60)
                    rl = _safe_to_str(row.get("rule", ""), 60)
                    # show remaining keys compactly
                    data = {k: v for k, v in row.items() if k not in ("event", "rule")}
                    t.add_row(str(idx), ev, rl, _safe_to_str(data, 400))
                console.print(t)

    if not printed_any:
        console.print("[dim]No INTERNAL_ERROR trace tail available. If the verifier returned UNSUPPORTED/CONTRADICTION, use the normal report output.[/dim]")
    console.print()

def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def _load_json_from_arg(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("JSON input must be an object/dict (e.g. {'amount': 5000}).")
    return obj


def _load_json_from_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file '{path}': {e}") from e


def _safe_to_str(x: Any, max_len: int = 200) -> str:
    s = str(x)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _print_banner():
    console.print()
    console.print(
        Panel(
            f"[bold white]CSL-Core[/bold white]  [dim]v{__version__}[/dim]\n"
            "[dim]Deterministic Safety Layer for Probabilistic AI Systems[/dim]",
            border_style="cyan",
            width=92,
        )
    )
    console.print()


def _print_error(title: str, message: str):
    console.print(
        Panel(
            f"[bold red]{title}[/bold red]\n{message}",
            border_style="red",
            width=92,
        )
    )


def _print_success(title: str, message: str):
    console.print(
        Panel(
            f"[bold green]{title}[/bold green]\n{message}",
            border_style="green",
            width=92,
        )
    )


def _print_kv_table(title: str, rows: List[List[str]]):
    t = Table(title=title, box=box.SIMPLE, show_header=True, header_style="bold cyan", width=92)
    t.add_column("Key", style="cyan", width=28)
    t.add_column("Value", style="white")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

def _result_to_json(result: Any, *, context: Dict[str, Any], compiled: Any) -> Dict[str, Any]:
    """
    Best-effort serializer for runtime verification outputs.
    Stays resilient even if internal dataclasses change.
    """
    out: Dict[str, Any] = {
        "allowed": bool(getattr(result, "allowed", False)),
        "domain": getattr(compiled, "domain_name", None),
        "policy_hash": getattr(compiled, "policy_hash", None),
        "policy_version": getattr(compiled, "policy_version", None),
        "engine_version": getattr(compiled, "engine_version", None),
        "context": context,
        "violations": [],
        "warnings": [],
    }

    # violations / warnings: try dict-like, dataclass, or str fallback
    for key in ("violations", "warnings"):
        items = getattr(result, key, None) or []
        norm = []
        for it in items:
            if is_dataclass(it):
                norm.append(asdict(it))
            elif isinstance(it, dict):
                norm.append(it)
            else:
                norm.append({"message": str(it)})
        out[key] = norm

    return out


def _emit_json(obj: Dict[str, Any], *, pretty: bool = False, out_path: Optional[str] = None):
    s = json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)
    if out_path:
        # JSONL append
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(s.replace("\n", " ") if not pretty else s)
            f.write("\n")
    else:
        console.print(s)

# -----------------------
# Core actions
# -----------------------
def _compile_policy(policy_path: str, *, skip_verify: bool, skip_validate: bool, debug_z3: bool = False):
    """
    Compile pipeline:
      parse -> (optionally adjust config) -> compiler.compile (validator+Z3+IR)
    NOTE: In this repo CSLCompiler.compile already performs validation + Z3 verification
          depending on constitution.config flags. CLI should not double-run them.
    """
    compiler = CSLCompiler()

    # Parse to AST
    try:
        constitution = parse_csl_file(policy_path)
    except Exception as e:
        raise CompilationError(f"Failed to parse policy: {_safe_to_str(e)}") from e

    # Respect CLI flags (best-effort, without fighting the compiler design)
    # Validation skip: current compiler always validates; keep flag for future, but don't break.
    if skip_validate:
        console.print("[yellow]⚠️  --skip-validate requested, but compiler currently always validates in v0.1.[/yellow]")

    # Verification skip: disable Z3 check in config
    if skip_verify:
        if constitution.config is None:
            # If config missing, create minimal config-like object is risky; instead warn.
            console.print("[yellow]⚠️  --skip-verify requested, but policy has no CONFIG block; cannot disable safely.[/yellow]")
        else:
            constitution.config.check_logical_consistency = False

    # Compile (this will run validator + (optional) Z3 + IR generation)
    compiled = compiler.compile(constitution)
    return compiled


def cmd_verify(args: argparse.Namespace) -> int:
    _print_banner()
    policy = args.policy

    console.print(Panel(f"[bold]Verifying policy[/bold]\n[cyan]{policy}[/cyan]", border_style="cyan", width=92))
    console.print()

    try:
        compiled = _compile_policy(
            policy,
            skip_verify=args.skip_verify,
            skip_validate=args.skip_validate,
            debug_z3=bool(getattr(args, "debug_z3", False)),
        )
    except CompilationError as e:
        # Optional: deep Z3 debug trace (fail path only)
        if bool(getattr(args, "debug_z3", False)):
            _run_z3_debug_trace(policy_path=policy)
        _print_error("Verification failed", _safe_to_str(e, 2000))
        return 2
    except Exception as e:
        _print_error("Unexpected error", _safe_to_str(e, 2000))
        return 3

    # Success summary
    rows = []
    for attr, label in [
        ("domain_name", "Domain"),
        ("policy_name", "Policy"),
        ("policy_id", "Policy ID"),
        ("policy_version", "Policy Version"),
        ("policy_hash", "Policy Hash"),
        ("engine_version", "Engine Version"),
    ]:
        v = getattr(compiled, attr, None)
        if v is not None:
            rows.append([label, _safe_to_str(v)])
    if rows:
        _print_kv_table("Compiled Policy Metadata", rows)

    _print_success("Verification passed", "No logical contradictions found (given CSL-Core analysis scope).")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    _print_banner()
    policy = args.policy

    # Load inputs
    inputs: List[Dict[str, Any]] = []
    if args.input_json:
        inputs.append(_load_json_from_arg(args.input_json))

    if args.input_file:
        payload = _load_json_from_file(args.input_file)
        if isinstance(payload, dict):
            inputs.append(payload)
        elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            inputs.extend(payload)
        else:
            raise ValueError("Input file must contain a JSON object (dict) or a list of objects.")

    if not inputs:
        # Default minimal input
        inputs = [{"action": "TRANSFER", "amount": 5000}]

    # Compile policy
    try:
        compiled = _compile_policy(
            policy,
            skip_verify=args.skip_verify,
            skip_validate=args.skip_validate,
            debug_z3=bool(getattr(args, "debug_z3", False)),
        )
    except CompilationError as e:
        if bool(getattr(args, "debug_z3", False)):
            _run_z3_debug_trace(policy_path=policy)
        _print_error("Compile/verify failed", _safe_to_str(e, 2000))
        return 2

    guard = ChimeraGuard(
        compiled,
        config=RuntimeConfig(
            dry_run=bool(args.dry_run),
            missing_key_behavior=args.missing_key_behavior,
            evaluation_error_behavior=args.evaluation_error_behavior,
            collect_all_violations=not bool(args.fast_fail),
            raise_on_block=not bool(args.no_raise),
        ),
    )

    viz = RuntimeVisualizer() if args.dashboard else None

    # Run simulations
    blocked = 0
    allowed = 0

    for idx, ctx in enumerate(inputs, 1):
        if not bool(getattr(args, "quiet", False)):
            console.print(Panel(f"[bold]Simulation #{idx}[/bold]", border_style="magenta", width=92))
    
        try:
            result = guard.verify(ctx)
    
            # Normalize once
            is_allowed = bool(getattr(result, "allowed", False))
            violations = list(getattr(result, "violations", []) or [])
            warnings = list(getattr(result, "warnings", []) or [])
    
            # Machine output
            payload = _result_to_json(result, context=ctx, compiled=compiled)
            if bool(getattr(args, "json", False)) or bool(getattr(args, "json_out", None)):
                _emit_json(
                    payload,
                    pretty=bool(getattr(args, "pretty_json", False)),
                    out_path=getattr(args, "json_out", None),
                )
    
            # Optional rich dashboard (only when not quiet)
            if viz and not bool(getattr(args, "quiet", False)):
                viz.visualize(result=result, context=ctx, title="Chimera Gatekeeper")
    
            # Human output (single source of truth)
            if not bool(getattr(args, "quiet", False)):
                if is_allowed:
                    _print_success("ALLOWED", "No violations found. Action is permitted by the policy.")
                else:
                    _print_error("BLOCKED", "One or more policy constraints were violated.")
    
                if violations:
                    vt = Table(title="Violations", box=box.SIMPLE, show_header=True, header_style="bold red", width=92)
                    vt.add_column("#", style="dim", width=6)
                    vt.add_column("Message", style="white")
                    for i, v in enumerate(violations, 1):
                        vt.add_row(str(i), _safe_to_str(v, 800))
                    console.print(vt)
    
                if warnings:
                    wt = Table(title="Warnings", box=box.SIMPLE, show_header=True, header_style="bold yellow", width=92)
                    wt.add_column("#", style="dim", width=6)
                    wt.add_column("Message", style="white")
                    for i, w in enumerate(warnings, 1):
                        wt.add_row(str(i), _safe_to_str(w, 800))
                    console.print(wt)
    
            if is_allowed:
                allowed += 1
            else:
                blocked += 1
    
        except ChimeraError as e:
            blocked += 1
    
            payload = {
                "allowed": False,
                "domain": getattr(compiled, "domain_name", None),
                "policy_hash": getattr(compiled, "policy_hash", None),
                "policy_version": getattr(compiled, "policy_version", None),
                "engine_version": getattr(compiled, "engine_version", None),
                "context": ctx,
                "error": str(e),
                "violations": [],
                "warnings": [],
            }
    
            if bool(getattr(args, "json", False)) or bool(getattr(args, "json_out", None)):
                _emit_json(
                    payload,
                    pretty=bool(getattr(args, "pretty_json", False)),
                    out_path=getattr(args, "json_out", None),
                )
    
            if not bool(getattr(args, "quiet", False)):
                _print_error("BLOCKED (exception)", _safe_to_str(e, 2000))
    
        if not bool(getattr(args, "quiet", False)):
            console.print()

    if not bool(getattr(args, "quiet", False)):
        _print_kv_table(
            "Simulation Summary",
            [
                ["Total", str(len(inputs))],
                ["Allowed", str(allowed)],
                ["Blocked", str(blocked)],
                ["Dry-run", str(bool(args.dry_run))],
            ],
        )
    return 0 if blocked == 0 or args.dry_run else 10


def cmd_repl(args: argparse.Namespace) -> int:
    _print_banner()
    policy = args.policy

    try:
        compiled = _compile_policy(
            policy,
            skip_verify=args.skip_verify,
            skip_validate=args.skip_validate,
            debug_z3=bool(getattr(args, "debug_z3", False)),
        )
    except CompilationError as e:
        if bool(getattr(args, "debug_z3", False)):
            _run_z3_debug_trace(policy_path=policy)
        _print_error("Compile/verify failed", _safe_to_str(e, 2000))
        return 2

    guard = ChimeraGuard(
        compiled,
        config=RuntimeConfig(
            dry_run=bool(args.dry_run),
            missing_key_behavior=args.missing_key_behavior,
            evaluation_error_behavior=args.evaluation_error_behavior,
            collect_all_violations=True,
            raise_on_block=False,  # REPL: never throw, always return a result view
        ),
    )
    viz = RuntimeVisualizer() if args.dashboard else None

    console.print(
        Panel(
            "[bold]REPL mode[/bold]\n"
            "Paste JSON objects (one per line). Empty line exits.\n\n"
            "Example:\n"
            "  {\"action\": \"TRANSFER\", \"amount\": 5000, \"country\": \"TR\"}",
            border_style="cyan",
            width=92,
        )
    )

    while True:
        try:
            line = input("cslcore> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        if not line:
            break

        try:
            ctx = _load_json_from_arg(line)
        except Exception as e:
            _print_error("Invalid input", _safe_to_str(e, 2000))
            continue

        try:
            result = guard.verify(ctx)
        except ChimeraError as e:
            # In REPL we set raise_on_block False, but keep safety
            _print_error("Runtime error", _safe_to_str(e, 2000))
            continue

        if viz:
            viz.visualize(result=result, context=ctx, title="Chimera Gatekeeper")
        else:
            console.print(f"allowed={result.allowed} violations={len(result.violations)} warnings={len(result.warnings)}")

    _print_success("Bye", "REPL exited cleanly.")
    return 0


# -----------------------
# CLI wiring
# -----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cslcore",
        description="CSL-Core CLI — deterministic safety for AI policies",
    )
    p.add_argument("--version", action="version", version=f"csl-core {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    # verify
    v = sub.add_parser("verify", help="Parse + validate + Z3 verify + compile a CSL policy")
    v.add_argument("policy", help="Path to .csl policy file")
    v.add_argument("--skip-verify", action="store_true", help="Skip Z3 verification (not recommended)")
    v.add_argument("--skip-validate", action="store_true", help="Skip semantic validation (not recommended)")
    v.add_argument("--debug-z3", action="store_true", help="On failure, print Z3 encoding trace (tail) to debug sort mismatches.")
    v.set_defaults(func=cmd_verify)

    # simulate
    s = sub.add_parser("simulate", help="Run runtime guard against one or more JSON inputs")
    s.add_argument("policy", help="Path to .csl policy file")
    s.add_argument("--input", dest="input_json", help="Single JSON object as a string")
    s.add_argument("--input-file", help="JSON file containing an object or a list of objects")
    s.add_argument("--dashboard", action="store_true", help="Show rich audit dashboard per simulation")
    s.add_argument("--dry-run", action="store_true", help="Do not block; report what WOULD be blocked")
    s.add_argument("--fast-fail", action="store_true", help="Stop at first violation (faster)")
    s.add_argument("--no-raise", action="store_true", help="Do not raise on BLOCK (still returns a result)")
    s.add_argument("--missing-key-behavior", choices=["block", "warn", "ignore"], default="block",
                   help="What to do if a rule references a missing key (default: block)")
    s.add_argument("--evaluation-error-behavior", choices=["block", "warn", "ignore"], default="block",
                   help="What to do on evaluation errors (type mismatch, etc.) (default: block)")
    s.add_argument("--skip-verify", action="store_true", help="Skip Z3 verification (not recommended)")
    s.add_argument("--skip-validate", action="store_true", help="Skip semantic validation (not recommended)")
    s.add_argument("--debug-z3", action="store_true", help="On compile failure, print Z3 encoding trace (tail).")
    s.add_argument("--json", action="store_true", help="Print machine-readable JSON output per simulation.")
    s.add_argument("--pretty-json", action="store_true", help="Pretty-print JSON (with indentation).")
    s.add_argument("--json-out", help="Append JSONL output to a file (one line per simulation).")
    s.add_argument("--quiet", action="store_true", help="Reduce human output (useful with --json).")
    s.set_defaults(func=cmd_simulate)

    # repl
    r = sub.add_parser("repl", help="Interactive REPL for rapid policy testing (JSON lines)")
    r.add_argument("policy", help="Path to .csl policy file")
    r.add_argument("--dashboard", action="store_true", help="Show rich audit dashboard per input")
    r.add_argument("--dry-run", action="store_true", help="Do not block; report what WOULD be blocked")
    r.add_argument("--missing-key-behavior", choices=["block", "warn", "ignore"], default="block")
    r.add_argument("--evaluation-error-behavior", choices=["block", "warn", "ignore"], default="block")
    r.add_argument("--skip-verify", action="store_true")
    r.add_argument("--skip-validate", action="store_true")
    r.add_argument("--debug-z3", action="store_true", help="On compile failure, print Z3 encoding trace (tail).")
    r.set_defaults(func=cmd_repl)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        rc = args.func(args)
        # Fail-safe: some handlers may forget to return an int
        if rc is None:
            return 0
        return int(rc)
    except KeyboardInterrupt:
        console.print()
        _print_error("Interrupted", "Operation cancelled by user.")
        return 130
    except Exception as e:
        _print_error("Fatal error", _safe_to_str(e, 2000))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
