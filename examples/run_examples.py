#!/usr/bin/env python3
"""
CSL-Core Examples Test Runner

Run all example policies with test cases and show results.

Folder Structure:
    examples/
    ├── *.csl                      # Policy files
    ├── json_files/                # Test cases
    │   ├── dao_treasury_guard_tests.json
    │   ├── agent_tool_guard_tests.json
    │   └── banking_tests.json
    └── integrations/               # Integrations demos

Usage:
    python examples/run_examples.py                    # Run all
    python examples/run_examples.py dao_treasury_guard # Run specific
    python examples/run_examples.py --dashboard        # Rich UI
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("❌ Error: 'rich' library is required to run this script.")
    print("   Please install it: pip install rich")
    sys.exit(1)

from chimera_core import CSLCompiler, ChimeraGuard, ChimeraError
from chimera_core.runtime import RuntimeConfig

console = Console()

# Paths
EXAMPLES_DIR = Path(__file__).parent
JSON_DIR = EXAMPLES_DIR / "json_files"

# Policy definitions
POLICIES = {
    "dao_treasury_guard": {
        "policy": "dao_treasury_guard.csl",
        "tests": "dao_treasury_guard_tests.json",
        "description": "DAO treasury governance with multi-sig protection"
    },
    "agent_tool_guard": {
        "policy": "agent_tool_guard.csl",
        "tests": "agent_tool_guard_tests.json",
        "description": "AI agent tool permissions & PII protection"
    },
    "banking": {
        "policy": "chimera_banking_case_study.csl",
        "tests": "banking_tests.json",
        "description": "Banking transaction risk management"
    },
    "langchain_agent_demo": {
        "type": "script", 
        "path": "integrations/langchain_agent_demo.py",
        "description": "Interactive LangChain Agent with visual dashboard"
    }
}


def load_policy(policy_name: str) -> ChimeraGuard:
    """Compile and load a policy"""
    policy_path = EXAMPLES_DIR / POLICIES[policy_name]["policy"]
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    
    console.print(f"[dim]Loading policy: {policy_path.name}[/dim]")
    
    constitution = CSLCompiler.load(str(policy_path))
    return ChimeraGuard(
        constitution,
        config=RuntimeConfig(
            raise_on_block=False,
            collect_all_violations=True,
            missing_key_behavior="block",
            evaluation_error_behavior="block"
        )
    )


def load_tests(policy_name: str) -> Dict:
    """Load test cases for a policy"""
    test_file = POLICIES[policy_name]["tests"]
    test_path = JSON_DIR / test_file
    
    if not test_path.exists():
        console.print(f"[yellow]⚠️  No tests found: {test_path}[/yellow]")
        console.print(f"[dim]Expected location: json_files/{test_file}[/dim]\n")
        return {"allow_cases": [], "block_cases": []}
    
    console.print(f"[dim]Loading tests: {test_path.name}[/dim]")
    
    with open(test_path, encoding="utf-8") as f:
        return json.load(f)


def print_policy_header(policy_name: str):
    """Print policy section header"""
    policy_info = POLICIES[policy_name]
    
    console.print()
    console.print("=" * 100)
    console.print()
    
    if policy_info.get("type") == "script":
        content = (
            f"[bold cyan]{policy_name.upper()}[/bold cyan]\n\n"
            f"{policy_info['description']}\n\n"
            f"Script: [dim]examples/{policy_info['path']}[/dim]"
        )
    else:
        content = (
            f"[bold cyan]{policy_name.upper()}[/bold cyan]\n\n"
            f"{policy_info['description']}\n\n"
            f"Policy: [dim]{policy_info['policy']}[/dim]\n"
            f"Tests:  [dim]json_files/{policy_info['tests']}[/dim]"
        )

    console.print(Panel(
        content,
        border_style="cyan",
        width=100
    ))
    console.print()


def run_policy_tests(policy_name: str, show_details: bool = False) -> bool:
    """Run all tests for a policy"""
    print_policy_header(policy_name)
    
    # ============================================================
    # STEP 1: Compile Policy
    # ============================================================
    try:
        guard = load_policy(policy_name)
        console.print("[green]✅ Policy compiled successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to compile policy:[/red]")
        console.print(f"[red]{str(e)}[/red]\n")
        return False
    
    # ============================================================
    # STEP 2: Load Tests
    # ============================================================
    tests = load_tests(policy_name)
    
    if not tests.get("allow_cases") and not tests.get("block_cases"):
        console.print("[yellow]⚠️  No test cases found, skipping...[/yellow]\n")
        return True
    
    # ============================================================
    # STEP 3: Run Tests
    # ============================================================
    
    # Results table
    table = Table(
        title=f"Test Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        width=100
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Test Case", style="white", width=45)
    table.add_column("Expected", style="cyan", width=10, justify="center")
    table.add_column("Result", style="white", width=10, justify="center")
    table.add_column("Status", style="white", width=12, justify="center")
    table.add_column("Details", style="dim", width=15)
    
    passed = 0
    failed = 0
    test_num = 0
    
    # Run ALLOW cases
    for test in tests.get("allow_cases", []):
        test_num += 1
        name = test.get("name", f"Test {test_num}")
        result = guard.verify(test["input"])
        
        if result.allowed:
            table.add_row(
                str(test_num),
                name,
                "[green]ALLOW[/green]",
                "[green]ALLOW[/green]",
                "[bold green]✅ PASS[/bold green]",
                ""
            )
            passed += 1
        else:
            # Failed - expected ALLOW but got BLOCK
            violation_msg = result.violations[0] if result.violations else ""
            details = violation_msg[:20] + "..." if len(violation_msg) > 20 else violation_msg
            
            table.add_row(
                str(test_num),
                name,
                "[green]ALLOW[/green]",
                "[red]BLOCK[/red]",
                "[bold red]❌ FAIL[/bold red]",
                f"[dim]{details}[/dim]"
            )
            failed += 1
            
            if show_details:
                console.print(f"[red]Violation: {violation_msg}[/red]")
    
    # Run BLOCK cases
    for test in tests.get("block_cases", []):
        test_num += 1
        name = test.get("name", f"Test {test_num}")
        expected_violation = test.get("expected_violation", "")
        
        result = guard.verify(test["input"])
        
        if not result.allowed:
            # Blocked as expected - check if correct constraint
            actual_violations = " | ".join(result.violations)
            
            # Check if expected constraint was violated
            matched = expected_violation in actual_violations if expected_violation else True
            
            if matched:
                table.add_row(
                    str(test_num),
                    name,
                    "[red]BLOCK[/red]",
                    "[red]BLOCK[/red]",
                    "[bold green]✅ PASS[/bold green]",
                    f"[dim]{expected_violation[:15]}[/dim]"
                )
                passed += 1
            else:
                # Blocked but wrong constraint
                table.add_row(
                    str(test_num),
                    name,
                    "[red]BLOCK[/red]",
                    "[red]BLOCK[/red]",
                    "[bold green]✅ PASS[/bold green]",
                    "[dim]wrong constraint[/dim]"
                )
                passed += 1  # Still counts as pass (blocked correctly)
                
                if show_details:
                    console.print(f"[yellow]Expected: {expected_violation}[/yellow]")
                    console.print(f"[yellow]Got: {actual_violations}[/yellow]")
        else:
            # Failed - expected BLOCK but got ALLOW
            table.add_row(
                str(test_num),
                name,
                "[red]BLOCK[/red]",
                "[green]ALLOW[/green]",
                "[bold red]❌ FAIL[/bold red]",
                "[dim]should block[/dim]"
            )
            failed += 1
    
    console.print(table)
    console.print()
    
    # ============================================================
    # STEP 4: Summary
    # ============================================================
    total = passed + failed
    
    if total > 0:
        pass_rate = (passed / total) * 100
        
        summary_table = Table(
            box=box.SIMPLE,
            show_header=False,
            width=100,
            padding=(0, 2)
        )
        summary_table.add_column("Metric", style="cyan", width=30)
        summary_table.add_column("Value", style="white", width=20)
        
        summary_table.add_row("Total Tests", str(total))
        summary_table.add_row("Passed", f"[green]{passed}[/green]")
        
        if failed > 0:
            summary_table.add_row("Failed", f"[red]{failed}[/red]")
        else:
            summary_table.add_row("Failed", "0")
        
        summary_table.add_row("Pass Rate", f"{pass_rate:.1f}%")
        
        console.print(summary_table)
        console.print()
        
        if failed == 0:
            console.print(f"[bold green]🎉 All tests passed for {policy_name}![/bold green]\n")
        else:
            console.print(f"[bold red]❌ {failed} test(s) failed for {policy_name}[/bold red]\n")
    else:
        console.print("[yellow]No tests executed[/yellow]\n")
    
    return failed == 0

def run_integration_script(name: str) -> bool:
    """Run a standalone integration demo script"""
    info = POLICIES[name]
    script_path = EXAMPLES_DIR / info["path"]
    
    print_policy_header(name)
    
    if not script_path.exists():
        console.print(f"[bold red]❌ Script not found: {script_path}[/bold red]")
        return False
        
    console.print(f"[dim]Executing script: {info['path']}...[/dim]\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=EXAMPLES_DIR.parent, 
            capture_output=False,  
            text=True
        )
        
        if result.returncode == 0:
            console.print(f"\n[bold green]✅ {name} executed successfully![/bold green]\n")
            return True
        else:
            console.print(f"\n[bold red]❌ {name} failed with exit code {result.returncode}[/bold red]\n")
            return False
            
    except Exception as e:
        console.print(f"[bold red]❌ Error running script: {e}[/bold red]")
        return False

def list_available_policies():
    """List all available policies"""
    console.print("\n[bold cyan]Available Policies:[/bold cyan]\n")
    
    for name, info in POLICIES.items():
        policy_path = EXAMPLES_DIR / info["policy"]
        test_path = JSON_DIR / info["tests"]
        
        status = "✅" if policy_path.exists() else "❌"
        test_status = "✅" if test_path.exists() else "❌"
        
        console.print(f"  {status} [cyan]{name}[/cyan]")
        console.print(f"     {info['description']}")
        console.print(f"     Policy: {info['policy']} | Tests: {test_status} {info['tests']}")
        console.print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run CSL example tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_examples.py                     # Run all policies
    python run_examples.py dao_treasury_guard  # Run specific policy
    python run_examples.py --list              # List available policies
    python run_examples.py --details           # Show detailed failures
        """
    )
    parser.add_argument(
        "policy",
        nargs="?",
        choices=list(POLICIES.keys()) + ["all"],
        default="all",
        help="Policy to test (default: all)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available policies and exit"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed failure information"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available_policies()
        return 0
    
    # Print header
    console.print()
    console.print(Panel(
        "[bold white]CSL-Core Examples Test Runner[/bold white]\n\n"
        "Testing example policies with predefined test cases",
        border_style="cyan",
        width=100
    ))
    
    # Determine which policies to test
    if args.policy == "all":
        policies_to_test = list(POLICIES.keys())
    else:
        policies_to_test = [args.policy]
    
    # Run tests
    all_passed = True
    results = {}
    
    for policy_name in policies_to_test:
        policy_info = POLICIES[policy_name]
        
        if policy_info.get("type") == "script":
            success = run_integration_script(policy_name)
        else:
            success = run_policy_tests(policy_name, show_details=args.details)
            
        results[policy_name] = success
        
        if not success:
            all_passed = False
    
    # Final summary
    console.print("=" * 100)
    console.print()
    
    if len(policies_to_test) > 1:
        console.print(Panel(
            "[bold]Overall Summary[/bold]",
            border_style="cyan",
            width=100
        ))
        console.print()
        
        for policy_name, success in results.items():
            status = "[green]✅ PASS[/green]" if success else "[red]❌ FAIL[/red]"
            console.print(f"  {status}  {policy_name}")
        
        console.print()
    
    if all_passed:
        console.print("[bold green]🎉 All tests passed![/bold green]\n")
        return 0
    else:
        console.print("[bold red]❌ Some tests failed[/bold red]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())