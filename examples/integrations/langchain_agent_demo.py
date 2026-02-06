"""
ChimeraGuard x LangChain: Enterprise Agent Integration Demo
===========================================================

Target Audience: LangChain Developers & AI Engineers.
Goal: Demonstrate "Policy-as-Code" enforcement in a runtime agent loop.

Visuals: Uses 'rich' library to mimic an Enterprise Security Dashboard.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

# --- 1. Environment & Dependency Setup ---
current_file = Path(__file__).resolve()
# Adjust path to find 'chimera_core' (examples/integrations/ -> root)
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Rich Library Check (Visuals)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.status import Status
except ImportError:
    print("❌ Error: 'rich' library is required. Install: pip install rich")
    sys.exit(1)

# LangChain Check
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError:
    print("❌ Error: 'langchain-core' is required. Install: pip install langchain-core")
    sys.exit(1)

# Chimera Core Imports
from chimera_core.language.parser import parse_csl_file
from chimera_core.language.compiler import CSLCompiler
from chimera_core.runtime import ChimeraGuard, ChimeraError
from chimera_core.plugins.langchain import guard_tools

# Initialize Console
console = Console()

# ==============================================================================
# 2. DEFINE BUSINESS TOOLS (Simulated)
# ==============================================================================

class TransferInput(BaseModel):
    amount: int = Field(description="Amount to transfer")
    approval_token: str = Field(default="NO", description="Approval token (YES/NO)")

class TransferFundsTool(BaseTool):
    name: str = "TRANSFER_FUNDS"
    description: str = "Transfers funds to an account."
    args_schema: type[BaseModel] = TransferInput

    def _run(self, amount: int, approval_token: str = "NO") -> str:
        return f"💸 SUCCESS: Transferred ${amount}"

class EmailInput(BaseModel):
    recipient_domain: str = Field(description="Domain (INTERNAL/EXTERNAL)")
    pii_present: str = Field(description="Contains PII? (YES/NO)")

class SendEmailTool(BaseTool):
    name: str = "SEND_EMAIL"
    description: str = "Sends an email."
    args_schema: type[BaseModel] = EmailInput

    def _run(self, recipient_domain: str, pii_present: str) -> str:
        return f"📧 SENT: Email to {recipient_domain}"

class DBInput(BaseModel):
    table_name: str = Field(description="DB Table")

class QueryDBTool(BaseTool):
    name: str = "QUERY_DB"
    description: str = "Queries DB."
    args_schema: type[BaseModel] = DBInput

    def _run(self, table_name: str) -> str:
        return f"🔍 QUERY: Retrieved {table_name}"

# ==============================================================================
# 3. HELPER FUNCTIONS (Visuals & Logic)
# ==============================================================================

def load_security_guard() -> ChimeraGuard:
    """Compiles policy with visual feedback."""
    # Look for policy in 'examples/' folder
    policy_path = project_root / "examples" / "agent_tool_guard.csl"
    
    if not policy_path.exists():
        console.print(f"[bold red]❌ Policy not found at: {policy_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]📄 Loading Policy: {policy_path.name}[/dim]")
    
    # Visual simulation of the compilation pipeline
    with console.status("[bold green]Compiling Domain: AgentToolGuard...[/bold green]", spinner="dots"):
        time.sleep(0.8) # Slight pause for dramatic effect
        constitution = parse_csl_file(str(policy_path))
        compiled = CSLCompiler().compile(constitution)
        
    console.print("   [green]✔ Validating Syntax... OK[/green]")
    console.print("   [green]✔ Verifying Logic Model (Z3)... OK[/green]")
    console.print("   [green]✔ Generating IR... OK[/green]")
    console.print()
    return ChimeraGuard(compiled)

def run_scenario(
    title: str,
    tool: BaseTool,
    input_args: Dict[str, Any],
    expected_outcome: str, # "ALLOW" or "BLOCK"
    description: str
) -> Tuple[str, str, str, str]:
    """
    Executes a single scenario and returns data for the summary table.
    Returns: (Test Case Name, Expected, Result, Details)
    """
    console.print(f"[bold white]🔹 {title}[/bold white]")
    console.print(f"   [dim]{description}[/dim]")
    console.print(f"   [cyan]Attempting:[/cyan] {tool.name}({input_args})")

    outcome = "UNKNOWN"
    details = ""

    try:
        # EXECUTE TOOL
        # The guard logic happens INSIDE the wrapper here
        result_msg = tool.invoke(input_args)
        
        # If we are here, it was ALLOWED
        outcome = "ALLOW"
        details = result_msg
        console.print(f"   [green]✅ ACTION ALLOWED:[/green] {result_msg}")

    except ChimeraError as e:
        # If we are here, it was BLOCKED
        outcome = "BLOCK"
        # Extract the constraint name for cleaner output
        msg = str(e)
        if "Violation '" in msg:
            constraint = msg.split("'")[1]
            details = f"Constraint: {constraint}"
        else:
            details = msg
            
        console.print(f"   [red]🛡️  BLOCKED by Guard:[/red] {details}")

    except Exception as e:
        outcome = "ERROR"
        details = str(e)
        console.print(f"   [bold red]❌ SYSTEM ERROR:[/bold red] {e}")

    console.print() # Newline separator
    return (title, expected_outcome, outcome, details)

# ==============================================================================
# 4. MAIN EXECUTION FLOW
# ==============================================================================

def main():
    # --- A. Header ---
    console.print()
    console.print(Panel(
        "[bold cyan]🚀 ChimeraGuard x LangChain Integration Demo[/bold cyan]\n\n"
        "Simulating an Enterprise Financial Agent with Policy Enforcement.\n"
        "Demonstrates: Tool Wrapping, Context Injection, and Deterministic Blocking.",
        border_style="cyan",
        width=100
    ))
    console.print()

    # --- B. Initialization ---
    guard = load_security_guard()
    
    # Mapper Setup: Bridges LangChain inputs (kwargs) to CSL Variables
    def agent_context_mapper(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "amount": tool_input.get("amount", 0),
            "approval_token": tool_input.get("approval_token", "NO"),
            "recipient_domain": tool_input.get("recipient_domain"),
            "pii_present": tool_input.get("pii_present"),
            "db_table": tool_input.get("table_name"), # Mapping 'table_name' -> 'db_table'
        }
    
    raw_tools = [TransferFundsTool(), SendEmailTool(), QueryDBTool()]

    # --- C. Scenario Setup ---
    results_data: List[Tuple[str, str, str, str]] = []

    # 1. Setup USER Tools
    # We create a version of tools specifically for a "USER" session.
    # The 'inject' parameter ensures the LLM cannot override the user_role.
    user_tools = guard_tools(
        raw_tools, guard, context_mapper=agent_context_mapper,
        inject={"user_role": "USER"}, tool_field="tool"
    )

    # 2. Setup ADMIN Tools
    # We create a version of tools specifically for an "ADMIN" session.
    admin_tools = guard_tools(
        raw_tools, guard, context_mapper=agent_context_mapper,
        inject={"user_role": "ADMIN"}, tool_field="tool"
    )

    # --- D. Run Scenarios ---
    
    # Scenario 1: DLP
    r1 = run_scenario(
        "SCENARIO 1: DLP / Prompt Injection",
        user_tools[1], # SendEmail
        {"recipient_domain": "EXTERNAL", "pii_present": "YES"},
        "BLOCK",
        "Context: User tries to email PII to an EXTERNAL domain."
    )
    results_data.append(r1)
    time.sleep(0.5)

    # Scenario 2: RBAC
    r2 = run_scenario(
        "SCENARIO 2: RBAC Enforcement",
        user_tools[0], # Transfer
        {"amount": 100},
        "BLOCK",
        "Context: A standard USER tries to transfer funds."
    )
    results_data.append(r2)
    time.sleep(0.5)

    # Scenario 3: Business Logic
    r3 = run_scenario(
        "SCENARIO 3: Business Logic Limits",
        admin_tools[0], # Transfer (as Admin)
        {"amount": 6000, "approval_token": "YES"}, # Limit is 5000
        "BLOCK",
        "Context: ADMIN tries to transfer $6000 (Limit is $5000)."
    )
    results_data.append(r3)
    time.sleep(0.5)

    # Scenario 4: Happy Path
    r4 = run_scenario(
        "SCENARIO 4: Happy Path",
        admin_tools[2], # QueryDB
        {"table_name": "CUSTOMERS"},
        "ALLOW",
        "Context: ADMIN queries the CUSTOMERS table."
    )
    results_data.append(r4)

    # --- E. Summary Table (Matching run_examples.py style) ---
    
    table = Table(
        title="Integration Test Results",
        box=box.ROUNDED,
        header_style="bold cyan",
        width=100
    )
    table.add_column("Scenario", style="white", width=35)
    table.add_column("Expected", style="cyan", justify="center", width=10)
    table.add_column("Result", style="white", justify="center", width=10)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Details", style="dim")

    passed_count = 0
    
    for name, expected, actual, details in results_data:
        # Determine Pass/Fail (Pass means Expected == Actual)
        if expected == actual:
            status = "[bold green]✅ PASS[/bold green]"
            res_color = "[green]" if actual == "ALLOW" else "[red]"
            passed_count += 1
        else:
            status = "[bold red]❌ FAIL[/bold red]"
            res_color = "[yellow]"

        # Truncate details if too long
        display_details = details if len(details) < 30 else details[:27]+"..."

        table.add_row(
            name,
            expected,
            f"{res_color}{actual}[/]",
            status,
            display_details
        )

    console.print()
    console.print(table)
    console.print()
    
    if passed_count == len(results_data):
        console.print("[bold green]🎉 Integration Demo Completed Successfully![/bold green]")
    else:
        console.print("[bold red]⚠️  Some scenarios did not match expectations.[/bold red]")
    console.print()

if __name__ == "__main__":
    main()