"""
🚀 CSL-Core LangChain Integration Template
==========================================
A "Copy-Paste" ready template to secure your LangChain Agent with ChimeraGuard.

USAGE:
1. Copy the "CSL INTEGRATION" section into your agent's python file.
2. Replace 'raw_tools' with your actual LangChain tools.
3. Point 'load_guard' to your .csl policy file.
"""

import os
import sys
from pathlib import Path

# --- [PLACEHOLDER] LANGCHAIN IMPORTS ---
# (Uncomment these lines in your actual project)
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_openai_tools_agent, AgentExecutor
# from langchain_core.tools import tool

# ==============================================================================
# 🛡️ CSL-CORE INTEGRATION (COPY THIS BLOCK)
# ==============================================================================
from chimera_core import load_guard
from chimera_core.plugins.langchain import guard_tools

def get_secure_tools(raw_tools_list):
    """
    Wraps standard LangChain tools with ChimeraGuard enforcement.
    """
    
    # 1. LOAD POLICY
    # Uses pathlib to find the policy file relative to THIS script location.
    current_dir = Path(__file__).resolve().parent
    # TODO: Change filename if using a different policy
    policy_path = str(current_dir / "01_hello_world.csl")
    
    try:
        guard = load_guard(policy_path)
        print(f"✅ Policy loaded: {policy_path}")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        sys.exit(1)

    # 2. WRAP TOOLS
    safe_tools = guard_tools(
        tools=raw_tools_list,
        guard=guard,
        
        # INJECT: Static context the LLM doesn't know about but the policy needs.
        inject={
            "user_role": os.getenv("USER_ROLE", "USER"),
            "environment": os.getenv("ENV", "DEV"), 
        },
        
        # DASHBOARD: Real-time decision table.
        enable_dashboard=os.getenv("CHIMERA_DEBUG", "1") == "1",

        # TOOL FIELD: (Optional)
        # If your CSL policy has a variable like: `VARIABLES { tool: {"SEARCH", "CALC"} }`
        # then uncomment the line below. It puts the tool's name into the 'tool' variable.
        # tool_field="tool" 
    )
    
    return safe_tools
# ==============================================================================


def main():
    """
    Example of how to initialize the agent with secure tools.
    """
    print("🤖 Initializing Agent...")

    # --- [STEP 1] DEFINE YOUR RAW TOOLS ---
    # raw_tools = [my_search_tool, my_calculator_tool]
    raw_tools = [] # <--- TODO: INSERT YOUR TOOLS HERE

    # --- [STEP 2] SECURE THEM ---
    safe_tools = get_secure_tools(raw_tools)

    if not safe_tools:
        print("⚠️ No tools provided. Add tools to 'raw_tools' list.")
        return

    # --- [STEP 3] CREATE AGENT ---
    # agent = create_openai_tools_agent(llm, safe_tools, prompt)
    # executor = AgentExecutor(agent=agent, tools=safe_tools)
    
    print("🚀 Agent is ready and guarded! (Uncomment LLM code to run)")

if __name__ == "__main__":
    main()