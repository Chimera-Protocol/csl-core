"""
CSL-Core — LangChain Integration Plugin

Uses the `ChimeraPlugin` architecture from base.py to provide safe,
visualized, and policy-agnostic guarding for LangChain components.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Iterable

# LangChain Dependency Check
try:
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool
except ImportError:
    raise ImportError("langchain-core required. Install with: pip install langchain-core")

from ..runtime import ChimeraGuard
from .base import ChimeraPlugin, ContextMapper

# ==============================================================================
# 1. LCEL Runnable Gate
# ==============================================================================

class ChimeraRunnableGate(Runnable, ChimeraPlugin):
    """
    Acts as a Policy Gate in an LCEL chain.
    Inherits Logic from ChimeraPlugin, Interface from Runnable.
    """
    
    def __init__(
        self, 
        guard_or_constitution: Any, 
        *,
        context_mapper: Optional[ContextMapper] = None,
        inject: Optional[Dict[str, Any]] = None,
        enable_dashboard: bool = False
    ):
        # Unwrap ChimeraGuard if passed directly
        if isinstance(guard_or_constitution, ChimeraGuard):
            constitution = guard_or_constitution.constitution
        else:
            constitution = guard_or_constitution

        # Initialize Base Plugin logic
        ChimeraPlugin.__init__(
            self, 
            constitution=constitution, 
            enable_dashboard=enable_dashboard,
            context_mapper=context_mapper,
            title="LangChain::Gate"
        )
        self.inject = inject or {}

    def process(self, input_data: Any) -> Any:
        return self.invoke(input_data)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        # normalize -> verify -> visualize
        self.run_guard(input, extra_context=self.inject)
        # pass-through
        return input


def gate(
    guard: Any,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    enable_dashboard: bool = False
) -> ChimeraRunnableGate:
    """Helper to create a ChimeraRunnableGate."""
    return ChimeraRunnableGate(
        guard, 
        context_mapper=context_mapper, 
        inject=inject, 
        enable_dashboard=enable_dashboard
    )


# ==============================================================================
# 2. Tool Wrapper
# ==============================================================================

class _ToolPlugin(ChimeraPlugin):
    """Internal helper to bridge Tool -> ChimeraPlugin logic"""
    def process(self, input_data): pass

class GuardedTool(BaseTool):
    """
    Proxy that enforces policy before executing a LangChain Tool.
    Uses Composition instead of Inheritance for ChimeraPlugin to avoid Pydantic conflicts.
    """
    name: str
    description: str
    args_schema: Optional[Any] = None
    original_tool: Any
    
    # Internal state (Excluded from Pydantic schema via private naming)
    _plugin: Any = None 
    _inject: Dict = {}
    _tool_field: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Internal fields are set via factory, initialized empty here
        self._inject = {}

    def _run(self, *args, **kwargs):
        # 1. Prepare Input
        tool_input = kwargs if kwargs else (args[0] if args else {})
        
        # 2. Context Injection
        extra = self._inject.copy()
        if self._tool_field:
            extra[self._tool_field] = getattr(self.original_tool, "name", self.name)

        # 3. Run Guard (via internal plugin)
        if self._plugin:
            self._plugin.run_guard(tool_input, extra_context=extra)
            
        # 4. Execute Original
        return self.original_tool._run(*args, **kwargs)

    async def _arun(self, *args, **kwargs):
        tool_input = kwargs if kwargs else (args[0] if args else {})
        extra = self._inject.copy()
        if self._tool_field:
            extra[self._tool_field] = getattr(self.original_tool, "name", self.name)

        if self._plugin:
            self._plugin.run_guard(tool_input, extra_context=extra)

        if hasattr(self.original_tool, "_arun"):
            return await self.original_tool._arun(*args, **kwargs)
        return self.original_tool._run(*args, **kwargs)


def wrap_tool(
    tool: BaseTool,
    guard: ChimeraGuard,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    tool_field: Optional[str] = None,
    enable_dashboard: bool = False
) -> BaseTool:
    """Wraps a tool with ChimeraGuard protection."""
    
    # 1. Create Wrapper
    wrapper = GuardedTool(
        name=getattr(tool, "name", "unknown"),
        description=getattr(tool, "description", ""),
        args_schema=getattr(tool, "args_schema", None),
        original_tool=tool
    )
    
    # 2. Attach Logic Engine (with mapper!)
    wrapper._plugin = _ToolPlugin(
        constitution=guard.constitution,
        enable_dashboard=enable_dashboard,
        context_mapper=context_mapper, # <--- Critical: Pass mapper to engine
        title=f"Tool::{wrapper.name}"
    )
    wrapper._inject = inject or {}
    wrapper._tool_field = tool_field
    
    return wrapper

def guard_tools(
    tools: Iterable[BaseTool],
    guard: ChimeraGuard,
    *,
    context_mapper: Optional[ContextMapper] = None,
    inject: Optional[Dict[str, Any]] = None,
    tool_field: Optional[str] = None,
    enable_dashboard: bool = False
) -> List[BaseTool]:
    """Wraps multiple tools at once."""
    return [
        wrap_tool(
            t, 
            guard, 
            context_mapper=context_mapper, 
            inject=inject, 
            tool_field=tool_field, 
            enable_dashboard=enable_dashboard
        )
        for t in tools
    ]