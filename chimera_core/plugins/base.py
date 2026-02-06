"""
CSL-Core Plugin Architecture
============================
Base infrastructure for all ChimeraGuard integrations.

This module provides:
1. Universal Data Mappers: Convert arbitrary Python objects to CSL context.
2. ChimeraPlugin: Abstract Base Class ensuring consistent lifecycle & visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Mapping
from dataclasses import is_dataclass, asdict

from ..runtime import ChimeraGuard, ChimeraError, GuardResult
from ..language.compiler import CompiledConstitution

# Visualizer import with fallback for environments without 'rich'
try:
    from ..audit.visualizer import RuntimeVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    RuntimeVisualizer = None  # type: ignore

# --- PART 1: UNIVERSAL DATA MAPPERS (The Brain) ---

ContextMapper = Callable[[Any], Dict[str, Any]]

def safe_model_dump(obj: Any) -> Optional[Dict[str, Any]]:
    """Helper: Best-effort Pydantic v1/v2 to dict conversion."""
    for method in ["model_dump", "dict"]:
        try:
            func = getattr(obj, method, None)
            if callable(func):
                out = func()
                if isinstance(out, dict):
                    return out
        except Exception:
            continue
    return None

def default_context_mapper(input_data: Any) -> Dict[str, Any]:
    """
    Universal Normalizer: Converts ANY object into a CSL context dict.
    Strategy: Dict > String > Dataclass > Pydantic > Object Attributes > Fallback
    """
    if isinstance(input_data, Mapping):
        return dict(input_data)
    
    if isinstance(input_data, str):
        return {"content": input_data}
    
    if is_dataclass(input_data):
        return asdict(input_data)
    
    pd_dict = safe_model_dump(input_data)
    if pd_dict:
        return pd_dict

    # Generic object with .content (e.g., framework messages)
    if hasattr(input_data, "content"):
        ctx = {"content": getattr(input_data, "content")}
        if hasattr(input_data, "role"):
            ctx["role"] = str(input_data.role)
        return ctx

    try:
        return dict(vars(input_data))
    except Exception:
        return {"content": str(input_data)}

# --- PART 2: THE PLUGIN ARCHITECTURE (The Skeleton) ---

class ChimeraPlugin(ABC):
    """
    Base class for Chimera integrations.
    Provides a shared deterministic pipeline:
      normalize -> guard.verify -> (optional) visualize -> pass-through/raise
    """

    def __init__(
        self,
        constitution: CompiledConstitution,
        enable_dashboard: bool = False,
        title: Optional[str] = None,
        context_mapper: Optional[ContextMapper] = None,
    ):
        self.guard = ChimeraGuard(constitution)
        self.enable_dashboard = enable_dashboard and VISUALIZER_AVAILABLE
        self.visualizer = RuntimeVisualizer() if self.enable_dashboard else None
        
        # Policy name extraction handling potential missing attributes safely
        domain = getattr(constitution, 'domain_name', 'unknown')
        self.title = title or f"ChimeraGuard::{domain}"
        
        # Use user-provided mapper or fallback to the universal one
        self.context_mapper = context_mapper or default_context_mapper

    def normalize_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Default normalization using the universal mapper.
        Subclasses can override this for framework-specific extraction logic.
        """
        return self.context_mapper(input_data)

    def run_guard(self, input_data: Any, extra_context: Optional[Dict] = None) -> GuardResult:
        """
        Executes the policy verification loop.
        Returns the GuardResult object. Raises ChimeraError on BLOCK.
        """
        # 1. Prepare Context
        context = self.normalize_input(input_data)
        if extra_context:
            context.update(extra_context)

        try:
            # 2. Verify
            result = self.guard.verify(context)
            
            # 3. Visualize (Success Case)
            if self.enable_dashboard and self.visualizer:
                self.visualizer.visualize(result=result, context=context, title=self.title)
            
            return result

        except ChimeraError as e:
            # 4. Visualize (Failure Case) - deterministic reporting
            if self.enable_dashboard and self.visualizer:
                # Reconstruct a result object from the error for visualization
                fake_result = GuardResult(
                    allowed=False,
                    violations=[str(e)],
                    latency_ms=0.0, # Could be improved with timing logic
                )
                self.visualizer.visualize(result=fake_result, context=context, title=self.title)
            
            # 5. Re-raise to block execution
            raise

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Plugin-specific entry point (e.g., invoke, run, call).
        Should call self.run_guard(input_data).
        """
        ...