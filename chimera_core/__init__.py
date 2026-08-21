"""
CSL-Core (Chimera Core)

Deterministic safety layer for probabilistic models:
- Compile-time: parse -> validate -> Z3 verify -> compile to IR
- Runtime: deterministic guard enforcement (no Z3)
"""

from __future__ import annotations

__version__ = "0.5.0"

# --- 1. Core Runtime (Lightweight) ---
from .runtime import ChimeraGuard, ChimeraError, GuardResult, RuntimeConfig

# --- 2. Factory Methods ---
from .factory import load_guard, create_guard_from_string

# --- 3. Compiler (Advanced Usage) ---
from .language.compiler import CSLCompiler, CompilationError

# --- 4. Plugins (Optional Dependencies) ---
try:
    from .plugins.langchain import ChimeraLCEL  # requires extras: csl-core[langchain]
except (ImportError, AttributeError): 
    ChimeraLCEL = None  # type: ignore

__all__ = [
    "__version__",
    # Factory (Primary Entry Points)
    "load_guard",               
    "create_guard_from_string",
    # Runtime
    "ChimeraGuard",
    "ChimeraError",
    "GuardResult",
    "RuntimeConfig",
    # Compiler
    "CSLCompiler",
    "CompilationError",
    # Plugins
    "ChimeraLCEL",
]