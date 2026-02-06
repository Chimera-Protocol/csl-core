"""
CSL-Core Factory
================
The manufacturing plant for ChimeraGuard instances.
This module handles the heavy lifting (Parsing & Compilation) 
so the Runtime can stay lightweight.
"""

import os
from typing import Optional, Union
from pathlib import Path

from .runtime import ChimeraGuard, RuntimeConfig
from .language.parser import parse_csl_file, parse_csl
from .language.compiler import CSLCompiler

def load_guard(
    policy_path: Union[str, Path], 
    config: Optional[RuntimeConfig] = None
) -> ChimeraGuard:
    """
    Factory Method: Loads a .csl file from disk, compiles it, and returns a Guard.
    
    Args:
        policy_path: Path to the .csl file (str or Path object).
                     Supports relative paths and user expansion (~/).
        config: Optional runtime configuration override.

    Returns:
        A ready-to-use ChimeraGuard instance.

    Raises:
        FileNotFoundError: If the policy file does not exist.
        CompilationError: If the CSL syntax or logic is invalid.
    """
    # 1. Path Normalization & Validation
    # - expanduser: Handles "~/policies/..."
    # - resolve: Converts relative paths to absolute, fixing ".." confusion
    path_obj = Path(policy_path).expanduser().resolve()

    if not path_obj.exists():
        raise FileNotFoundError(
            f"❌ CSL Policy file not found at: '{path_obj}'\n"
            f"   (Current working directory: '{os.getcwd()}')"
        )

    # 2. Parse (Read file -> AST)
    # We pass the absolute path string to the parser
    constitution_ast = parse_csl_file(str(path_obj))
    
    # 3. Compile (AST -> Logic Model)
    # Note: Creating a fresh compiler instance is cheap for now.
    # If compilation gets heavy, we can add LRU caching here later.
    compiler = CSLCompiler()
    compiled_policy = compiler.compile(constitution_ast)
    
    # 4. Construct (Logic Model -> Runtime Guard)
    return ChimeraGuard(compiled_policy, config)

def create_guard_from_string(
    policy_content: str, 
    config: Optional[RuntimeConfig] = None
) -> ChimeraGuard:
    """
    Factory Method: Creates a Guard directly from a CSL code string.
    Useful for dynamic policies, unit testing, or REPL usage.
    
    Args:
        policy_content: The raw CSL code as a string.
        config: Optional runtime configuration override.
    """
    # 1. Parse String
    ast = parse_csl(policy_content)
    
    # 2. Compile
    compiler = CSLCompiler()
    compiled = compiler.compile(ast)
    
    # 3. Construct
    return ChimeraGuard(compiled, config)