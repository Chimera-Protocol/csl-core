from __future__ import annotations

from pathlib import Path
import pytest

from chimera_core.language.parser import parse_csl_file
from chimera_core.language.compiler import CSLCompiler


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # tests/ -> repo root
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    return repo_root / "examples"


@pytest.fixture(scope="session")
def agent_tool_guard_policy_path(examples_dir: Path) -> Path:
    p = examples_dir / "agent_tool_guard.csl"
    assert p.exists(), f"Missing example policy: {p}"
    return p


@pytest.fixture(scope="session")
def banking_policy_path(examples_dir: Path) -> Path:
    p = examples_dir / "chimera_banking_case_study.csl"
    assert p.exists(), f"Missing example policy: {p}"
    return p

@pytest.fixture(scope="session")
def compiled_agent_tool_guard(agent_tool_guard_policy_path: Path):
    compiler = CSLCompiler()
    constitution = parse_csl_file(str(agent_tool_guard_policy_path))
    compiled = compiler.compile(constitution)
    return compiled


def compile_policy(policy_path: Path):
    constitution = parse_csl_file(str(policy_path))
    compiler = CSLCompiler()
    return compiler.compile(constitution)


