"""
Venom data model.

Plain dataclasses with a generic, lossless to_dict / from_dict. Unknown values are
None ("n/a" on screen), never zero.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# Risk classes, ordered from least to most sensitive.
RISK_CLASSES = ["READ", "WRITE", "EXTERNAL", "IDENTITY", "EXEC", "SPEND", "DESTRUCTIVE", "UNCLASSIFIED"]
SENSITIVE = {"WRITE", "EXTERNAL", "IDENTITY", "EXEC", "SPEND", "DESTRUCTIVE", "UNCLASSIFIED"}
SEVERITIES = ["high", "medium", "low", "info"]


class _Serde:
    """Mixin: dataclass <-> plain dict, driven by type hints."""

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _to_plain(getattr(self, f.name)) for f in dataclasses.fields(self)}  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for f in dataclasses.fields(cls):  # type: ignore[arg-type]
            if f.name in data:
                kwargs[f.name] = _from_plain(hints[f.name], data[f.name])
        return cls(**kwargs)


def _to_plain(v: Any) -> Any:
    if isinstance(v, _Serde):
        return v.to_dict()
    if isinstance(v, list):
        return [_to_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    return v


def _from_plain(tp: Any, v: Any) -> Any:
    if v is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        inner = [a for a in args if a is not type(None)]
        return _from_plain(inner[0], v) if len(inner) == 1 else v
    if origin in (list, List):
        return [_from_plain(args[0], x) for x in v] if args else list(v)
    if origin in (dict, Dict):
        return {k: _from_plain(args[1], x) for k, x in v.items()} if args else dict(v)
    if isinstance(tp, type) and issubclass(tp, _Serde):
        return tp.from_dict(v)
    return v


@dataclass
class Evidence(_Serde):
    layer: str
    path: str
    line: Optional[int] = None
    detail: Optional[str] = None


@dataclass
class ToolParam(_Serde):
    name: str
    type: Optional[str] = None
    required: bool = True
    enum: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Tool(_Serde):
    name: str
    source: str  # decorator | Tool() | function_schema | mcp_server | mcp_tool | langgraph_node | builtin
    params: List[ToolParam] = field(default_factory=list)
    description: Optional[str] = None
    risk_class: str = "UNCLASSIFIED"
    risk_reason: Optional[str] = None
    mcp_server: Optional[str] = None
    coverage: Optional[str] = None  # guarded | wired_no_rule | unguarded | exempt
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Credential(_Serde):
    name: str
    file: str
    kind: str = "other"  # cloud_root | org_token | llm_key | database_url | payment | other


@dataclass
class Access(_Serde):
    credentials: List[Credential] = field(default_factory=list)
    fs_roots: List[str] = field(default_factory=list)
    network_listen: List[str] = field(default_factory=list)
    permission_mode: Optional[str] = None  # default | acceptEdits | plan | bypass
    elevated: bool = False


@dataclass
class Trigger(_Serde):
    type: str  # time | inbound_http | messaging | email | manual
    schedule: Optional[str] = None
    source: Optional[str] = None


@dataclass
class RunStats(_Serde):
    count: Optional[int] = None
    window: str = "7d"
    last_run: Optional[str] = None
    per_day: Optional[List[int]] = None
    source: Optional[str] = None


@dataclass
class Guard(_Serde):
    status: str = "none"  # none | wired | wired_no_rule
    policy_ids: List[str] = field(default_factory=list)
    mechanism: Optional[str] = None  # wrapper | hook | plugin
    mode: Optional[str] = None  # log | block
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Exemption(_Serde):
    agent: str
    scope: str = "agent"  # agent | tool | rule
    tool: Optional[str] = None
    reason: Optional[str] = None
    approved_by: Optional[str] = None
    expires: Optional[str] = None
    status: str = "proposed"  # proposed | approved
    rule: Optional[str] = None  # scope "rule": exempt the agent from this one rule


@dataclass
class PromptInfo(_Serde):
    present: bool = False
    length: Optional[int] = None
    sha256: Optional[str] = None


@dataclass
class Agent(_Serde):
    id: str
    display_name: str
    kind: str  # code | assistant | service | scheduled | unmanaged
    state: str = "unknown"  # running | stopped | scheduled | configured | unknown
    framework: List[str] = field(default_factory=list)
    model_ids: List[str] = field(default_factory=list)
    entrypoint: Optional[str] = None
    project: Optional[str] = None
    process_user: Optional[str] = None
    pids: List[int] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)
    access: Access = field(default_factory=Access)
    triggers: List[Trigger] = field(default_factory=list)
    runs: RunStats = field(default_factory=RunStats)
    guard: Guard = field(default_factory=Guard)
    exempt: Optional[Exemption] = None
    system_prompt: PromptInfo = field(default_factory=PromptInfo)
    sessions: Optional[int] = None


@dataclass
class PolicyRef(_Serde):
    path: str
    status: str = "active"  # active | draft | found
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    policy_hash: Optional[str] = None
    domain: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)
    vocabulary: Dict[str, List[str]] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    rule_values: Dict[str, List[str]] = field(default_factory=dict)  # rule -> enum literals it mentions
    error: Optional[str] = None


@dataclass
class Coverage(_Serde):
    tools_total: int = 0
    guarded: int = 0
    wired_no_rule: int = 0
    unguarded: int = 0
    exempt: int = 0

    @property
    def ratio(self) -> Optional[float]:
        denom = self.tools_total - self.exempt
        return None if denom <= 0 else self.guarded / denom


@dataclass
class DriftItem(_Serde):
    kind: str  # unknown_value | unsupplied_variable | coercion
    policy: str
    variable: str
    value: Optional[str] = None
    suggestion: Optional[str] = None
    agent_id: Optional[str] = None
    tool: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class Finding(_Serde):
    id: str
    rule: str
    severity: str
    summary: str
    agent_id: Optional[str] = None
    tool: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    recommendation: Optional[str] = None


@dataclass
class Host(_Serde):
    name: str = ""
    os: str = ""
    scanned_at: str = ""
    duration_ms: int = 0
    scope: str = ""
    mode: str = "host"  # host | folder | fixture
    layers_run: List[str] = field(default_factory=list)
    layers_unavailable: Dict[str, str] = field(default_factory=dict)
    files_scanned: int = 0
    parse_errors: int = 0
    not_readable: int = 0
    partial: bool = False


@dataclass
class Inventory(_Serde):
    schema_version: int = SCHEMA_VERSION
    tool_version: str = ""
    host: Host = field(default_factory=Host)
    agents: List[Agent] = field(default_factory=list)
    policies: List[PolicyRef] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    drift: List[DriftItem] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    exempted: List[Finding] = field(default_factory=list)
    rules_not_evaluated: Dict[str, str] = field(default_factory=dict)

    def agent(self, agent_id: str) -> Optional[Agent]:
        for a in self.agents:
            if a.id == agent_id or a.display_name == agent_id:
                return a
        return None


ALL_MODELS = [
    Evidence, ToolParam, Tool, Credential, Access, Trigger, RunStats, Guard, Exemption,
    PromptInfo, Agent, PolicyRef, Coverage, DriftItem, Finding, Host, Inventory,
]
