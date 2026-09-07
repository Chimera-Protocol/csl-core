"""
Context Mapper Generic Variable Extraction

Converts arbitrary tool call data into CSL context variables
WITHOUT knowing any tool names. Pure data-type inspection only.

Principle: Python is a dumb guard. It looks at data shapes, not tool semantics.
           Which tools need which checks → that's the .csl file's job.

Example:
    params = {"message_ids": ["a","b","c"], "path": "/etc/passwd"}
    metadata = {"sender_role": "PAIRED"}

    context = map_context("some_future_tool", params, metadata, config)
    # → {"tool": "some_future_tool", "sender_role": "PAIRED", "target_count": 3,
    #     "path_in_workspace": "NO", ...}

    Whether target_count=3 matters for "some_future_tool" → .csl decides, not Python.
    If someone adds "kubernetes_pod_destroy" tomorrow → this code doesn't change.
"""

from typing import Any, Dict

from .config import OpenClawConfig
from . import pii_detector


# Generic key families — we look for these in ANY tool's params
_PATH_KEYS = frozenset({
    "path", "file", "file_path", "filepath", "dir", "directory",
    "target", "dest", "destination", "source", "src", "filename",
})

_URL_KEYS = frozenset({
    "url", "href", "uri", "endpoint", "address", "link",
})


def map_context(
    tool_name: str,
    tool_params: Dict[str, Any],
    metadata: Dict[str, Any],
    config: OpenClawConfig,
) -> Dict[str, Any]:
    """Map tool call data to CSL context variables.

    This function inspects DATA TYPES and KEY NAMES, never tool names.
    All tool-specific logic lives exclusively in the .csl policy file.

    Any variable the policy doesn't reference is silently ignored by the
    CSL engine, so it's always safe to include all variables.

    Args:
        tool_name: Passed through as-is. Python never interprets this.
        tool_params: Tool arguments dict (arbitrary structure).
        metadata: Event metadata (sender info, session context).
        config: Plugin configuration.

    Returns:
        Dict with CSL context variables.
    """
    return {
        "tool": tool_name,  # Only tool-specific value: pass-through, never interpreted
        "sender_role": _extract_sender_role(metadata),
        "deployment_mode": _extract_deployment_mode(metadata, config),
        "target_count": _extract_list_count(tool_params),
        "path_in_workspace": _extract_path_check(tool_params, config),
        "domain_allowlisted": _extract_domain_check(tool_params, config),
        "skill_verified": _extract_flag(metadata, "skill_verified", "verified", default="YES"),
        "approval_granted": _extract_flag(metadata, "approval_granted", "approved", default="NO"),
        "sandbox_active": "YES" if config.sandbox_active else "NO",
        "pii_present": _extract_pii(tool_params, config),
    }


# -------------------------------------------------------------------
# Generic extractors — inspect data shapes, never tool semantics
# -------------------------------------------------------------------


def _extract_sender_role(metadata: Dict[str, Any]) -> str:
    """Extract sender trust tier from metadata.

    Checks explicit role first, then derives from auth flags.
    This is about WHO is calling, not WHAT they're calling.
    """
    # Direct role override (from gateway auth layer)
    role = metadata.get("sender_role", "")
    if role in ("OWNER", "PAIRED", "UNPAIRED", "UNKNOWN"):
        return role

    # Derive from auth flags
    if metadata.get("is_owner", False):
        return "OWNER"
    if metadata.get("is_paired", False) or metadata.get("paired", False):
        return "PAIRED"
    if metadata.get("senderId") or metadata.get("sender_id"):
        return "UNPAIRED"

    return "UNKNOWN"


def _extract_deployment_mode(metadata: Dict[str, Any], config: OpenClawConfig) -> str:
    """Extract deployment mode. Priority: metadata > config > env."""
    mode = metadata.get("deployment_mode", "")
    if mode in ("DESKTOP", "SERVER", "EMBEDDED", "UNATTENDED"):
        return mode
    return config.deployment_mode


def _extract_list_count(params: Dict[str, Any]) -> int:
    """Find the LARGEST list/array in params and return its length.

    Generic: doesn't care if it's message_ids, recipients, file_paths,
    pod_names, or anything else. If there's a list, count it.
    The .csl policy decides whether that count matters for a given tool.

    Returns max list length to catch the worst-case scenario
    (e.g. params with both "tags": [..] and "message_ids": [..]).
    """
    max_len = 0

    for value in params.values():
        if isinstance(value, (list, tuple)):
            max_len = max(max_len, len(value))

    # Check for comma-separated strings (common for recipients)
    if max_len <= 1:
        for value in params.values():
            if isinstance(value, str) and "," in value:
                parts = [p.strip() for p in value.split(",") if p.strip()]
                if len(parts) > 1:
                    max_len = max(max_len, len(parts))

    return max_len if max_len > 0 else 1


def _extract_path_check(params: Dict[str, Any], config: OpenClawConfig) -> str:
    """Look for any path-like value in params and check workspace boundary.

    Strategy:
      1. Scan keys matching _PATH_KEYS → check value against workspace
      2. Fallback: scan all string values starting with "/" (absolute paths)

    Tool-agnostic. Whether this result matters → .csl decides.

    When no path-like field is found at all, this is genuinely ambiguous: it might not be
    a filesystem operation, or it might be one using a param name this extractor doesn't
    recognize. config.strict_unrecognized_fields (default True) decides which way that
    ambiguity resolves: "NO" (not in workspace / treat as unsafe) so a policy gating on
    path_in_workspace fails closed, at the cost of false positives on legitimate calls
    with nonstandard param names. Set strict_unrecognized_fields=False to restore the
    previous fail-open "YES" default.
    """
    # Strategy 1: known key names
    for key, value in params.items():
        if key.lower() in _PATH_KEYS and isinstance(value, str) and value:
            return "YES" if config.is_path_in_workspace(value) else "NO"

    # Strategy 2: absolute path heuristic
    for value in params.values():
        if isinstance(value, str) and value.startswith("/"):
            return "YES" if config.is_path_in_workspace(value) else "NO"

    # No path found: ambiguous (not a filesystem op, or an unrecognized param name).
    # Fail closed by default; see docstring.
    return "NO" if config.strict_unrecognized_fields else "YES"


def _extract_domain_check(params: Dict[str, Any], config: OpenClawConfig) -> str:
    """Look for any URL-like value in params and check domain allowlist.

    Strategy:
      1. Scan keys matching _URL_KEYS → check domain
      2. Fallback: scan all string values containing "://" or "www."

    Tool-agnostic. Whether this result matters → .csl decides.

    Same ambiguity as _extract_path_check when no URL-like field is found: this could be
    a non-network operation, or a network operation this extractor didn't recognize.
    config.strict_unrecognized_fields (default True) fails this closed ("NO" / not
    allowlisted) rather than open, at the cost of false positives on legitimate calls
    with nonstandard param names.
    """
    # Strategy 1: known key names
    for key, value in params.items():
        if key.lower() in _URL_KEYS and isinstance(value, str) and value:
            return "YES" if config.is_domain_allowed(value) else "NO"

    # Strategy 2: URL pattern heuristic
    for value in params.values():
        if isinstance(value, str) and ("://" in value or value.startswith("www.")):
            return "YES" if config.is_domain_allowed(value) else "NO"

    # No URL found: ambiguous (not a network op, or an unrecognized param name).
    # Fail closed by default; see docstring.
    return "NO" if config.strict_unrecognized_fields else "YES"


def _extract_flag(metadata: Dict[str, Any], *keys: str, default: str = "NO") -> str:
    """Extract a YES/NO flag from metadata, trying multiple key names.

    Generic: accepts any boolean-like value (bool, "true", "yes", "1").
    """
    for key in keys:
        val = metadata.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            return "YES" if val else "NO"
        if isinstance(val, str):
            return "YES" if val.upper() in ("YES", "TRUE", "1") else "NO"
    return default


def _extract_pii(params: Dict[str, Any], config: OpenClawConfig) -> str:
    """Scan all params for PII. Tool-agnostic — scans everything."""
    if not config.pii_scanning_enabled:
        return "NO"
    return "YES" if pii_detector.scan(params) else "NO"
