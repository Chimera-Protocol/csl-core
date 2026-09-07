"""
OpenClaw Plugin Configuration

Sensible defaults for OpenClaw integration.
All values can be overridden via environment variables or constructor args.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set


# Default domains considered safe for navigation/fetch
_DEFAULT_DOMAIN_ALLOWLIST: Set[str] = {
    # Search & productivity
    "google.com",
    "gmail.com",
    "docs.google.com",
    "drive.google.com",
    "calendar.google.com",
    # Development
    "github.com",
    "gitlab.com",
    "stackoverflow.com",
    "npmjs.com",
    "pypi.org",
    # Communication
    "slack.com",
    "discord.com",
    "telegram.org",
    # Cloud providers
    "aws.amazon.com",
    "console.cloud.google.com",
    "portal.azure.com",
    # OpenClaw ecosystem
    "docs.openclaw.ai",
    "openclaw.ai",
}


@dataclass
class OpenClawConfig:
    """Configuration for CSL-Core OpenClaw plugin.

    Priority: constructor args > environment variables > defaults.
    """

    # Path to compiled CSL policy file
    policy_path: str = ""

    # Deployment mode: DESKTOP, SERVER, EMBEDDED, UNATTENDED
    # Override: CSL_DEPLOYMENT_MODE
    deployment_mode: str = "DESKTOP"

    # Workspace root for path boundary checks
    # Override: CSL_WORKSPACE_ROOT or OPENCLAW_WORKSPACE
    workspace_root: str = ""

    # Whether sandbox mode is active
    # Override: CSL_SANDBOX_ACTIVE
    sandbox_active: bool = False

    # Domain allowlist for browser/network safety
    domain_allowlist: Set[str] = field(default_factory=lambda: _DEFAULT_DOMAIN_ALLOWLIST.copy())

    # Enable PII scanning on tool params
    pii_scanning_enabled: bool = True

    # Log blocked actions to stderr
    log_blocks: bool = True

    # Fail-closed vs fail-open default for the generic extractors in context_mapper.py
    # (_extract_path_check, _extract_domain_check) when a tool call's params contain no
    # recognizable path-like or URL-like field. strict=True (default): treat "can't tell"
    # as "not safe" (path_in_workspace/domain_allowlisted -> "NO"), so an unrecognized
    # field shape blocks rather than silently passes. strict=False restores the previous
    # fail-open behavior ("can't tell" -> "YES"), which is more permissive but can let an
    # integration with nonstandard param names bypass path/domain checks entirely.
    # Cost of strict=True: an integration whose tool params use field names outside
    # _PATH_KEYS/_URL_KEYS (and don't match the absolute-path/URL heuristics either) will
    # get blocked by policies that gate on path_in_workspace/domain_allowlisted, even when
    # the call was legitimate — false positives, not silent bypasses.
    # Override: CSL_STRICT_UNRECOGNIZED_FIELDS
    strict_unrecognized_fields: bool = True

    def __post_init__(self):
        """Apply environment variable overrides."""

        # Deployment mode
        if not self.deployment_mode or self.deployment_mode == "DESKTOP":
            env_mode = os.environ.get("CSL_DEPLOYMENT_MODE", "")
            if env_mode:
                self.deployment_mode = env_mode.upper()

        # Workspace root
        if not self.workspace_root:
            self.workspace_root = os.environ.get(
                "CSL_WORKSPACE_ROOT",
                os.environ.get(
                    "OPENCLAW_WORKSPACE",
                    str(Path.home() / ".openclaw" / "workspace"),
                ),
            )

        # Sandbox
        env_sandbox = os.environ.get("CSL_SANDBOX_ACTIVE", "")
        if env_sandbox:
            self.sandbox_active = env_sandbox.lower() in ("1", "true", "yes")

        # Strict/permissive default for unrecognized path/URL fields
        env_strict = os.environ.get("CSL_STRICT_UNRECOGNIZED_FIELDS", "")
        if env_strict:
            self.strict_unrecognized_fields = env_strict.lower() in ("1", "true", "yes")

        # Extra domains from env (comma-separated)
        env_domains = os.environ.get("CSL_DOMAIN_ALLOWLIST", "")
        if env_domains:
            for domain in env_domains.split(","):
                domain = domain.strip().lower()
                if domain:
                    self.domain_allowlist.add(domain)

    def is_domain_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the allowlist.

        Handles subdomains: 'mail.google.com' matches 'google.com'.
        """
        try:
            # Extract domain from URL
            domain = url.lower().strip()

            # Strip protocol
            if "://" in domain:
                domain = domain.split("://", 1)[1]

            # Strip path, query, fragment
            domain = domain.split("/", 1)[0]
            domain = domain.split("?", 1)[0]
            domain = domain.split("#", 1)[0]

            # Strip port
            domain = domain.split(":", 1)[0]

            # Check exact match and parent domain match
            for allowed in self.domain_allowlist:
                if domain == allowed or domain.endswith("." + allowed):
                    return True

            return False
        except Exception:
            return False

    def is_path_in_workspace(self, file_path: str) -> bool:
        """Check if a file path is within the workspace boundary."""
        try:
            resolved = Path(file_path).resolve()
            workspace = Path(self.workspace_root).resolve()
            return str(resolved).startswith(str(workspace))
        except Exception:
            return False
