from __future__ import annotations

from chimera_core.plugins.openclaw.config import OpenClawConfig
from chimera_core.plugins.openclaw.context_mapper import map_context


def test_unrecognized_path_field_fails_closed_by_default():
    """An unrecognized param shape must not silently look 'in workspace'."""
    config = OpenClawConfig(workspace_root="/home/user/workspace")
    ctx = map_context("unknown_future_tool", {"weird_field_name": "something"}, {}, config)
    assert ctx["path_in_workspace"] == "NO"


def test_unrecognized_domain_field_fails_closed_by_default():
    config = OpenClawConfig(workspace_root="/home/user/workspace")
    ctx = map_context("unknown_future_tool", {"weird_field_name": "something"}, {}, config)
    assert ctx["domain_allowlisted"] == "NO"


def test_strict_unrecognized_fields_can_be_disabled():
    config = OpenClawConfig(
        workspace_root="/home/user/workspace",
        strict_unrecognized_fields=False,
    )
    ctx = map_context("unknown_future_tool", {"weird_field_name": "something"}, {}, config)
    assert ctx["path_in_workspace"] == "YES"
    assert ctx["domain_allowlisted"] == "YES"


def test_recognized_path_field_still_checked_normally():
    config = OpenClawConfig(workspace_root="/home/user/workspace")
    ctx = map_context("read_file", {"path": "/home/user/workspace/notes.txt"}, {}, config)
    assert ctx["path_in_workspace"] == "YES"

    ctx_outside = map_context("read_file", {"path": "/etc/passwd"}, {}, config)
    assert ctx_outside["path_in_workspace"] == "NO"


def test_recognized_domain_field_still_checked_normally():
    config = OpenClawConfig()
    ctx = map_context("fetch_url", {"url": "https://github.com/foo/bar"}, {}, config)
    assert ctx["domain_allowlisted"] == "YES"

    ctx_blocked = map_context("fetch_url", {"url": "https://evil.example.com"}, {}, config)
    assert ctx_blocked["domain_allowlisted"] == "NO"


def test_env_override_flips_strict_default(monkeypatch):
    monkeypatch.setenv("CSL_STRICT_UNRECOGNIZED_FIELDS", "false")
    config = OpenClawConfig(workspace_root="/home/user/workspace")
    assert config.strict_unrecognized_fields is False
