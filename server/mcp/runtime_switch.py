"""Runtime selector for MCP logic versions."""

from __future__ import annotations

import importlib
import os
from typing import Any

from mcp.config import mcp_settings

_VALID_VERSIONS = {"v1", "v2"}


def _get_toggle_file_version() -> str | None:
    try:
        from mcp.logic_switch import USE_V2

        if isinstance(USE_V2, bool):
            return "v2" if USE_V2 else "v1"
    except Exception:
        return None
    return None


def get_logic_version() -> str:
    """Return active MCP logic version (`v1` or `v2`)."""
    toggle_version = _get_toggle_file_version()
    if toggle_version in _VALID_VERSIONS:
        return toggle_version

    env_value = os.getenv("MCP_LOGIC_VERSION")
    configured = (env_value or getattr(mcp_settings, "MCP_LOGIC_VERSION", "v1") or "v1").strip().lower()
    if configured not in _VALID_VERSIONS:
        return "v1"
    return configured


def get_module_path(module: str) -> str:
    """
    Return fully-qualified module path for selected logic version.

    Example:
        get_module_path("server_http") -> "mcp.v1.server_http" or "mcp.v2.server_http"
    """
    return f"mcp.{get_logic_version()}.{module}"


def import_versioned_module(module: str) -> Any:
    """Import module from the selected logic version."""
    return importlib.import_module(get_module_path(module))
