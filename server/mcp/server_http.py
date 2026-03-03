"""Version-dispatch module for MCP HTTP server."""

from __future__ import annotations

import sys
from types import ModuleType

from mcp.runtime_switch import get_logic_version, import_versioned_module

ACTIVE_LOGIC_VERSION = get_logic_version()
_selected_module: ModuleType = import_versioned_module("server_http")

setattr(_selected_module, "ACTIVE_LOGIC_VERSION", ACTIVE_LOGIC_VERSION)

# Expose active module directly so imports/patches bind to live globals.
sys.modules[__name__] = _selected_module
