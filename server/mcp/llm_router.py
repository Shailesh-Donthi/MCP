"""Version-dispatch module for MCP LLM router."""

from __future__ import annotations

import sys
from types import ModuleType

from mcp.runtime_switch import get_logic_version, import_versioned_module

ACTIVE_LOGIC_VERSION = get_logic_version()
_selected_module: ModuleType = import_versioned_module("llm_router")

# Preserve visibility of the active selector for diagnostics.
setattr(_selected_module, "ACTIVE_LOGIC_VERSION", ACTIVE_LOGIC_VERSION)

# Expose the selected module object itself so monkeypatching
# `mcp.llm_router.*` updates globals actually used by routed functions.
sys.modules[__name__] = _selected_module
