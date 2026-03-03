"""Version-dispatch module for MCP LLM router."""

from mcp.runtime_switch import get_logic_version, import_versioned_module

ACTIVE_LOGIC_VERSION = get_logic_version()
_selected_module = import_versioned_module("llm_router")

for _name in dir(_selected_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_selected_module, _name)

if hasattr(_selected_module, "__all__"):
    __all__ = list(getattr(_selected_module, "__all__"))
else:
    __all__ = [name for name in globals() if not name.startswith("__")]
