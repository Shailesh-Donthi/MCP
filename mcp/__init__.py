"""
Compatibility shim for the MCP package.

Actual backend package code now lives under ``server/mcp``. This shim keeps
existing imports like ``import mcp.server_http`` working from the project root.
"""

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
_SERVER_PKG_DIR = _ROOT / "server" / "mcp"
_SERVER_INIT = _SERVER_PKG_DIR / "__init__.py"

if not _SERVER_PKG_DIR.is_dir() or not _SERVER_INIT.is_file():
    raise ImportError(f"Backend package not found at {_SERVER_PKG_DIR}")

# Point package submodule discovery to the real backend package location.
__path__ = [str(_SERVER_PKG_DIR)]
__file__ = str(_SERVER_INIT)

# Execute the real package __init__ so exported names (e.g., mcp_settings)
# remain available exactly as before.
exec(compile(_SERVER_INIT.read_text(encoding="utf-8-sig"), __file__, "exec"), globals(), globals())
