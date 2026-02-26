"""Compatibility launcher for the MCP HTTP server.

Actual server code now lives under ``server/``.
"""

from pathlib import Path
import sys

_SERVER_DIR = Path(__file__).resolve().parent / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from mcp.server_http import main  # type: ignore  # resolved from server/mcp


if __name__ == "__main__":
    main()

