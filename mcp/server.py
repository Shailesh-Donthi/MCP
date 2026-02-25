"""MCP stdio placeholder entrypoint.

Use HTTP server for local execution:
    python mcp_server.py
or:
    uvicorn mcp.server_http:app --host 127.0.0.1 --port 8090
"""

from mcp.server_http import main


if __name__ == "__main__":
    main()
