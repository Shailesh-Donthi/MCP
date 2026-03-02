# Police Personnel Query Assistant - Technology & Model Overview

## 1) Core Stack

- Language: Python 3 (server-side), JavaScript (client-side)
- API framework: FastAPI
- ASGI server: Uvicorn
- Database: MongoDB
- Mongo drivers: `motor` (async), `pymongo`
- Cache/rate-limit support: Redis (`redis.asyncio`)
- HTTP client for LLM calls: `httpx`
- Config/validation: `pydantic`, `pydantic-settings`
- Auth/JWT libs: `python-jose`, `PyJWT`

Dependency source: [server/requirements.txt](C:/Users/shail/Downloads/MCP_Proj/server/requirements.txt)

## 2) Architecture

- Server entrypoint:
  - [mcp_server.py](C:/Users/shail/Downloads/MCP_Proj/mcp_server.py) (compatibility launcher)
  - [server/mcp_server.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp_server.py)
- Main HTTP/SSE app:
  - [server/mcp/server_http.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/server_http.py)
- Query orchestration layer:
  - [server/mcp/llm_router.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/llm_router.py)
- Tool execution registry:
  - [server/mcp/handlers/tool_handler.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/handlers/tool_handler.py)

The system uses deterministic tool execution with optional LLM-based routing/formatting on top.

## 3) LLM Model & Provider

LLM integration code: [server/mcp/router/llm_client.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/router/llm_client.py)

- OpenAI / Azure OpenAI-compatible:
  - Chat endpoint style: `/chat/completions`
  - Default model fallback: `gpt-4o-mini`
  - OpenAI env options: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
  - Azure env options: `AZURE_AI_ENDPOINT` or `AZURE_OPENAI_ENDPOINT`, plus `AZURE_AI_API_KEY` or `AZURE_OPENAI_API_KEY`, and `AZURE_AI_MODEL` or `AZURE_OPENAI_DEPLOYMENT`
  - Auto-detects Azure-compatible base URLs and uses `api-key` header for Azure.

Behavior:
- LLM routing is used when a key is configured.
- If LLM fails or returns invalid routing JSON, the app falls back to heuristic routing.

## 4) Data Layer (Mongo Collections)

Configured whitelist: [server/mcp/config.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/config.py)

Primary collections used include:
- `personnel_master`
- `assignment_master`
- `unit_master`
- `district_master`
- `rank_master`
- `designation_master`
- `department_master`
- plus master-data collections (`modules_master`, `prompt_master`, `notification_master`, etc.)

## 5) Tooling (Query Capabilities)

Registered tools: [server/mcp/handlers/tool_handler.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/handlers/tool_handler.py)

- `search_personnel`
- `search_unit`
- `check_responsible_user`
- `query_personnel_by_unit`
- `query_personnel_by_rank`
- `get_unit_hierarchy`
- `list_units_in_district`
- `list_districts`
- `count_vacancies_by_unit_rank`
- `get_personnel_distribution`
- `query_recent_transfers`
- `get_unit_command_history`
- `find_missing_village_mappings`
- `get_village_coverage`
- `query_linked_master_data`

## 6) Frontend

Client is plain HTML/CSS/JavaScript (no React/Vue build system in this repo):
- [client/chatbot.html](C:/Users/shail/Downloads/MCP_Proj/client/chatbot.html)
- [client/chatbot_app.js](C:/Users/shail/Downloads/MCP_Proj/client/chatbot_app.js)
- [client/chatbot_output.js](C:/Users/shail/Downloads/MCP_Proj/client/chatbot_output.js)

## 7) Runtime Configuration

Core app settings: [server/mcp/core/config.py](C:/Users/shail/Downloads/MCP_Proj/server/mcp/core/config.py)

Key runtime variables:
- DB: `MONGODB_URI`, `MONGODB_DB_NAME`
- Server: `MCP_HOST`, `MCP_PORT`
- CORS: `ALLOWED_ORIGINS`
- Redis: `REDIS_URL`
- Auth: `JWT_SECRET_KEY`, `JWT_ALGORITHM`
- Logging: `MCP_LOG_LEVEL`, `MCP_LOG_FILE`, `MCP_LOG_MAX_BYTES`, `MCP_LOG_BACKUP_COUNT`

## 8) Protocols & APIs

- REST endpoints for tools/query/chat
- SSE endpoint for MCP protocol streaming:
  - `/api/v1/mcp/sse`
- Health/metrics endpoints are also available in the FastAPI server.
