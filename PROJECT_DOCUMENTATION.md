# MCP_Proj Documentation

## Overview
This project is a Police Personnel Reporting assistant built on FastAPI, MongoDB, and a tool-based query layer (MCP style).

It supports:
- Direct tool execution via REST.
- Keyword-routed natural language queries.
- LLM-assisted intelligent routing and response formatting.
- A browser chat UI (`client/chatbot.html`) that calls `/api/v1/ask`.

Project layout note:
- `client/`: frontend UI assets (`chatbot.html`, `chatbot_app.js`, `chatbot_output.js`)
- `server/`: backend package and launcher (`server/mcp`, `server/mcp_server.py`, `server/requirements.txt`)
- Root wrappers remain for compatibility (`mcp_server.py`, `requirements.txt`, `mcp` shim package)

---

## High-Level Architecture
- `client/chatbot.html`: Frontend chat interface, quick actions, and API calls.
- `server/mcp/server_http.py`: Main HTTP server, endpoints, auth/context extraction, caching, metrics, fallback routing.
- `server/mcp/llm_router.py`: Orchestrator for intelligent query processing, session-state memory, tool execution, and response assembly.
- `server/mcp/router/*`: Router internals split into modular chunks (`prompts.py`, `llm_client.py`, `extractors.py`, `routing_rules.py`).
- `server/mcp/handlers/tool_handler.py`: Tool registry, validation, execution, and standardized error handling.
- `server/mcp/tools/*.py`: Domain tools for personnel, units, vacancies, transfers, village mapping, and search.
- `server/mcp/query_builder/filters.py`: Scope filtering by unit/district/state access.
- `server/mcp/core/database.py`: MongoDB connection lifecycle.
- `server/mcp/core/security.py`: JWT decode helper.
- `server/mcp/schemas/context_schema.py`: User access context model.

---

## Entry Points
- `mcp_server.py`: Root compatibility launcher (delegates to `server/` backend code).
- `server/mcp_server.py`: Backend launcher that calls `mcp.server_http.main()`.
- `server/mcp/server_http.py:main()`: Starts Uvicorn on `MCP_HOST:MCP_PORT`.

Run locally:
```bash
python mcp_server.py
```
or:
```bash
uvicorn mcp.server_http:app --host 127.0.0.1 --port 8090
```

---

## Main API Endpoints (`mcp/server_http.py`)
- `GET /health`: Service health status (MongoDB/Redis/tools loaded).
- `GET /ready`: Readiness probe.
- `GET /metrics`: Prometheus-style counters.
- `GET /api/v1/mcp/tools`: List available tools and schemas.
- `GET /api/v1/mcp/tools/{tool_name}`: Get one tool schema.
- `POST /api/v1/mcp/tools/{tool_name}/execute`: Execute one tool directly.
- `POST /api/v1/query`: Keyword-based NL routing to tools.
- `POST /api/v1/ask`: Intelligent query handler (LLM + rule/fallback) with output formatting layer.
- `GET /api/v1/mcp/sse`: SSE stream with tool list + keepalive heartbeat.

---

## Core Server Functions (`mcp/server_http.py`)
- `lifespan(app)`: Startup/shutdown for DB, tool initialization, optional Redis.
- `get_current_user_context(...)`: Builds `UserContext` from JWT or defaults to state scope.
- `_determine_scope_level(payload)`: Maps role names to `unit`/`district`/`state`.
- `check_rate_limit(...)`: Redis-based rate limiting.
- `generate_cache_key(...)`: Cache key based on tool, args, and access scope.
- `route_query_to_tool(query, hint)`: Rule-based router used by `/api/v1/query` and fallback paths.
- `natural_language_query(...)`: Executes keyword-routed query and formats response.
- `intelligent_query(...)`: Delegates to `IntelligentQueryHandler.process_query(...)`.

---

## Intelligent Query Flow (`mcp/llm_router.py`)
- `IntelligentQueryHandler.process_query(...)`: Main orchestration for `/api/v1/ask`.
- `needs_clarification(...)` (from `mcp/router/routing_rules.py`): Detects vague/incomplete prompts and asks user for clarity.
- `llm_route_query(...)`: Uses Claude/OpenAI to map query to `(tool, arguments)`.
- `repair_route(...)` (from `mcp/router/routing_rules.py`): Heuristic repair for sparse/ambiguous routing and conversational follow-ups.
- `llm_format_response(...)`: LLM-based response formatting.
- `fallback_route_query(...)`: Keyword fallback via `route_query_to_tool(...)`.
- `fallback_format_response(...)`: Deterministic formatting fallback.
- `build_output_payload(...)` (from `mcp/utils/output_layer.py`): final output shaping (`text`/`json`/`tree`) and optional download payload.

Conversation memory:
- Stored in-memory per `session_id` using deque.
- Last messages are included to resolve follow-ups like "there", "who are the constables?", etc.
- Additional structured session state stores last tool/arguments/results and last list user IDs for robust follow-up references (e.g., "about 2").

---

## Tool Execution Layer (`mcp/handlers/tool_handler.py`)
- `ToolHandler.initialize()`: Instantiates all tool classes with DB instance.
- `ToolHandler.execute(tool_name, arguments, context)`: Validates and runs selected tool.
- `_validate_arguments(...)`: Basic schema/type/enum checks.
- `_format_error(...)`: Uniform error shape for API responses.

Registered tool classes include:
- `SearchPersonnelTool`, `SearchUnitTool`, `CheckResponsibleUserTool`
- `QueryPersonnelByUnitTool`, `QueryPersonnelByRankTool`
- `GetUnitHierarchyTool`, `ListUnitsInDistrictTool`, `ListDistrictsTool`
- `CountVacanciesByUnitRankTool`, `GetPersonnelDistributionTool`
- `QueryRecentTransfersTool`, `GetUnitCommandHistoryTool`
- `FindMissingVillageMappingsTool`, `GetVillageCoverageTool`

---

## Base Tool Contract (`mcp/tools/base_tool.py`)
All tools inherit `BaseTool` and implement:
- `execute(arguments, context)`: Business logic.
- `get_input_schema()`: JSON schema for inputs.

Shared helpers:
- `apply_scope_filter(...)`: RBAC filtering by user scope.
- `get_pagination_params(...)`: Page/page_size normalization.
- `format_success_response(...)`: Standard success payload.
- `format_error_response(...)`: Standard error payload.
- `validate_required_params(...)` and `validate_at_least_one(...)`.

---

## Tool Modules and Responsibilities
- `mcp/tools/search_tools.py`
  - `SearchPersonnelTool`: Find personnel by `name`/`user_id`/`badge_no`/`mobile`/`email`.
  - `SearchUnitTool`: Find unit by name/reference/city/district.
  - `CheckResponsibleUserTool`: Check whether a person is responsible officer of a unit.
- `mcp/tools/personnel_tools.py`
  - `QueryPersonnelByUnitTool`: Personnel list for a unit.
  - `QueryPersonnelByRankTool`: Personnel list by rank with optional district filter; includes tolerant rank resolution (aliases, short codes, partial match).
- `mcp/tools/unit_tools.py`
  - `GetUnitHierarchyTool`: Unit parent-child structure.
  - `ListUnitsInDistrictTool`: Units under a district.
  - `ListDistrictsTool`: District catalog.
- `mcp/tools/vacancy_tools.py`
  - `CountVacanciesByUnitRankTool`: Vacancy/strength-oriented reporting.
  - `GetPersonnelDistributionTool`: Count/distribution views (`rank`/`unit_type`/`district`).
- `mcp/tools/transfer_tools.py`
  - `QueryRecentTransfersTool`: Transfer movements over recent time windows.
  - `GetUnitCommandHistoryTool`: Responsible officer history by unit.
- `mcp/tools/village_mapping_tools.py`
  - `FindMissingVillageMappingsTool`: Coverage gaps.
  - `GetVillageCoverageTool`: Village mapping/coverage report.

---

## Frontend Functions (`chatbot.html`)
Main interaction functions:
- `sendMessage()`: Sends query to `/api/v1/ask` and renders response.
- `createMessage(content, isUser)`: Builds message bubbles.
- `formatResponse(text)`: Lightweight markdown-like rendering.
- `showTypingIndicator()` / `removeTypingIndicator()`: Request progress UX.
- `setConnectionStatus(online, label)`: Header connection state.
- `startPersonnelSearch()`: Guided helper prompt for personnel search inputs.
- `triggerDownload(payload)`: Downloads the last formatted output when enabled.

Session handling:
- Persists `chatSessionId` in `localStorage`.
- Includes `session_id` in each request so backend can use conversation context.
- UI no longer includes a Settings panel; API URL is derived from current host as `http(s)://<host>:8090`.

---

## Configuration
Primary settings files:
- `mcp/core/config.py`: Runtime settings (`MONGODB_URI`, `MCP_HOST`, `MCP_PORT`, `JWT_*`, `REDIS_URL`, CORS).
- `mcp/config.py`: MCP-specific settings (`MCP_MAX_RESULTS`, page size, allowed collections).

Dependencies are listed in `requirements.txt` (FastAPI, Uvicorn, Motor/PyMongo, Redis, Pydantic, JWT libs, HTTPX).

---

## Typical Request Lifecycle (`/api/v1/ask`)
1. Frontend sends `{ query, session_id, output_format, allow_download }`.
2. Backend derives `UserContext` from JWT or defaults to state scope.
3. `IntelligentQueryHandler` checks if clarification is needed for vague prompts.
4. If clear enough, query is routed via LLM or fallback router.
5. `_repair_route(...)` enriches missing context and follow-up intent.
6. `ToolHandler.execute(...)` runs selected tool with scope filtering.
7. Response text is formatted (LLM or deterministic fallback).
8. Output layer resolves requested format (`auto`/`text`/`json`/`tree`) and download intent.
9. Frontend renders formatted output, and auto-downloads when download content is returned.

---

## Recent Behavior Updates
- Removed Settings UI from `chatbot.html`.
- Added clarification fallback for vague prompts.
- Improved rank-search robustness (phrase extraction + alias/shortcode/partial resolution).
- Fixed hierarchy text symbols to plain ASCII bullets.
- Improved district hierarchy root detection for datasets where root nodes are not null-parent.

---

## Notes
- Conversation memory is in-process only. Restarting the server resets chat context.
- Redis caching and rate limiting are optional and active only when `REDIS_URL` is set.
- If no LLM API key is configured, `/api/v1/ask` still works using rule-based fallback.
