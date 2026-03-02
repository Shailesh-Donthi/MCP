# Technical Functional Specification (TFS)
## MCP Architecture - Police Personnel Query Assistant

## 1. Document Purpose

This TFS defines the technical architecture and functional flow of the MCP-based Police Personnel Query Assistant.  
It describes:

- System components and responsibilities
- Request lifecycle and routing behavior
- Tool execution and data access model
- Security, scope control, observability, and operational constraints

This specification is based on the current implementation in `server/mcp/*`.

## 2. System Scope

The MCP service provides natural-language querying over police personnel and organizational data, including:

- Personnel lookup
- Rank/district/unit-based listings
- Unit hierarchy and command history
- Vacancy and transfer analytics
- Village coverage/mapping checks
- Cross-collection master-data relationship queries

It supports two query paths:

- `POST /api/v1/query`: rule-first endpoint with optional LLM routing
- `POST /api/v1/ask`: intelligent orchestrator (`IntelligentQueryHandler`) with context-aware recovery and follow-up handling

## 3. High-Level Architecture

```text
+------------------------+         +--------------------------------------+
| Web Client (HTML/JS)   | <-----> | FastAPI MCP Server (server_http.py)  |
| chatbot.html/app.js    |  HTTP   |                                      |
+------------------------+         |  - Auth & Scope extraction           |
                                   |  - Rate limit / cache / audit logs   |
                                   |  - /query and /ask NL endpoints      |
                                   +------------------+-------------------+
                                                      |
                                                      v
                                   +--------------------------------------+
                                   | Query Orchestration Layer            |
                                   |                                      |
                                   | /query: route_query_to_tool + repair |
                                   | /ask: IntelligentQueryHandler        |
                                   |  - LLM route                         |
                                   |  - deterministic repair/recovery     |
                                   |  - context memory & follow-ups       |
                                   +------------------+-------------------+
                                                      |
                                                      v
                                   +--------------------------------------+
                                   | Tool Handler + MCP Tools             |
                                   | (tool_handler.py + tools/*.py)       |
                                   |  - schema validation                 |
                                   |  - scope-filtered Mongo queries      |
                                   +------------------+-------------------+
                                                      |
                                                      v
                                   +--------------------------------------+
                                   | MongoDB (primary datastore)          |
                                   | assignment/personnel/unit/district...|
                                   +--------------------------------------+
```

Supporting infrastructure:

- Redis (optional): rate limiting + cache
- OpenAI/Azure OpenAI-compatible API (optional): LLM routing/formatting
- In-process chat/session memory (current implementation)

## 4. Runtime Modules and Responsibilities

### 4.1 API Layer

Primary module: `server/mcp/server_http.py`

Responsibilities:

- Bootstraps FastAPI app and lifespan
- Connects MongoDB on startup (`connect_to_mongodb`)
- Initializes tool registry via `ToolHandler`
- Optional Redis connectivity
- Exposes:
  - Health/readiness: `/health`, `/ready`
  - Tool introspection and direct execution
  - Query endpoints (`/api/v1/query`, `/api/v1/ask`)
  - Chat thread endpoints (`/api/v1/chats*`)
  - SSE endpoint (`/api/v1/mcp/sse`)

### 4.2 Query Orchestration

Modules:

- `server/mcp/llm_router.py`
- `server/mcp/router/routing_rules.py`
- `server/mcp/router/extractors.py`
- `server/mcp/router/prompts.py`
- `server/mcp/router/llm_client.py`

Responsibilities:

- Parse user intent into `(tool, arguments)`
- Repair and normalize routes
- Apply follow-up context hints
- Recover from missing/ambiguous data with deterministic fallback logic
- Format output (deterministic or LLM-driven based on tool policy)

### 4.3 Tool Execution Layer

Modules:

- `server/mcp/handlers/tool_handler.py`
- `server/mcp/tools/*.py`

Responsibilities:

- Register all MCP tools (`TOOL_CLASSES`)
- Validate input schema
- Execute tool with proper scope context
- Return standardized success/error payload

### 4.4 Data and Scope Layer

Modules:

- `server/mcp/core/database.py`
- `server/mcp/query_builder/filters.py`
- `server/mcp/query_builder/*`

Responsibilities:

- Manage MongoDB connection lifecycle
- Apply scope-level visibility rules (`unit`, `district`, `state`)
- Construct safe query/aggregation pipelines

## 5. Functional Request Flows

## 5.1 Flow A: `POST /api/v1/query` (Rule-first)

1. Request received with `QueryRequest`.
2. Auth context extracted from JWT or anonymous fallback scope (`MCP_ANON_SCOPE_LEVEL`).
3. Clarification check via `needs_clarification`.
4. Routing:
   - If explicit tool hint: use it.
   - Else if LLM key exists: `llm_route_query`.
   - Else: `route_query_to_tool` heuristics.
5. Route repaired by `repair_route`.
6. Tool executed through `ToolHandler`.
7. Natural-language response generated by formatter.
8. Output payload enriched (`build_output_payload`) for text/json/tree/table/chart/download support.
9. Query history and chat thread context updated.

## 5.2 Flow B: `POST /api/v1/ask` (Intelligent Handler)

1. Request delegated to singleton `IntelligentQueryHandler`.
2. Query normalization and output preference resolution.
3. Context-aware shortcuts:
   - Pagination follow-ups (`next page`, `previous page`)
   - Pronoun follow-ups (`their details`, `mobile number?`)
   - Unit follow-ups (`details on ...`, `personnel there`)
4. Route generation via LLM or heuristic fallback.
5. Route repair + ordinal overrides + session-state injection.
6. Recovery routines handle:
   - Rank/designation mismatch
   - Missing command-history fallback
   - Role-of-unit disambiguation via unit personnel lookup
7. Tool execution.
8. Deterministic/LLM response formatting policy applied.
9. Session memory updated with last list context, selected entities, and person/unit anchors.

## 6. LLM Routing and Formatting Strategy

LLM provider path: OpenAI/Azure OpenAI-compatible only.

Module: `server/mcp/router/llm_client.py`

Key behaviors:

- API key detection via OpenAI/Azure environment variables.
- Base URL normalization supports OpenAI and Azure endpoints.
- Model resolution:
  - Primary from env (`OPENAI_MODEL` / Azure deployment vars)
  - Fallback `gpt-4o-mini`
- `llm_route_query`:
  - First pass with contextual prompt
  - Strict retry pass if invalid JSON route output
  - Final fallback to heuristic router if still invalid

Formatting mode:

- Certain tools are deterministic-formatted to avoid response drift.
- Others can use LLM formatting when enabled.
- Output layer can additionally render text/json/tree/table/chart payloads.

## 7. Tooling Architecture

Registered tools (15):

1. `search_personnel`
2. `search_unit`
3. `check_responsible_user`
4. `query_personnel_by_unit`
5. `query_personnel_by_rank`
6. `get_unit_hierarchy`
7. `list_units_in_district`
8. `list_districts`
9. `count_vacancies_by_unit_rank`
10. `get_personnel_distribution`
11. `query_recent_transfers`
12. `get_unit_command_history`
13. `find_missing_village_mappings`
14. `get_village_coverage`
15. `query_linked_master_data`

Design notes:

- Tools are independently schema-driven.
- ToolHandler enforces contract and centralized error normalization.
- Cross-tool recovery in orchestration layer compensates for ambiguous user language and partial DB population.

## 8. Data Architecture

Core collections:

- `personnel_master`
- `assignment_master`
- `unit_master`
- `district_master`
- `rank_master`
- `designation_master`
- `department_master`

Important relationship examples:

- `assignment_master.userId -> personnel_master._id`
- `assignment_master.unitId -> unit_master._id`
- `unit_master.districtId -> district_master._id`
- `personnel_master.rankId -> rank_master._id`

Master-data relationship tooling covers modules/prompts/notifications/roles/permissions/value sets and related collections.

## 9. Security and Access Control

Authentication:

- JWT Bearer token optional by endpoint usage.
- If absent, anonymous context is created with configurable scope level.

Authorization model:

- Scope-based filtering (`state`, `district`, `unit`) in `ScopeFilter`.
- Query filters are merged into base queries to enforce row-level visibility.

Controls:

- Rate limit (Redis-backed, if enabled)
- Structured error catalog mapping to stable MCP error codes
- Endpoint-level dependency injection for context and limit checks

## 10. State and Session Model

Two memory tracks are used:

- Query history (`_query_history`) for `/query` lightweight follow-up hints
- Intelligent session memory (`IntelligentQueryHandler`) for `/ask`:
  - last tool, arguments, list context, selected entity
  - last person and last unit anchors
  - pagination and ordinal resolution

Chat threads:

- In-process server-side chat thread store in current implementation
- Includes messages, session ID binding, and last assistant result snapshot

## 11. Observability and Operations

Built-in:

- `/health`, `/ready`
- `/metrics` (basic counters)
- Structured logging with request/session IDs
- Session/audit logging persisted in Mongo where configured

Startup behavior:

- Connect MongoDB
- Initialize tools
- Connect Redis if configured (degrades gracefully if unavailable)

## 12. Error Handling Model

Layers:

- Tool-level errors normalized via ToolHandler
- API-level exceptions mapped through MCP error catalog
- Distinct codes for auth/rate/tool/query/input/scope/internal classes

User-facing behavior:

- Friendly fallback responses for not-found/no-data scenarios
- Clarification prompts for ambiguous requests
- Deterministic fallback when LLM responses are invalid

## 13. Non-Functional Characteristics

Performance controls:

- Pagination defaults and max limits
- Bounded in-memory context caches
- Optional Redis cache

Reliability:

- Graceful degradation without Redis
- Graceful fallback from LLM route failures to rule-based routing

Security posture:

- JWT-based identity parsing
- Scope-constrained query execution
- CORS and secret/env-based configuration

## 16. Primary Source Files

- `server/mcp/server_http.py`
- `server/mcp/llm_router.py`
- `server/mcp/router/routing_rules.py`
- `server/mcp/router/extractors.py`
- `server/mcp/router/llm_client.py`
- `server/mcp/handlers/tool_handler.py`
- `server/mcp/tools/*.py`
- `server/mcp/query_builder/filters.py`
- `server/mcp/core/database.py`
- `server/mcp/utils/output_layer.py`

