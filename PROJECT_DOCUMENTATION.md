# Police Personnel Query Assistant - Technical Documentation

---

## Part 1: Technology & Model Overview

### 1.1 Core Stack

- Language: Python 3 (server-side), JavaScript (client-side)
- API framework: FastAPI
- ASGI server: Uvicorn
- Database: MongoDB
- Mongo drivers: `motor` (async), `pymongo`
- Cache/rate-limit support: Redis (`redis.asyncio`)
- HTTP client for LLM calls: `httpx`
- Config/validation: `pydantic`, `pydantic-settings`
- Auth/JWT libs: `python-jose`, `PyJWT`

Dependency source: [server/requirements.txt](server/requirements.txt)

### 1.2 Architecture

- Server entrypoint:
  - [mcp_server.py](mcp_server.py) (compatibility launcher)
  - [server/mcp_server.py](server/mcp_server.py)
- Main HTTP/SSE app:
  - [server/mcp/server_http.py](server/mcp/server_http.py)
- Query orchestration layer:
  - [server/mcp/llm_router.py](server/mcp/llm_router.py)
- Tool execution registry:
  - [server/mcp/handlers/tool_handler.py](server/mcp/handlers/tool_handler.py)

The system uses deterministic tool execution with optional LLM-based routing/formatting on top.

### 1.3 LLM Model & Provider

LLM integration code: [server/mcp/router/llm_client.py](server/mcp/router/llm_client.py)

- OpenAI / Azure OpenAI-compatible:
  - Chat endpoint style: `/chat/completions`
  - Default model fallback: `gpt-4o-mini`
  - OpenAI env options: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
  - Azure env options: `AZURE_AI_ENDPOINT` or `AZURE_OPENAI_ENDPOINT`, plus `AZURE_AI_API_KEY` or `AZURE_OPENAI_API_KEY`, and `AZURE_AI_MODEL` or `AZURE_OPENAI_DEPLOYMENT`
  - Auto-detects Azure-compatible base URLs and uses `api-key` header for Azure.

Behavior:
- LLM routing is used when a key is configured.
- If LLM fails or returns invalid routing JSON, the app falls back to heuristic routing.

### 1.4 Data Layer (Mongo Collections)

Configured whitelist: [server/mcp/config.py](server/mcp/config.py)

Primary collections used include:
- `personnel_master`
- `assignment_master`
- `unit_master`
- `district_master`
- `rank_master`
- `designation_master`
- `department_master`
- plus master-data collections (`modules_master`, `prompt_master`, `notification_master`, etc.)

### 1.5 Tooling (Query Capabilities)

Registered tools: [server/mcp/handlers/tool_handler.py](server/mcp/handlers/tool_handler.py)

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

### 1.6 Frontend

Client is plain HTML/CSS/JavaScript (no React/Vue build system in this repo):
- [client/chatbot.html](client/chatbot.html)
- [client/chatbot_app.js](client/chatbot_app.js)
- [client/chatbot_output.js](client/chatbot_output.js)

### 1.7 Runtime Configuration

Core app settings: [server/mcp/core/config.py](server/mcp/core/config.py)

Key runtime variables:
- DB: `MONGODB_URI`, `MONGODB_DB_NAME`
- Server: `MCP_HOST`, `MCP_PORT`
- CORS: `ALLOWED_ORIGINS`
- Redis: `REDIS_URL`
- Auth: `JWT_SECRET_KEY`, `JWT_ALGORITHM`
- Logging: `MCP_LOG_LEVEL`, `MCP_LOG_FILE`, `MCP_LOG_MAX_BYTES`, `MCP_LOG_BACKUP_COUNT`

### 1.8 Protocols & APIs

- REST endpoints for tools/query/chat
- SSE endpoint for MCP protocol streaming:
  - `/api/v1/mcp/sse`
- Health/metrics endpoints are also available in the FastAPI server.

---

## Part 2: Improved MCP Architecture - Relationship-Aware Query System

### 2.1 Problem Statement

**Current Issue:**
- Tools query only single collections (e.g., `personnel_master`)
- When asking "who is John Doe?", only basic personnel data is returned
- Missing related data from:
  - `assignment_master` (current/past assignments)
  - `unit_master` (unit details, hierarchy)
  - `district_master` (district information)
  - `rank_master` (rank details)
  - `designation_master` (designation details)
  - `department_master` (department info)
- Complex queries require multiple round trips to database
- No automatic enrichment of related entities
- Scope filtering (postCode-based data binding) not integrated into relationship queries

**Root Cause:**
- Tools execute simple `find()` queries instead of aggregation pipelines with `$lookup`
- No relationship-aware data access layer
- Each tool operates in isolation without understanding data relationships
- Existing `response_enricher.py` does batch lookups but not in single aggregation pipeline
- Scope filtering (JWT postCode) applied separately, not integrated into enriched queries

### 2.2 Solution Architecture

#### Core Principle: Relationship-Aware Data Access Layer

All queries should automatically enrich results with related data using MongoDB aggregation pipelines.

#### New Architecture Layers

```
+-------------------------------------------------------------+
|                    API Layer (FastAPI)                       |
|  - /api/v1/query, /api/v1/ask                               |
|  - Auth, rate limiting, request validation                   |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|              Query Orchestration Layer                       |
|  - LLM/rule-based routing                                   |
|  - Intent understanding                                     |
|  - Multi-tool composition (if needed)                       |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|                  Tool Execution Layer                        |
|  - Tool registry & validation                               |
|  - Tool-specific business logic                             |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|            Relationship-Aware Repository Layer               |
|  +------------------------------------------------------+  |
|  |  Entity Repositories (Personnel, Unit, Assignment)    |  |
|  |  - Understand entity relationships                    |  |
|  |  - Auto-enrich with related data                      |  |
|  +------------------------------------------------------+  |
|  +------------------------------------------------------+  |
|  |  Relationship Mapper                                  |  |
|  |  - Defines collection relationships                   |  |
|  |  - Generates $lookup pipelines                        |  |
|  +------------------------------------------------------+  |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|              MongoDB Aggregation Builder                     |
|  - Builds $lookup/$match/$project pipelines                |
|  - Handles scope filtering                                  |
|  - Optimizes query performance                              |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|                    MongoDB Database                          |
|  - personnel_master, assignment_master, unit_master, etc.  |
+-------------------------------------------------------------+
```

---

### 2.3 Key Components

#### 2.3.1 Relationship Mapper

**Purpose:** Central registry of all collection relationships and how to join them.

```python
RELATIONSHIP_MAP = {
    "personnel_master": {
        "assignments": {
            "collection": "assignment_master",
            "local_field": "_id",
            "foreign_field": "userId",
            "as": "assignments",
            "enrich": True,
            "type": "one-to-many"
        },
        "rank": {
            "collection": "rank_master",
            "local_field": "rankId",
            "foreign_field": "_id",
            "as": "rank",
            "enrich": True,
            "type": "one-to-one"
        },
        "designation": {
            "collection": "designation_master",
            "local_field": "designationId",
            "foreign_field": "_id",
            "as": "designation",
            "enrich": True,
            "type": "one-to-one"
        },
        "department": {
            "collection": "department_master",
            "local_field": "departmentId",
            "foreign_field": "_id",
            "as": "department",
            "enrich": False,
            "type": "one-to-one"
        }
    },
    "assignment_master": {
        "personnel": {
            "collection": "personnel_master",
            "local_field": "userId",
            "foreign_field": "_id",
            "as": "personnel",
            "enrich": True,
            "type": "many-to-one"
        },
        "unit": {
            "collection": "unit_master",
            "local_field": "unitId",
            "foreign_field": "_id",
            "as": "unit",
            "enrich": True,
            "type": "many-to-one"
        },
        "designation": {
            "collection": "designation_master",
            "local_field": "designationId",
            "foreign_field": "_id",
            "as": "designation",
            "enrich": True,
            "type": "many-to-one"
        }
    },
    "unit_master": {
        "district": {
            "collection": "district_master",
            "local_field": "districtId",
            "foreign_field": "_id",
            "as": "district",
            "enrich": True,
            "type": "many-to-one"
        },
        "parent_unit": {
            "collection": "unit_master",
            "local_field": "parentUnitId",
            "foreign_field": "_id",
            "as": "parentUnit",
            "enrich": False,
            "type": "many-to-one",
            "recursive": True
        },
        "unit_type": {
            "collection": "unit_type_master",
            "local_field": "unitTypeId",
            "foreign_field": "_id",
            "as": "unitType",
            "enrich": False,
            "type": "many-to-one"
        }
    },
    "district_master": {
        "mandals": {
            "collection": "mandal_master",
            "local_field": "_id",
            "foreign_field": "districtId",
            "as": "mandals",
            "enrich": False,
            "type": "one-to-many"
        }
    },
    "crpc_requests": {
        "unit": { ... },
        "district": { ... },
        "requested_by_personnel": { ... },
        "pipelines": {
            "collection": "crpc_request_pipelines",
            "local_field": "_id",
            "foreign_field": "crpcRequestId",
            "as": "pipelines",
            "enrich": True,
            "type": "one-to-many"
        }
    },
    "crpc_request_pipelines": {
        "crpc_request": { ... },
        "service": { ... },
        "operators": {
            "collection": "operators_list_master",
            "local_field": "operatorIds",
            "foreign_field": "_id",
            "as": "operators",
            "enrich": True,
            "type": "many-to-many"
        }
    }
}
```

#### 2.3.2 Aggregation Pipeline Builder

**Purpose:** Builds MongoDB aggregation pipelines with automatic relationship joins.

**Key Methods:**
- `build_enriched_query(base_collection, filters, enrichments, scope_filter)`
- `add_lookups(pipeline, collection, relationships_to_enrich)`
- `add_scope_filter(pipeline, scope_context)`
- `optimize_pipeline(pipeline)`

#### 2.3.3 Entity Repositories

**Purpose:** High-level repository classes that automatically enrich queries.

```python
class PersonnelRepository:
    async def find_by_name(self, name, scope_context, enrichments=None) -> List[Dict]:
        """Find personnel by name with automatic relationship enrichment."""

class AssignmentRepository:
    async def find_by_personnel_id(self, personnel_id, scope_context, include_inactive=False) -> List[Dict]:
        """Find all assignments for a personnel with unit/district enrichment."""

class UnitRepository:
    async def find_by_id(self, unit_id, scope_context, include_hierarchy=True) -> Optional[Dict]:
        """Find unit with district and optionally parent unit."""
```

#### 2.3.4 Enhanced Tool Implementation

**Before (Single Collection):**
```python
async def search_personnel(name, scope_context):
    result = await db["personnel_master"].find_one(
        {"name": {"$regex": name, "$options": "i"}}
    )
    return result
```

**After (Relationship-Aware):**
```python
async def search_personnel(name, scope_context,
                          include_assignments=True, include_rank=True):
    repo = PersonnelRepository(db, relationship_mapper, pipeline_builder)
    enrichments = []
    if include_assignments: enrichments.append("assignments")
    if include_rank: enrichments.append("rank")
    return await repo.find_by_name(name, scope_context, enrichments)
```

---

### 2.4 Implementation Phases

| Phase | Scope | Details |
|-------|-------|---------|
| 1 - Core Infrastructure | Relationship Mapper, Pipeline Builder, Base Repository | Define relationships, build $lookup pipelines, scope filter integration |
| 2 - Entity Repositories | PersonnelRepository, AssignmentRepository, UnitRepository | `find_by_name()`, `find_by_id()`, `find_current_assignment()`, etc. |
| 3 - Tool Refactoring | Refactor existing tools to use repositories | Maintain backward compatibility |
| 4 - Advanced Features | Nested relationships, caching, field projection | Performance monitoring |

---

### 2.5 Example Query Flow

**User Query:** "Who is John Doe?"

1. **Router** identifies intent -> `search_personnel` tool
2. **Tool Handler** validates input -> `name="John Doe"`
3. **Tool** calls `PersonnelRepository.find_by_name("John Doe", scope_context)`
4. **Repository** consults `RelationshipMapper` for personnel relationships
5. **Pipeline Builder** creates aggregation pipeline with $lookup for assignments, rank, designation, units, districts + scope filtering
6. **MongoDB** executes pipeline -> returns enriched document
7. **API** returns comprehensive response with all related data

---

### 2.6 Scope Filtering (JWT-Based Data Binding)

```python
class ScopeContext:
    def __init__(self, user_id=None, unit_id=None, district_id=None,
                 post_code=None, role_name=None, reports_to_post=None): ...

    @classmethod
    async def from_request(cls, request):
        """Create ScopeContext from FastAPI Request (extracts JWT)."""
        user_info = await get_user_info_from_request(request)
        return cls(user_id=user_info.get("_id"), ...)
```

Supports:
- `postCode` filtering for crpc_requests, crpc_request_pipelines, crpc_data_received
- `unitId` filtering for unit-scoped queries
- `districtId` filtering for district-scoped queries

---

### 2.7 Complex Query Patterns

| Pattern | Example | Approach |
|---------|---------|----------|
| Multi-Level Enrichment | "Who is the SHO of Central PS?" | unit -> assignment -> personnel -> rank |
| Reverse Lookup | "Find all units where John Doe was assigned" | personnel -> assignments -> units |
| Aggregation + Relationships | "CrPC requests per unit with details" | group by unitId, enrich with unit+district |
| Temporal Queries | "Personnel at Unit X during 2023" | match by date range + personnel enrichment |
| Hierarchical Queries | "Child units of Guntur DPO" | recursive parentUnitId lookup |

---

### 2.8 Performance & Migration

**Performance SLAs:**
- Simple enriched query (personnel + rank): < 200ms
- Medium enriched query (personnel + assignments + unit): < 500ms
- Complex enriched query (request + pipelines + services): < 1000ms

**Migration Path:**
1. Add new layer (non-breaking) alongside existing tools
2. Feature flag `ENABLE_RELATIONSHIP_ENRICHMENT`
3. Gradual migration, one tool at a time
4. Remove old code after full migration

**Risk Mitigation:**
- Query timeout (5s), complexity limits, indexes, caching
- Feature flag for rollback, backward compatibility
- Pipeline validation before execution, fallback to simple queries

**Success Metrics:**
- Query response time within SLA targets
- Error rate < 0.1%, pipeline success > 99.9%, test coverage > 80%
- Fewer follow-up queries, higher MCP endpoint adoption
