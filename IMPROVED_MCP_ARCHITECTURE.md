# Improved MCP Architecture - Relationship-Aware Query System

## Problem Statement

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

## Solution Architecture

### 1. Core Principle: Relationship-Aware Data Access Layer

All queries should automatically enrich results with related data using MongoDB aggregation pipelines.

---

## 2. New Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  - /api/v1/query, /api/v1/ask                               │
│  - Auth, rate limiting, request validation                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              Query Orchestration Layer                       │
│  - LLM/rule-based routing                                   │
│  - Intent understanding                                     │
│  - Multi-tool composition (if needed)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                  Tool Execution Layer                        │
│  - Tool registry & validation                               │
│  - Tool-specific business logic                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│            Relationship-Aware Repository Layer               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Entity Repositories (Personnel, Unit, Assignment)    │  │
│  │  - Understand entity relationships                    │  │
│  │  - Auto-enrich with related data                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Relationship Mapper                                  │  │
│  │  - Defines collection relationships                   │  │
│  │  - Generates $lookup pipelines                        │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│              MongoDB Aggregation Builder                     │
│  - Builds $lookup/$match/$project pipelines                │
│  - Handles scope filtering                                  │
│  - Optimizes query performance                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────────────┐
│                    MongoDB Database                          │
│  - personnel_master, assignment_master, unit_master, etc.  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### 3.1 Relationship Mapper (`server/mcp/repositories/relationship_mapper.py`)

**Purpose:** Central registry of all collection relationships and how to join them.

```python
# Conceptual structure
RELATIONSHIP_MAP = {
    "personnel_master": {
        "assignments": {
            "collection": "assignment_master",
            "local_field": "_id",
            "foreign_field": "userId",
            "as": "assignments",
            "enrich": True  # Always include in personnel queries
        },
        "rank": {
            "collection": "rank_master",
            "local_field": "rankId",
            "foreign_field": "_id",
            "as": "rank",
            "enrich": True
        },
        "designation": {
            "collection": "designation_master",
            "local_field": "designationId",
            "foreign_field": "_id",
            "as": "designation",
            "enrich": True
        },
        "department": {
            "collection": "department_master",
            "local_field": "departmentId",
            "foreign_field": "_id",
            "as": "department",
            "enrich": False  # Optional enrichment
        }
    },
    "assignment_master": {
        "personnel": {
            "collection": "personnel_master",
            "local_field": "userId",
            "foreign_field": "_id",
            "as": "personnel",
            "enrich": True
        },
        "unit": {
            "collection": "unit_master",
            "local_field": "unitId",
            "foreign_field": "_id",
            "as": "unit",
            "enrich": True
        },
        "designation": {
            "collection": "designation_master",
            "local_field": "designationId",
            "foreign_field": "_id",
            "as": "designation",
            "enrich": True
        }
    },
    "unit_master": {
        "district": {
            "collection": "district_master",
            "local_field": "districtId",
            "foreign_field": "_id",
            "as": "district",
            "enrich": True
        },
        "parent_unit": {
            "collection": "unit_master",
            "local_field": "parentUnitId",
            "foreign_field": "_id",
            "as": "parentUnit",
            "enrich": False  # Optional
        }
    }
}
```

### 3.2 Aggregation Pipeline Builder (`server/mcp/repositories/pipeline_builder.py`)

**Purpose:** Builds MongoDB aggregation pipelines with automatic relationship joins.

**Key Methods:**
- `build_enriched_query(base_collection, filters, enrichments, scope_filter)`
- `add_lookups(pipeline, collection, relationships_to_enrich)`
- `add_scope_filter(pipeline, scope_context)`
- `optimize_pipeline(pipeline)`

**Example Pipeline for Personnel Query:**
```python
[
    # Stage 1: Match base filters
    {"$match": {"name": {"$regex": "John Doe", "$options": "i"}}},
    
    # Stage 2: Lookup assignments
    {
        "$lookup": {
            "from": "assignment_master",
            "localField": "_id",
            "foreignField": "userId",
            "as": "assignments"
        }
    },
    
    # Stage 3: Lookup rank
    {
        "$lookup": {
            "from": "rank_master",
            "localField": "rankId",
            "foreignField": "_id",
            "as": "rank"
        }
    },
    
    # Stage 4: Unwind rank (single relationship)
    {"$unwind": {"path": "$rank", "preserveNullAndEmptyArrays": True}},
    
    # Stage 5: Lookup designation
    {
        "$lookup": {
            "from": "designation_master",
            "localField": "designationId",
            "foreignField": "_id",
            "as": "designation"
        }
    },
    {"$unwind": {"path": "$designation", "preserveNullAndEmptyArrays": True}},
    
    # Stage 6: Enrich assignments with unit details
    {
        "$lookup": {
            "from": "unit_master",
            "localField": "assignments.unitId",
            "foreignField": "_id",
            "as": "assignments_units"
        }
    },
    
    # Stage 7: Merge unit data into assignments
    {
        "$addFields": {
            "assignments": {
                "$map": {
                    "input": "$assignments",
                    "as": "assignment",
                    "in": {
                        "$mergeObjects": [
                            "$$assignment",
                            {
                                "unit": {
                                    "$arrayElemAt": [
                                        {
                                            "$filter": {
                                                "input": "$assignments_units",
                                                "cond": {"$eq": ["$$this._id", "$$assignment.unitId"]}
                                            }
                                        },
                                        0
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    },
    
    # Stage 8: Enrich unit's district
    {
        "$addFields": {
            "assignments": {
                "$map": {
                    "input": "$assignments",
                    "as": "assignment",
                    "in": {
                        "$mergeObjects": [
                            "$$assignment",
                            {
                                "unit": {
                                    "$mergeObjects": [
                                        "$$assignment.unit",
                                        {
                                            "district": {
                                                "$let": {
                                                    "vars": {
                                                        "districtId": "$$assignment.unit.districtId"
                                                    },
                                                    "in": {
                                                        "$arrayElemAt": [
                                                            {
                                                                "$filter": {
                                                                    "input": {"$lookup": {...}},
                                                                    "cond": {"$eq": ["$$this._id", "$$districtId"]}
                                                                }
                                                            },
                                                            0
                                                        ]
                                                    }
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    },
    
    # Stage 9: Project final structure
    {
        "$project": {
            "_id": 1,
            "name": 1,
            "employeeId": 1,
            "mobile": 1,
            "email": 1,
            "rank": 1,
            "designation": 1,
            "assignments": {
                "$map": {
                    "input": "$assignments",
                    "as": "assignment",
                    "in": {
                        "_id": "$$assignment._id",
                        "unitId": "$$assignment.unitId",
                        "unit": {
                            "name": "$$assignment.unit.name",
                            "code": "$$assignment.unit.code",
                            "district": {
                                "name": "$$assignment.unit.district.name",
                                "code": "$$assignment.unit.district.code"
                            }
                        },
                        "designation": "$$assignment.designation",
                        "startDate": "$$assignment.startDate",
                        "endDate": "$$assignment.endDate",
                        "isActive": "$$assignment.isActive"
                    }
                }
            },
            "currentAssignment": {
                "$arrayElemAt": [
                    {
                        "$filter": {
                            "input": "$assignments",
                            "cond": {"$eq": ["$$this.isActive", True]}
                        }
                    },
                    0
                ]
            }
        }
    }
]
```

### 3.3 Entity Repositories (`server/mcp/repositories/entity_repositories.py`)

**Purpose:** High-level repository classes that automatically enrich queries.

**PersonnelRepository:**
```python
class PersonnelRepository:
    def __init__(self, db, relationship_mapper, pipeline_builder):
        self.db = db
        self.mapper = relationship_mapper
        self.builder = pipeline_builder
    
    async def find_by_name(self, name: str, scope_context: ScopeContext, 
                          enrichments: Optional[List[str]] = None) -> List[Dict]:
        """
        Find personnel by name with automatic relationship enrichment.
        
        Args:
            name: Name to search
            scope_context: Scope filtering context
            enrichments: Optional list of specific relationships to include
                        If None, includes all 'enrich: True' relationships
        
        Returns:
            List of personnel documents with enriched related data
        """
        # Get relationships to enrich
        if enrichments is None:
            relationships = self.mapper.get_auto_enrich_relationships("personnel_master")
        else:
            relationships = {k: v for k, v in 
                           self.mapper.get_relationships("personnel_master").items() 
                           if k in enrichments}
        
        # Build aggregation pipeline
        pipeline = self.builder.build_enriched_query(
            base_collection="personnel_master",
            filters={"name": {"$regex": name, "$options": "i"}},
            relationships=relationships,
            scope_filter=scope_context
        )
        
        # Execute
        cursor = self.db["personnel_master"].aggregate(pipeline)
        return await cursor.to_list(length=None)
    
    async def find_by_id(self, personnel_id: str, scope_context: ScopeContext,
                        enrichments: Optional[List[str]] = None) -> Optional[Dict]:
        """Find single personnel by ID with enrichment."""
        results = await self.find_by_name(
            name="",  # Will use ID filter instead
            scope_context=scope_context,
            enrichments=enrichments
        )
        # Filter by ID in pipeline instead
        return results[0] if results else None
```

**AssignmentRepository:**
```python
class AssignmentRepository:
    async def find_by_personnel_id(self, personnel_id: str, 
                                   scope_context: ScopeContext,
                                   include_inactive: bool = False) -> List[Dict]:
        """Find all assignments for a personnel with unit/district enrichment."""
        # Similar pattern with automatic enrichment
        pass
    
    async def find_current_assignment(self, personnel_id: str,
                                     scope_context: ScopeContext) -> Optional[Dict]:
        """Find current active assignment with full enrichment."""
        pass
```

**UnitRepository:**
```python
class UnitRepository:
    async def find_by_id(self, unit_id: str, scope_context: ScopeContext,
                        include_hierarchy: bool = True) -> Optional[Dict]:
        """Find unit with district and optionally parent unit."""
        pass
    
    async def find_personnel_in_unit(self, unit_id: str, 
                                    scope_context: ScopeContext) -> List[Dict]:
        """Find all personnel in unit with their rank/designation."""
        pass
```

### 3.4 Enhanced Tool Implementation

**Before (Current - Single Collection):**
```python
async def search_personnel(name: str, scope_context: ScopeContext):
    # Simple query - only personnel_master
    result = await db["personnel_master"].find_one(
        {"name": {"$regex": name, "$options": "i"}}
    )
    return result
```

**After (Relationship-Aware):**
```python
async def search_personnel(name: str, scope_context: ScopeContext,
                          include_assignments: bool = True,
                          include_rank: bool = True,
                          include_designation: bool = True):
    """
    Search personnel with automatic enrichment of related data.
    
    Returns enriched personnel data including:
    - Current and past assignments
    - Unit details (with district)
    - Rank information
    - Designation information
    """
    repo = PersonnelRepository(db, relationship_mapper, pipeline_builder)
    
    enrichments = []
    if include_assignments:
        enrichments.append("assignments")
    if include_rank:
        enrichments.append("rank")
    if include_designation:
        enrichments.append("designation")
    
    results = await repo.find_by_name(
        name=name,
        scope_context=scope_context,
        enrichments=enrichments if enrichments else None
    )
    
    return results
```

---

## 4. Implementation Strategy

### Phase 1: Core Infrastructure
1. **Create Relationship Mapper**
   - Define all collection relationships
   - Support for one-to-one, one-to-many, many-to-many
   - Configuration-driven (can be extended via config file)

2. **Build Pipeline Builder**
   - Core aggregation pipeline construction
   - Automatic $lookup generation
   - Scope filter integration
   - Pipeline optimization (remove unnecessary stages)

3. **Create Base Repository Class**
   - Common query patterns
   - Error handling
   - Logging

### Phase 2: Entity Repositories
1. **PersonnelRepository**
   - `find_by_name()` - with enrichment
   - `find_by_id()` - with enrichment
   - `find_by_employee_id()` - with enrichment
   - `find_by_mobile()` - with enrichment

2. **AssignmentRepository**
   - `find_by_personnel_id()` - with unit/district enrichment
   - `find_current_assignment()` - with full context
   - `find_by_unit_id()` - with personnel enrichment

3. **UnitRepository**
   - `find_by_id()` - with district/parent
   - `find_personnel_in_unit()` - with personnel enrichment
   - `find_units_in_district()` - with district context

### Phase 3: Tool Refactoring
1. **Refactor existing tools** to use repositories
2. **Update tool schemas** to support enrichment options
3. **Maintain backward compatibility** where possible

### Phase 4: Advanced Features
1. **Nested relationship support** (e.g., unit -> district -> state)
2. **Selective field projection** (only return needed fields)
3. **Caching layer** for frequently accessed relationships
4. **Query performance monitoring**

---

## 5. File Structure

**Integration with Existing Codebase:**

```
app/
├── api/
│   └── v1/
│       ├── repositories/
│       │   ├── base_repository.py                    # ✅ EXISTS - Extend this
│       │   ├── mcp/                                  # 🆕 NEW FOLDER
│       │   │   ├── __init__.py
│       │   │   ├── relationship_mapper.py           # Relationship definitions
│       │   │   ├── pipeline_builder.py              # Aggregation pipeline builder
│       │   │   ├── scope_context.py                  # JWT-based scope filtering
│       │   │   ├── enriched_base_repository.py      # Extends BaseRepository
│       │   │   └── entity_repositories.py            # Personnel, Unit, Assignment repos
│       │   └── ... (existing repos)
│       ├── utils/
│       │   ├── response_enricher.py                 # ✅ EXISTS - Can be replaced gradually
│       │   └── ... (existing utils)
│       └── routers/
│           └── mcp/                                 # 🆕 NEW - MCP API endpoints
│               ├── __init__.py
│               ├── query_router.py                  # /api/v1/mcp/query
│               └── ask_router.py                     # /api/v1/mcp/ask
└── mcp/                                              # 🆕 NEW - MCP Server Implementation
    ├── __init__.py
    ├── server.py                                     # MCP server entry point
├── tools/
    │   ├── __init__.py
    │   ├── search_personnel.py                      # Uses PersonnelRepository
    │   ├── search_unit.py                           # Uses UnitRepository
    │   ├── search_assignment.py                     # Uses AssignmentRepository
    │   ├── search_crpc_request.py                   # Uses CrpcRequestRepository (enriched)
    │   └── complex_query.py                         # Multi-entity complex queries
├── handlers/
    │   ├── __init__.py
    │   ├── tool_handler.py                          # Tool execution & routing
    │   └── query_orchestrator.py                     # LLM-based query routing
    └── schemas/
        ├── __init__.py
        ├── query_schemas.py                          # Pydantic models for queries
        └── response_schemas.py                       # Pydantic models for enriched responses
```

**Key Integration Points:**
1. **Extends `BaseRepository`** - Reuses existing patterns (`isDelete`, `isActive` handling)
2. **Integrates with JWT** - Uses existing `get_user_info_from_request()` for scope filtering
3. **Replaces `response_enricher.py` gradually** - Can coexist during migration
4. **Follows existing folder structure** - Matches `app/api/v1/repositories/` pattern

---

## 6. Example: Complete Query Flow

**User Query:** "Who is John Doe?"

**Flow:**
1. **Router** identifies intent → `search_personnel` tool
2. **Tool Handler** validates input → `name="John Doe"`
3. **Tool** calls `PersonnelRepository.find_by_name("John Doe", scope_context)`
4. **Repository** consults `RelationshipMapper` for personnel relationships
5. **Pipeline Builder** creates aggregation pipeline with:
   - Base match on name
   - $lookup for assignments
   - $lookup for rank
   - $lookup for designation
   - $lookup for units (nested in assignments)
   - $lookup for districts (nested in units)
   - Scope filtering
6. **MongoDB** executes pipeline → returns enriched document
7. **Tool** formats response with all related data
8. **API** returns comprehensive response

**Response Structure:**
```json
{
  "_id": "personnel_123",
  "name": "John Doe",
  "employeeId": "EMP001",
  "mobile": "9876543210",
  "email": "john.doe@police.gov",
  "rank": {
    "_id": "rank_456",
    "name": "Inspector",
    "code": "INS"
  },
  "designation": {
    "_id": "desig_789",
    "name": "Station House Officer",
    "code": "SHO"
  },
  "assignments": [
    {
      "_id": "assign_001",
      "unitId": "unit_123",
      "unit": {
        "name": "Central Police Station",
        "code": "CPS001",
        "district": {
          "name": "Downtown District",
          "code": "DT001"
        }
      },
      "designation": {
        "name": "Station House Officer",
        "code": "SHO"
      },
      "startDate": "2023-01-15",
      "endDate": null,
      "isActive": true
    }
  ],
  "currentAssignment": {
    // Current active assignment details
  }
}
```

---

## 7. Performance Considerations

### 7.1 Pipeline Optimization
- **Early filtering:** Apply $match stages as early as possible
- **Selective enrichment:** Only enrich relationships that are requested
- **Field projection:** Use $project to limit returned fields
- **Index usage:** Ensure indexes on foreign key fields (`userId`, `unitId`, etc.)

### 7.2 Caching Strategy
- Cache master data (ranks, designations, districts) - rarely changes
- Cache unit hierarchies - changes infrequently
- Invalidate cache on updates

### 7.3 Pagination
- Support pagination in aggregation pipelines
- Use `$facet` for count + data in single query
- Limit default enrichment depth for list queries

---

## 8. Migration Path

### Step 1: Add New Layer (Non-Breaking)
- Implement repositories alongside existing tools
- Tools can gradually migrate

### Step 2: Feature Flag
- Add `ENABLE_RELATIONSHIP_ENRICHMENT` flag
- Tools check flag to use old vs new approach

### Step 3: Gradual Migration
- Migrate one tool at a time
- Test thoroughly
- Monitor performance

### Step 4: Remove Old Code
- Once all tools migrated, remove old query code

---

## 9. Benefits

1. **Comprehensive Data:** Queries return complete context automatically
2. **Consistency:** All tools use same enrichment logic
3. **Maintainability:** Relationship definitions in one place
4. **Performance:** Optimized aggregation pipelines
5. **Flexibility:** Easy to add new relationships
6. **Type Safety:** Can add Pydantic models for enriched responses

---

## 10. Next Steps

1. **Review and approve** this architecture
2. **Implement Phase 1** (Relationship Mapper + Pipeline Builder)
3. **Create Pydantic models** for enriched response structures
4. **Implement PersonnelRepository** as proof of concept
5. **Refactor `search_personnel` tool** to use repository
6. **Test with real queries**
7. **Iterate and improve**

---

## 11. Complete Collection Relationship Map

Based on codebase analysis, here are ALL master collections and their relationships:

### Master Collections Identified:
1. `personnel_master` - Personnel records
2. `assignment_master` - Personnel assignments to units
3. `unit_master` - Police units/hierarchy
4. `district_master` - Districts
5. `rank_master` - Ranks (Inspector, DSP, etc.)
6. `designation_master` - Designations (SHO, CI, etc.)
7. `department_master` - Departments
8. `unit_type_master` - Unit types
9. `mandal_master` - Mandals (sub-districts)
10. `crpc_requests` - CrPC requests (with relationships)
11. `crpc_request_pipelines` - Request pipelines
12. `service_master` - Services (CDR, IPDR, etc.)
13. `operators_list_master` - Operators (Telecom, Bank, etc.)
14. `operator_service_mapping_master` - Operator-service mappings

### Complete Relationship Map:

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
        },
        "unit": {
            "collection": "unit_master",
            "local_field": "unitId",
            "foreign_field": "_id",
            "as": "unit",
            "enrich": False,  # Usually get from assignments
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
        },
        "department": {
            "collection": "department_master",
            "local_field": "departmentId",
            "foreign_field": "_id",
            "as": "department",
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
        "unit": {
            "collection": "unit_master",
            "local_field": "unitId",
            "foreign_field": "_id",
            "as": "unit",
            "enrich": True,
            "type": "many-to-one"
        },
        "district": {
            "collection": "district_master",
            "local_field": "districtId",
            "foreign_field": "_id",
            "as": "district",
            "enrich": True,
            "type": "many-to-one"
        },
        "requested_by_personnel": {
            "collection": "personnel_master",
            "local_field": "requestedBy",
            "foreign_field": "_id",
            "as": "requestedByPersonnel",
            "enrich": True,
            "type": "many-to-one"
        },
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
        "crpc_request": {
            "collection": "crpc_requests",
            "local_field": "crpcRequestId",
            "foreign_field": "_id",
            "as": "crpcRequest",
            "enrich": True,
            "type": "many-to-one"
        },
        "service": {
            "collection": "service_master",
            "local_field": "serviceId",
            "foreign_field": "_id",
            "as": "service",
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
        "district": {
            "collection": "district_master",
            "local_field": "districtId",
            "foreign_field": "_id",
            "as": "district",
            "enrich": True,
            "type": "many-to-one"
        },
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

## 12. Scope Filtering (JWT-Based Data Binding)

**Current Implementation:**
- Uses `postCode` from JWT token
- Filters by `createdByPostCode` field in collections
- Applied in services like `crpc_requests_service.py`, `crpc_request_pipelines_service.py`

**Integration with Relationship-Aware Queries:**

```python
class ScopeContext:
    """Context for scope filtering based on JWT token."""
    def __init__(
        self,
        user_id: Optional[PyObjectId] = None,
        unit_id: Optional[PyObjectId] = None,
        district_id: Optional[PyObjectId] = None,
        post_code: Optional[str] = None,
        role_name: Optional[str] = None,
        reports_to_post: Optional[Dict] = None
    ):
        self.user_id = user_id
        self.unit_id = unit_id
        self.district_id = district_id
        self.post_code = post_code
        self.role_name = role_name
        self.reports_to_post = reports_to_post
    
    @classmethod
    async def from_request(cls, request: Request) -> "ScopeContext":
        """Create ScopeContext from FastAPI Request (extracts JWT)."""
        from app.api.v1.utils.request_helpers import get_user_info_from_request
        user_info = await get_user_info_from_request(request)
        return cls(
            user_id=PyObjectId(user_info.get("_id")),
            unit_id=PyObjectId(user_info.get("unitId")) if user_info.get("unitId") else None,
            district_id=PyObjectId(user_info.get("districtId")) if user_info.get("districtId") else None,
            post_code=user_info.get("postCode"),
            role_name=user_info.get("roleName"),
            reports_to_post=user_info.get("reportsToPost")
        )
```

**Scope Filter Application in Pipeline:**

```python
def add_scope_filter(
    pipeline: List[Dict],
    scope_context: ScopeContext,
    collection_name: str
) -> List[Dict]:
    """
    Add scope filtering to pipeline based on collection type.
    
    Collections with postCode filtering:
    - crpc_requests: createdByPostCode
    - crpc_request_pipelines: createdByPostCode
    - crpc_data_received: createdByPostCode
    
    Collections with unitId filtering:
    - crpc_requests: unitId
    - crpc_request_pipelines: unitId (optional)
    
    Collections with districtId filtering:
    - crpc_requests: districtId
    - crpc_request_pipelines: districtId (required)
    """
    match_stage = {}
    
    # Apply postCode filter for data binding
    if scope_context.post_code and collection_name in [
        "crpc_requests",
        "crpc_request_pipelines",
        "crpc_data_received"
    ]:
        match_stage["createdByPostCode"] = scope_context.post_code
    
    # Apply unitId filter
    if scope_context.unit_id and collection_name in [
        "crpc_requests",
        "crpc_request_pipelines"
    ]:
        match_stage["unitId"] = scope_context.unit_id
    
    # Apply districtId filter
    if scope_context.district_id and collection_name in [
        "crpc_requests",
        "crpc_request_pipelines"
    ]:
        match_stage["districtId"] = scope_context.district_id
    
    if match_stage:
        # Add to existing $match or create new one
        if pipeline and "$match" in pipeline[0]:
            pipeline[0]["$match"].update(match_stage)
        else:
            pipeline.insert(0, {"$match": match_stage})
    
    return pipeline
```

## 13. Complex Query Examples

### Example 1: "Find all personnel in Central Police Station with their current assignments"

```python
# Query: Personnel in unit "Central Police Station" with active assignments
pipeline = [
    # Stage 1: Find unit by name
    {
        "$match": {
            "name": {"$regex": "Central Police Station", "$options": "i"},
            "isDelete": False
        }
    },
    # Stage 2: Lookup assignments in this unit
    {
        "$lookup": {
            "from": "assignment_master",
            "let": {"unitId": "$_id"},
            "pipeline": [
                {
                    "$match": {
                        "$expr": {"$eq": ["$unitId", "$$unitId"]},
                        "isActive": True,
                        "isDelete": False
                    }
                }
            ],
            "as": "assignments"
        }
    },
    # Stage 3: Lookup personnel for each assignment
    {
        "$unwind": "$assignments"
    },
    {
        "$lookup": {
            "from": "personnel_master",
            "localField": "assignments.userId",
            "foreignField": "_id",
            "as": "personnel"
        }
    },
    {
        "$unwind": "$personnel"
    },
    # Stage 4: Enrich personnel with rank
    {
        "$lookup": {
            "from": "rank_master",
            "localField": "personnel.rankId",
            "foreignField": "_id",
            "as": "rank"
        }
    },
    {
        "$unwind": {"path": "$rank", "preserveNullAndEmptyArrays": True}
    },
    # Stage 5: Project final structure
    {
        "$project": {
            "unit": {
                "_id": "$_id",
                "name": "$name",
                "code": "$code"
            },
            "personnel": {
                "_id": "$personnel._id",
                "name": "$personnel.name",
                "employeeId": "$personnel.employeeId",
                "rank": {
                    "name": "$rank.name",
                    "code": "$rank.code"
                }
            },
            "assignment": {
                "startDate": "$assignments.startDate",
                "endDate": "$assignments.endDate",
                "postCode": "$assignments.postCode"
            }
        }
    }
]
```

### Example 2: "Find all CrPC requests created by John Doe with their pipelines and services"

```python
# Multi-step query with nested relationships
pipeline = [
    # Stage 1: Find personnel by name
    {
        "$match": {
            "name": {"$regex": "John Doe", "$options": "i"},
            "isDelete": False
        }
    },
    # Stage 2: Lookup CrPC requests created by this personnel
    {
        "$lookup": {
            "from": "crpc_requests",
            "let": {"personnelId": "$_id"},
            "pipeline": [
                {
                    "$match": {
                        "$expr": {"$eq": ["$requestedBy", "$$personnelId"]},
                        "isDelete": False
                    }
                }
            ],
            "as": "crpcRequests"
        }
    },
    # Stage 3: Unwind requests
    {
        "$unwind": "$crpcRequests"
    },
    # Stage 4: Lookup pipelines for each request
    {
        "$lookup": {
            "from": "crpc_request_pipelines",
            "localField": "crpcRequests._id",
            "foreignField": "crpcRequestId",
            "as": "pipelines"
        }
    },
    # Stage 5: Enrich pipelines with services
    {
        "$unwind": "$pipelines"
    },
    {
        "$lookup": {
            "from": "service_master",
            "localField": "pipelines.serviceId",
            "foreignField": "_id",
            "as": "service"
        }
    },
    {
        "$unwind": {"path": "$service", "preserveNullAndEmptyArrays": True}
    },
    # Stage 6: Group back by request
    {
        "$group": {
            "_id": "$crpcRequests._id",
            "request": {"$first": "$crpcRequests"},
            "pipelines": {
                "$push": {
                    "$mergeObjects": [
                        "$pipelines",
                        {"service": "$service"}
                    ]
                }
            }
        }
    },
    # Stage 7: Final projection
    {
        "$project": {
            "request": 1,
            "pipelines": 1
        }
    }
]
```

## 14. Implementation Phases (Detailed)

### Phase 1: Core Infrastructure (Week 1-2)

**1.1 Relationship Mapper** (`app/api/v1/repositories/mcp/relationship_mapper.py`)
- [ ] Define complete RELATIONSHIP_MAP with all collections
- [ ] Support for one-to-one, one-to-many, many-to-many relationships
- [ ] Support for recursive relationships (unit → parent_unit)
- [ ] Configuration validation
- [ ] Unit tests for relationship definitions

**1.2 Scope Context** (`app/api/v1/repositories/mcp/scope_context.py`)
- [ ] Create ScopeContext class
- [ ] Integrate with existing `get_user_info_from_request()`
- [ ] Support for postCode, unitId, districtId filtering
- [ ] Unit tests for scope extraction

**1.3 Pipeline Builder** (`app/api/v1/repositories/mcp/pipeline_builder.py`)
- [ ] Core aggregation pipeline construction
- [ ] Automatic $lookup generation from relationship map
- [ ] Nested relationship support (personnel → assignment → unit → district)
- [ ] Scope filter integration
- [ ] Pipeline optimization (early $match, selective projection)
- [ ] Support for pagination ($facet for count + data)
- [ ] Unit tests for pipeline generation

**1.4 Enriched Base Repository** (`app/api/v1/repositories/mcp/enriched_base_repository.py`)
- [ ] Extends existing `BaseRepository`
- [ ] Adds `find_enriched()` method
- [ ] Integrates with PipelineBuilder
- [ ] Maintains compatibility with existing methods
- [ ] Unit tests

### Phase 2: Entity Repositories (Week 3-4)

**2.1 PersonnelRepository** (`app/api/v1/repositories/mcp/entity_repositories.py`)
- [ ] `find_by_name()` - with full enrichment
- [ ] `find_by_id()` - with full enrichment
- [ ] `find_by_employee_id()` - with enrichment
- [ ] `find_by_mobile()` - with enrichment
- [ ] `find_by_email()` - with enrichment
- [ ] `find_in_unit()` - personnel in a unit with assignments
- [ ] All methods support scope filtering
- [ ] Integration tests

**2.2 AssignmentRepository**
- [ ] `find_by_personnel_id()` - with unit/district enrichment
- [ ] `find_current_assignment()` - active assignment with full context
- [ ] `find_by_unit_id()` - all assignments in unit with personnel
- [ ] `find_by_post_code()` - assignments by postCode
- [ ] Integration tests

**2.3 UnitRepository**
- [ ] `find_by_id()` - with district/parent unit
- [ ] `find_by_name()` - with enrichment
- [ ] `find_personnel_in_unit()` - all personnel with their assignments
- [ ] `find_units_in_district()` - with district context
- [ ] `find_child_units()` - hierarchical queries
- [ ] Integration tests

**2.4 CrpcRequestRepository** (extends existing)
- [ ] `find_enriched_by_id()` - with pipelines, services, operators
- [ ] `find_enriched_by_creator()` - requests by personnel with full context
- [ ] `find_enriched_by_unit()` - requests in unit with all relationships
- [ ] Integration tests

### Phase 3: MCP Tools (Week 5-6)

**3.1 Basic Search Tools**
- [ ] `search_personnel` - Uses PersonnelRepository
- [ ] `search_unit` - Uses UnitRepository
- [ ] `search_assignment` - Uses AssignmentRepository
- [ ] `search_crpc_request` - Uses enriched CrpcRequestRepository

**3.2 Complex Query Tools**
- [ ] `complex_query` - Multi-entity queries
- [ ] `find_personnel_with_assignments` - Specific use case
- [ ] `find_requests_by_personnel` - With pipelines and services
- [ ] `find_unit_hierarchy` - Recursive unit queries

**3.3 Query Orchestrator**
- [ ] LLM-based intent understanding
- [ ] Tool selection and routing
- [ ] Multi-tool composition for complex queries
- [ ] Response formatting

### Phase 4: API Integration (Week 7)

**4.1 MCP Router** (`app/api/v1/routers/mcp/`)
- [ ] `/api/v1/mcp/query` - Direct query endpoint
- [ ] `/api/v1/mcp/ask` - Natural language query endpoint
- [ ] JWT authentication integration
- [ ] Request validation
- [ ] Error handling

**4.2 Response Models**
- [ ] Pydantic models for enriched responses
- [ ] Type safety for all repository methods
- [ ] Serialization/deserialization

### Phase 5: Testing & Optimization (Week 8)

**5.1 Performance Testing**
- [ ] Query performance benchmarks
- [ ] Pipeline optimization validation
- [ ] Index verification
- [ ] Load testing

**5.2 Migration**
- [ ] Feature flag: `ENABLE_MCP_ENRICHMENT`
- [ ] Gradual migration of existing endpoints
- [ ] A/B testing
- [ ] Rollback plan

## 15. Performance SLAs & Acceptance Criteria

### Performance SLAs:
- **Simple enriched query** (personnel with rank): < 200ms
- **Medium enriched query** (personnel with assignments + unit): < 500ms
- **Complex enriched query** (request with pipelines + services + operators): < 1000ms
- **Pagination**: < 300ms per page

### Acceptance Criteria:

**AC1: Relationship Enrichment**
- ✅ All relationships defined in RELATIONSHIP_MAP are automatically enriched
- ✅ Nested relationships work (personnel → assignment → unit → district)
- ✅ One-to-many relationships return arrays
- ✅ One-to-one relationships return objects

**AC2: Scope Filtering**
- ✅ postCode filtering applied automatically from JWT
- ✅ unitId filtering applied when available
- ✅ districtId filtering applied when available
- ✅ Scope filters work with enriched queries

**AC3: Complex Queries**
- ✅ Multi-entity queries execute in single aggregation pipeline
- ✅ Nested lookups work correctly
- ✅ Recursive relationships supported (unit hierarchy)
- ✅ Many-to-many relationships handled (operators in pipelines)

**AC4: Backward Compatibility**
- ✅ Existing BaseRepository methods unchanged
- ✅ Existing endpoints continue to work
- ✅ Feature flag allows gradual migration
- ✅ Old response_enricher.py can coexist

## 16. PARKED ITEMS (Decisions Needed)

1. **Enrichment Depth:** 
   - **DECISION:** Maximum 4 levels of nesting (e.g., personnel → assignment → unit → district)
   - **RATIONALE:** Prevents performance degradation, most queries need 2-3 levels

2. **Default Behavior:**
   - **DECISION:** Opt-in enrichment (explicit `enrichments` parameter)
   - **RATIONALE:** Performance - only enrich what's needed

3. **Performance SLA:**
   - **DECISION:** See Section 15 above
   - **MONITORING:** Add query timing logs

4. **Caching:**
   - **DECISION:** Phase 2 - Add Redis caching for master data (ranks, designations, districts)
   - **RATIONALE:** Master data rarely changes, frequent lookups

5. **Backward Compatibility:**
   - **DECISION:** Maintain old behavior via feature flag for 3 months
   - **RATIONALE:** Gradual migration, risk mitigation

6. **Selective Fields:**
   - **DECISION:** Support field projection in Phase 2
   - **RATIONALE:** Performance optimization for large documents

7. **MCP Server Location:**
   - **DECISION:** Separate `mcp/` folder at app root, or integrate into `app/api/v1/`?
   - **RECOMMENDATION:** Start with `app/api/v1/repositories/mcp/` for repositories, separate `mcp/` for server

8. **Tool Registration:**
   - **DECISION:** Auto-discovery vs manual registration
   - **RECOMMENDATION:** Auto-discovery with decorator pattern

## 17. Complex Query Patterns & Use Cases

### Pattern 1: Multi-Level Enrichment
**Query:** "Who is the current SHO of Central Police Station and what's their rank?"

```python
# This requires:
# 1. Find unit by name
# 2. Find assignment with postCode="SHO" and isActive=True
# 3. Enrich with personnel
# 4. Enrich personnel with rank
# All in single pipeline
```

### Pattern 2: Reverse Lookup
**Query:** "Find all units where John Doe has been assigned"

```python
# This requires:
# 1. Find personnel by name
# 2. Lookup all assignments (including inactive)
# 3. Enrich assignments with units
# 4. Return unique units
```

### Pattern 3: Aggregation with Relationships
**Query:** "How many CrPC requests does each unit have, with unit details?"

```python
# This requires:
# 1. Group crpc_requests by unitId
# 2. Count requests per unit
# 3. Enrich with unit details
# 4. Enrich unit with district
```

### Pattern 4: Temporal Queries
**Query:** "Find all personnel who were assigned to Unit X between Jan 2023 and Dec 2023"

```python
# This requires:
# 1. Match assignments by unitId and date range
# 2. Enrich with personnel
# 3. Enrich with unit
# 4. Handle overlapping assignments
```

### Pattern 5: Hierarchical Queries
**Query:** "Find all child units of Guntur DPO with their personnel count"

```python
# This requires:
# 1. Find parent unit
# 2. Recursive lookup of child units (parentUnitId or parentUnitPath)
# 3. Count personnel per unit
# 4. Enrich with district
```

## 18. Implementation Checklist

### Pre-Implementation
- [ ] Review and approve architecture
- [ ] Set up feature flag: `ENABLE_MCP_ENRICHMENT`
- [ ] Create database indexes for foreign keys (if missing)
- [ ] Set up monitoring/logging for query performance

### Phase 1: Core Infrastructure
- [ ] Create `app/api/v1/repositories/mcp/` folder
- [ ] Implement `relationship_mapper.py`
- [ ] Implement `scope_context.py`
- [ ] Implement `pipeline_builder.py`
- [ ] Implement `enriched_base_repository.py`
- [ ] Write unit tests (target: 80% coverage)
- [ ] Integration test with MongoDB

### Phase 2: Entity Repositories
- [ ] Implement `PersonnelRepository`
- [ ] Implement `AssignmentRepository`
- [ ] Implement `UnitRepository`
- [ ] Extend `CrpcRequestRepository` with enrichment
- [ ] Write integration tests
- [ ] Performance benchmarks

### Phase 3: MCP Tools
- [ ] Create `mcp/tools/` folder
- [ ] Implement basic search tools
- [ ] Implement complex query tools
- [ ] Implement query orchestrator
- [ ] Write tool tests

### Phase 4: API Integration
- [ ] Create `app/api/v1/routers/mcp/` folder
- [ ] Implement query router
- [ ] Implement ask router
- [ ] Add authentication middleware
- [ ] Write API tests

### Phase 5: Migration & Optimization
- [ ] Enable feature flag in staging
- [ ] Migrate one endpoint at a time
- [ ] Monitor performance
- [ ] Optimize slow queries
- [ ] Full migration to production

## 19. Risk Mitigation

### Risk 1: Performance Degradation
**Mitigation:**
- Implement query timeout (5 seconds)
- Add query complexity limits (max 5 levels of nesting)
- Use indexes on all foreign keys
- Monitor query execution time
- Implement caching for master data

### Risk 2: Breaking Changes
**Mitigation:**
- Feature flag for gradual rollout
- Maintain backward compatibility
- Version API endpoints (`/api/v1/mcp/v1/query`)
- Comprehensive testing before migration

### Risk 3: Complex Pipeline Errors
**Mitigation:**
- Validate pipeline before execution
- Catch and log aggregation errors
- Fallback to simple queries on error
- Unit tests for all pipeline patterns

### Risk 4: Scope Filtering Issues
**Mitigation:**
- Validate JWT token structure
- Handle missing postCode gracefully
- Log scope filter application
- Test with different user roles

## 20. Success Metrics

### Technical Metrics:
- Query response time: < SLA targets (see Section 15)
- Error rate: < 0.1%
- Pipeline execution success rate: > 99.9%
- Test coverage: > 80%

### Business Metrics:
- User satisfaction with query results
- Reduction in follow-up queries
- Adoption rate of MCP endpoints
- Reduction in API calls (fewer round trips)

## 21. Next Steps (Immediate Actions)

1. **Week 1:**
   - [ ] Review and finalize this architecture document
   - [ ] Set up development environment
   - [ ] Create folder structure
   - [ ] Implement `relationship_mapper.py` (proof of concept)

2. **Week 2:**
   - [ ] Implement `pipeline_builder.py` with basic lookups
   - [ ] Implement `scope_context.py`
   - [ ] Write unit tests
   - [ ] Test with simple personnel query

3. **Week 3:**
   - [ ] Implement `PersonnelRepository` with enrichment
   - [ ] Create first MCP tool: `search_personnel`
   - [ ] Integration testing
   - [ ] Performance benchmarking

4. **Week 4:**
   - [ ] Implement remaining entity repositories
   - [ ] Implement complex query tools
   - [ ] API endpoint implementation
   - [ ] End-to-end testing

**Ready to start implementation!** 🚀
