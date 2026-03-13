"""Dynamic MongoDB schema scanner.

Scans all readable collections at server startup to discover fields, types,
foreign-key relationships, and low-cardinality enum values.  The result is
used to generate the LLM system prompt dynamically instead of relying on a
hardcoded static schema.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.core.database import get_database
from mcp.tools.dynamic_query_tool import COLLECTION_RELATIONSHIPS, READABLE_COLLECTIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_SAMPLE_SIZE = 20
_ENUM_MAX_DISTINCT = 25


@dataclass
class FieldInfo:
    name: str
    field_type: str  # "string", "objectId", "bool", "date", "number", "array", "object"
    is_fk: bool = False
    fk_target: Optional[str] = None  # e.g. "rank_master._id"
    enum_values: Optional[List[str]] = None


@dataclass
class CollectionSchema:
    name: str
    fields: List[FieldInfo] = field(default_factory=list)
    doc_count: int = 0


@dataclass
class SchemaInfo:
    collections: Dict[str, CollectionSchema] = field(default_factory=dict)
    scanned_at: str = ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_schema_info: Optional[SchemaInfo] = None


def get_schema_info() -> SchemaInfo:
    """Return the cached schema info.  Falls back to empty if not yet scanned."""
    if _schema_info is None:
        return SchemaInfo(scanned_at=datetime.now(timezone.utc).isoformat())
    return _schema_info


async def scan_schema() -> SchemaInfo:
    """Scan all readable collections and cache the result."""
    global _schema_info
    db = get_database()
    if db is None:
        logger.warning("schema_scan skipped: no database connection")
        _schema_info = SchemaInfo(scanned_at=datetime.now(timezone.utc).isoformat())
        return _schema_info

    t0 = time.perf_counter()
    schema = SchemaInfo(scanned_at=datetime.now(timezone.utc).isoformat())

    for coll_name in sorted(READABLE_COLLECTIONS):
        try:
            coll_schema = await _scan_collection(db, coll_name)
            schema.collections[coll_name] = coll_schema
        except Exception:
            logger.exception("schema_scan error for %s", coll_name)
            schema.collections[coll_name] = CollectionSchema(name=coll_name)

    elapsed = time.perf_counter() - t0
    total_fields = sum(len(cs.fields) for cs in schema.collections.values())
    logger.info(
        '{"event": "schema_scan_complete", "collections": %d, "fields": %d, "seconds": %.2f}',
        len(schema.collections),
        total_fields,
        elapsed,
    )

    _schema_info = schema
    return schema


# ---------------------------------------------------------------------------
# Per-collection scanning
# ---------------------------------------------------------------------------

_BSON_TYPE_MAP = {
    str: "string",
    int: "number",
    float: "number",
    bool: "bool",
    ObjectId: "objectId",
    datetime: "date",
    list: "array",
    dict: "object",
}


def _infer_type(value: Any) -> str:
    """Map a Python/BSON value to a simple type label."""
    for py_type, label in _BSON_TYPE_MAP.items():
        if isinstance(value, py_type):
            return label
    return "unknown"


def _guess_fk_target(field_name: str, readable: Set[str]) -> Optional[str]:
    """Heuristic: if field ends with 'Id', try to find a matching collection."""
    if not field_name.endswith("Id") or field_name == "_id":
        return None

    # rankId -> rank, districtId -> district, parentUnitId -> parentUnit
    base = field_name[:-2]  # strip 'Id'

    # Try common collection naming patterns
    candidates = [
        f"{base}_master",
        f"{base}",
        # camelCase -> snake_case: unitType -> unit_type_master
        _camel_to_snake(base) + "_master",
        _camel_to_snake(base),
    ]
    for candidate in candidates:
        if candidate in readable:
            return candidate
    return None


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


async def _scan_collection(db: AsyncIOMotorDatabase, coll_name: str) -> CollectionSchema:
    """Sample documents, infer types, detect FKs and enums for one collection."""
    coll = db[coll_name]

    # Get approximate doc count
    doc_count = await coll.estimated_document_count()

    # Sample documents (prefer non-deleted)
    samples = await coll.find({"isDelete": False}).limit(_SAMPLE_SIZE).to_list(length=_SAMPLE_SIZE)
    if not samples:
        samples = await coll.find().limit(_SAMPLE_SIZE).to_list(length=_SAMPLE_SIZE)
    if not samples:
        return CollectionSchema(name=coll_name, doc_count=doc_count)

    # Collect all field names and type counts
    field_types: Dict[str, Counter] = {}
    for doc in samples:
        for key, value in doc.items():
            if key not in field_types:
                field_types[key] = Counter()
            field_types[key][_infer_type(value)] += 1

    # Build known FK map from COLLECTION_RELATIONSHIPS
    known_fks: Dict[str, str] = {}
    if coll_name in COLLECTION_RELATIONSHIPS:
        for rel in COLLECTION_RELATIONSHIPS[coll_name]:
            known_fks[rel["field"]] = f"{rel['references']}.{rel['ref_field']}"

    # Build FieldInfo list
    fields: List[FieldInfo] = []
    for fname in sorted(field_types.keys()):
        most_common_type = field_types[fname].most_common(1)[0][0]

        # FK detection: known FKs take priority, then heuristic
        fk_target = known_fks.get(fname)
        if fk_target is None and most_common_type == "objectId" and fname != "_id":
            guessed = _guess_fk_target(fname, READABLE_COLLECTIONS)
            if guessed:
                fk_target = f"{guessed}._id"

        is_fk = fk_target is not None

        # Enum detection for string fields
        enum_values = None
        if most_common_type == "string" and fname not in ("_id", "name") and doc_count > 0:
            try:
                distinct_vals = await coll.distinct(fname, {"isDelete": False})
                # Filter to actual strings
                str_vals = sorted(set(str(v) for v in distinct_vals if isinstance(v, str) and v))
                if 1 < len(str_vals) <= _ENUM_MAX_DISTINCT:
                    enum_values = str_vals
            except Exception:
                pass  # skip enum detection on error

        fields.append(FieldInfo(
            name=fname,
            field_type=most_common_type,
            is_fk=is_fk,
            fk_target=fk_target,
            enum_values=enum_values,
        ))

    return CollectionSchema(name=coll_name, fields=fields, doc_count=doc_count)


# ---------------------------------------------------------------------------
# Prompt & response generation
# ---------------------------------------------------------------------------

_RULES = """\

## Rules
1. Always add "isDelete":false to filters. ONLY add "isActive":true on assignment_master (for current postings). Do NOT filter by isActive on personnel_master — most personnel have isActive=false but are still valid records.
2. Use $lookup in aggregate for cross-collection joins — never multiple sequential finds. ALWAYS use aggregate with $lookup to resolve foreign key IDs to human-readable names before calling "done".
3. For rank queries: $lookup rank_master on rankId, then filter by rank.shortCode or rank.name.
4. For district queries: use a SINGLE aggregate on assignment_master with $lookup unit_master (on unitId), $match unit's districtId, then $lookup personnel_master (on userId). This avoids multiple round-trips.
5. For missing mappings: $lookup unit_villages_master on unitId, $match villages size==0.
6. For transfers: filter assignment_master.fromDate >= date range.
7. Do NOT use $where,$function,$accumulator,$out,$merge — blocked.
8. Scope filters are applied automatically — do not add districtId/unitId scope yourself.
9. You have at most {max_turns} turns. Call "done" before running out.
10. If the query is unrelated to police personnel, call done immediately with a polite note.
11. If data is absent, say so clearly (e.g. "No vacancy data found in the system").
12. When the user asks to "list all" or "show all", return individual records with names/details — do NOT return counts or group-by summaries unless explicitly asked for counts. Use a $limit of 500. Do NOT use small limits like 10 or 50 for listing queries. For $count/$group queries, omit $limit entirely so all records are counted.
13. NEVER include raw ObjectIDs (24-character hex strings) in your final "done" answer. Always resolve ALL foreign keys to their actual names using $lookup BEFORE answering. For personnel queries, always $lookup: rankId->rank_master.name, departmentId->department_master.name, and via assignment_master: unitId->unit_master.name, unit_master.districtId->district_master.name, designationId->designation_master.name. If a name cannot be resolved, omit the field — do NOT show the raw ID.
14. In your final "done" answer, always echo back the key entities from the user's question (district names, rank names, unit names, person names). For example, if the user asks about "Chittoor district", your answer must mention "Chittoor". If no results found, still mention the queried entity (e.g. "No ASI personnel found in Annamayya district" not just "No records found").

## Query Strategy Templates
Use these proven patterns for common query types instead of guessing.

### Pattern A: Personnel by rank in a district
Step 1 - get the district ObjectId:
  find on district_master: {"name":{"$regex":"Chittoor","$options":"i"},"isDelete":false}, projection: {"_id":1}
Step 2 - aggregate from assignment_master:
  $match: {"isActive":true,"isDelete":false}
  $lookup: {"from":"unit_master","localField":"unitId","foreignField":"_id","as":"unit"}
  $unwind: "$unit"
  $match: {"unit.districtId":<district_oid_from_step1>}
  $lookup: {"from":"personnel_master","localField":"userId","foreignField":"_id","as":"person"}
  $unwind: "$person"
  $lookup: {"from":"rank_master","localField":"person.rankId","foreignField":"_id","as":"rank"}
  $unwind: "$rank"
  $match: {"rank.shortCode":"<RANK>"}  (or use "rank.name" with $regex for full name)
  $project: {"name":"$person.name","rank":"$rank.name","unit":"$unit.name"}

### Pattern B: Count personnel by rank per district
Same as Pattern A but replace the final $project with:
  $group: {"_id":"$district.name","count":{"$sum":1}}
  $sort: {"count":-1}
(Add $lookup district_master on unit.districtId before $group)

### Pattern C: Personnel by designation (SDPO, SHO, etc.)
Designation != rank. Designation is in designation_master. Use:
  aggregate on assignment_master:
  $match: {"isActive":true,"isDelete":false}
  $lookup: {"from":"designation_master","localField":"designationId","foreignField":"_id","as":"desig"}
  $unwind: "$desig"
  $match: {"desig.name":{"$regex":"SDPO","$options":"i"}}
  $lookup personnel_master on userId, $lookup unit_master on unitId
  $project: {"name":"$person.name","designation":"$desig.name","unit":"$unit.name"}

### Pattern D: Resolve a district name to ObjectId (prerequisite for district-filtered queries)
  find on district_master: {"name":{"$regex":"<name>","$options":"i"},"isDelete":false}
  Use the returned _id in subsequent $match stages on districtId fields.

### Strategy tips
- assignment_master is the HUB table connecting personnel to units. Start there for "who is posted where" queries.
- Rank (rank_master) = pay/seniority level (SI, DSP, SP). Designation (designation_master) = functional role (SHO, SDPO). They are SEPARATE.
- Always use case-insensitive $regex for name matching: {"$regex":"...","$options":"i"}.
- When 0 results: check field names, regex case, ObjectId resolution, isDelete:false filter.
- For "how many" queries, use $group + $count instead of fetching all docs.
"""


def generate_system_prompt(schema: SchemaInfo) -> str:
    """Build the full LLM system prompt from scanned schema."""
    lines = [
        "You are a read-only MongoDB query assistant for a Police Personnel management system.",
        "Answer the user's intent by querying the database step-by-step.",
        "Output ONLY a single JSON object per turn — no prose before or after.",
        "",
        "## Actions",
        '{"action":"find","collection":"<col>","filter":{...},"projection":{...},"limit":<1-500>}',
        '{"action":"aggregate","collection":"<col>","pipeline":[...]}',
        '{"action":"done","answer":"<natural language answer>"}',
        "",
        "## Schema (auto-discovered)",
    ]

    # Schema section: one line per collection with typed fields
    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        if not cs.fields:
            lines.append(f"{coll_name}: (empty)")
            continue

        parts = []
        for f in cs.fields:
            if f.name == "__v":
                continue  # skip Mongoose version key
            type_suffix = ""
            if f.is_fk and f.fk_target:
                type_suffix = f"(oid->{f.fk_target.split('.')[0]})"
            elif f.field_type not in ("string", "unknown"):
                type_suffix = f"({f.field_type})"
            parts.append(f"{f.name}{type_suffix}")
        lines.append(f"{coll_name} ({cs.doc_count} docs): {', '.join(parts)}")

    # FK relationships section
    fk_lines = []
    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        for f in cs.fields:
            if f.is_fk and f.fk_target:
                fk_lines.append(f"{coll_name}.{f.name}->{f.fk_target}")
    if fk_lines:
        lines.append("")
        lines.append("## Foreign keys")
        # Group FKs into lines of 3 for readability
        for i in range(0, len(fk_lines), 3):
            lines.append(" | ".join(fk_lines[i:i + 3]))

    # Enum values section
    enum_lines = []
    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        for f in cs.fields:
            if f.enum_values:
                vals = ", ".join(f.enum_values[:20])
                enum_lines.append(f"{coll_name}.{f.name}: [{vals}]")
    if enum_lines:
        lines.append("")
        lines.append("## Known values (enums)")
        lines.extend(enum_lines)

    # Append static rules
    lines.append(_RULES)

    return "\n".join(lines)


def generate_mcp_system_prompt(schema: SchemaInfo) -> str:
    """Minimal system prompt for MongoDB MCP mode -- schema only, no domain rules."""
    lines = [
        "You are a read-only MongoDB query assistant for a Police Personnel management system.",
        "The database contains police personnel, units, districts, assignments, and related data.",
        "Answer the user's intent by querying the database step-by-step.",
        "Output ONLY a single JSON object per turn -- no prose, no markdown, no code fences, no tool calls.",
        "IMPORTANT: Your entire response must be a single valid JSON object and nothing else.",
        "",
        "## Actions",
        '{"action":"find","collection":"<col>","filter":{...},"projection":{...},"limit":<1-500>}',
        '{"action":"aggregate","collection":"<col>","pipeline":[...]}',
        '{"action":"describe_collections"}',
        '{"action":"done","answer":"<natural language answer>"}',
        "",
        "## Constraints",
        "- Blocked operators: $where, $function, $accumulator, $out, $merge",
        "- Read-only. No writes.",
        "- You have at most {max_turns} turns. Call done before running out.",
        "- Use $lookup to resolve foreign key ObjectIDs to human-readable names before answering.",
        "- Most collections have an isDelete(bool) field. Add isDelete:false to filters to exclude deleted records.",
        "- Boolean fields use true/false (not 1/0).",
        "- ObjectId fields: use plain 24-char hex strings (e.g. \"64a1b2c3...\"). Do NOT use {\"$oid\":\"...\"}.",
        "- For current assignments, filter assignment_master by isActive:true.",
        "",
        "## Schema (auto-discovered)",
    ]

    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        if not cs.fields:
            lines.append(f"{coll_name}: (empty)")
            continue
        parts = []
        for f in cs.fields:
            if f.name == "__v":
                continue
            type_suffix = ""
            if f.is_fk and f.fk_target:
                type_suffix = f"(oid->{f.fk_target.split('.')[0]})"
            elif f.field_type not in ("string", "unknown"):
                type_suffix = f"({f.field_type})"
            parts.append(f"{f.name}{type_suffix}")
        lines.append(f"{coll_name} ({cs.doc_count} docs): {', '.join(parts)}")

    fk_lines = []
    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        for f in cs.fields:
            if f.is_fk and f.fk_target:
                fk_lines.append(f"{coll_name}.{f.name}->{f.fk_target}")
    if fk_lines:
        lines.append("")
        lines.append("## Foreign keys")
        for i in range(0, len(fk_lines), 3):
            lines.append(" | ".join(fk_lines[i:i + 3]))

    enum_lines = []
    for coll_name in sorted(schema.collections):
        cs = schema.collections[coll_name]
        for f in cs.fields:
            if f.enum_values:
                vals = ", ".join(f.enum_values[:20])
                enum_lines.append(f"{coll_name}.{f.name}: [{vals}]")
    if enum_lines:
        lines.append("")
        lines.append("## Known values (enums)")
        lines.extend(enum_lines)

    return "\n".join(lines)


def generate_schema_response(schema: SchemaInfo) -> Dict[str, Any]:
    """Build the describe_collections response dict from scanned schema."""
    collections: Dict[str, Any] = {}
    for coll_name, cs in schema.collections.items():
        entry: Dict[str, Any] = {
            "fields": [f.name for f in cs.fields],
            "doc_count": cs.doc_count,
        }
        # Add relationships
        rels = {}
        for f in cs.fields:
            if f.is_fk and f.fk_target:
                rels[f.name] = f.fk_target
        if rels:
            entry["relationships"] = rels
        collections[coll_name] = entry
    return {"collections": collections}
