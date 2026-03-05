"""Dynamic read-only query executor for LLM-driven MongoDB access.

Used internally by DynamicQueryOrchestrator. These classes are NOT registered
with ToolHandler and are never picked directly by the router.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.constants import Collections
from mcp.query_builder.filters import ScopeFilter
from mcp.schemas.context_schema import UserContext

logger = logging.getLogger(__name__)

_MAX_RESULTS = 500

# ---------------------------------------------------------------------------
# Collection whitelist
# ---------------------------------------------------------------------------

_ALL_COLLECTIONS: Set[str] = {
    v for k, v in vars(Collections).items()
    if not k.startswith("_") and isinstance(v, str)
}

# Collections excluded from LLM access (sensitive / operational)
_DENIED_COLLECTIONS: Set[str] = {
    Collections.LOG_MASTER,
    Collections.NOTIFICATION_MASTER,
    Collections.JOBS_MASTER,
    Collections.PERMISSIONS_MASTER,
    Collections.PERMISSIONS_MAPPING_MASTER,
    Collections.USER_ROLE_PERMISSIONS_MASTER,
    Collections.APPROVAL_FLOW_MASTER,
}

READABLE_COLLECTIONS: Set[str] = _ALL_COLLECTIONS - _DENIED_COLLECTIONS

# ---------------------------------------------------------------------------
# Known foreign-key relationships between collections.
# The LLM uses this to write correct $lookup stages for cross-collection joins.
# Format per entry: {field, references, ref_field, description}
#   field       — field in THIS collection that holds the foreign key
#   references  — the target collection name
#   ref_field   — field in the target collection that matches (usually "_id")
#   description — plain-English meaning of the join
# ---------------------------------------------------------------------------

COLLECTION_RELATIONSHIPS: Dict[str, List[Dict[str, str]]] = {
    Collections.PERSONNEL_MASTER: [
        {
            "field": "rankId",
            "references": Collections.RANK_MASTER,
            "ref_field": "_id",
            "description": "Rank of the officer (name, shortCode)",
        },
        {
            "field": "designationId",
            "references": Collections.DESIGNATION_MASTER,
            "ref_field": "_id",
            "description": "Designation/role title of the officer",
        },
        {
            "field": "departmentId",
            "references": Collections.DEPARTMENT,
            "ref_field": "_id",
            "description": "Department the officer belongs to",
        },
    ],
    Collections.ASSIGNMENT_MASTER: [
        {
            "field": "userId",
            "references": Collections.PERSONNEL_MASTER,
            "ref_field": "_id",
            "description": "The personnel record for this posting",
        },
        {
            "field": "unitId",
            "references": Collections.UNIT,
            "ref_field": "_id",
            "description": "The unit/station this officer is posted at",
        },
        {
            "field": "designationId",
            "references": Collections.DESIGNATION_MASTER,
            "ref_field": "_id",
            "description": "Role/designation within this posting",
        },
    ],
    Collections.UNIT: [
        {
            "field": "districtId",
            "references": Collections.DISTRICT,
            "ref_field": "_id",
            "description": "District this unit belongs to",
        },
        {
            "field": "parentUnitId",
            "references": Collections.UNIT,
            "ref_field": "_id",
            "description": "Parent unit in the hierarchy (self-referential)",
        },
        {
            "field": "unitTypeId",
            "references": Collections.UNIT_TYPE,
            "ref_field": "_id",
            "description": "Type/category of the unit",
        },
    ],
    Collections.UNIT_VILLAGES: [
        {
            "field": "unitId",
            "references": Collections.UNIT,
            "ref_field": "_id",
            "description": "Unit responsible for this village",
        },
        {
            "field": "mandalId",
            "references": Collections.MANDAL,
            "ref_field": "_id",
            "description": "Mandal (sub-district) the village is in",
        },
    ],
    Collections.MANDAL: [
        {
            "field": "districtId",
            "references": Collections.DISTRICT,
            "ref_field": "_id",
            "description": "District this mandal belongs to",
        },
    ],
}

# ---------------------------------------------------------------------------
# Blocked operators / write stages
# ---------------------------------------------------------------------------

_BLOCKED_OPERATORS: Set[str] = {
    "$where", "$function", "$accumulator",
    "$out", "$merge",
}

_WRITE_STAGES: Set[str] = {"$out", "$merge"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class QueryValidationError(Exception):
    """Raised when an LLM-generated query fails safety validation."""


class QueryValidator:
    """Validates that LLM-generated queries are safe to execute."""

    @staticmethod
    def validate_collection(collection: str) -> None:
        if collection not in READABLE_COLLECTIONS:
            raise QueryValidationError(
                f"Collection '{collection}' is not accessible. "
                "Call describe_collections to see valid options."
            )

    @staticmethod
    def _scan_blocked(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in _BLOCKED_OPERATORS:
                    raise QueryValidationError(f"Operator '{key}' is not allowed.")
                QueryValidator._scan_blocked(val)
        elif isinstance(obj, list):
            for item in obj:
                QueryValidator._scan_blocked(item)

    @staticmethod
    def validate_filter(filter_doc: Any) -> None:
        if not isinstance(filter_doc, dict):
            raise QueryValidationError("Filter must be a JSON object.")
        QueryValidator._scan_blocked(filter_doc)

    @staticmethod
    def validate_pipeline(pipeline: Any) -> None:
        if not isinstance(pipeline, list):
            raise QueryValidationError("Pipeline must be a JSON array.")
        for stage in pipeline:
            if not isinstance(stage, dict):
                raise QueryValidationError("Each pipeline stage must be a JSON object.")
            for key in stage:
                if key in _WRITE_STAGES:
                    raise QueryValidationError(
                        f"Stage '{key}' is not allowed (writes are prohibited)."
                    )
            QueryValidator._scan_blocked(stage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_scope_type(collection: str) -> str:
    """Map collection name to the ScopeFilter collection_type string."""
    if collection == Collections.PERSONNEL_MASTER:
        return "personnel"
    if collection in {Collections.UNIT, Collections.DISTRICT}:
        return "unit"
    if collection == Collections.UNIT_VILLAGES:
        return "unit_villages"
    return "generic"


def _serialize(doc: Any) -> Any:
    """Convert a MongoDB document to a JSON-safe value."""
    if isinstance(doc, dict):
        return {k: _serialize(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_serialize(i) for i in doc]
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, datetime):
        return doc.isoformat()
    return doc


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class DynamicQueryExecutor:
    """Executes validated read-only MongoDB queries with scope enforcement."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.scope_filter = ScopeFilter(db)

    async def describe_collections(self) -> Dict[str, Any]:
        """Return field names and known FK relationships for every readable collection."""
        schema: Dict[str, Any] = {}
        for coll_name in sorted(READABLE_COLLECTIONS):
            try:
                sample = await self.db[coll_name].find_one(
                    {"isDelete": False}, {"_id": 0}
                )
                if sample is None:
                    sample = await self.db[coll_name].find_one({}, {"_id": 0})
                fields = sorted(sample.keys()) if sample else []
            except Exception:
                fields = []

            entry: Dict[str, Any] = {"fields": fields}
            if coll_name in COLLECTION_RELATIONSHIPS:
                entry["relationships"] = COLLECTION_RELATIONSHIPS[coll_name]
            schema[coll_name] = entry

        return {"collections": schema}

    async def find(
        self,
        collection: str,
        filter_doc: Dict[str, Any],
        projection: Optional[Dict[str, Any]],
        limit: int,
        context: UserContext,
    ) -> Dict[str, Any]:
        QueryValidator.validate_collection(collection)
        QueryValidator.validate_filter(filter_doc)
        limit = min(max(1, limit), _MAX_RESULTS)

        scope_type = _infer_scope_type(collection)
        scoped_filter = await self.scope_filter.apply(filter_doc, context, scope_type)

        proj = projection if isinstance(projection, dict) else None
        cursor = self.db[collection].find(scoped_filter, proj).limit(limit)
        docs = await cursor.to_list(length=limit)
        return {"data": [_serialize(d) for d in docs], "count": len(docs)}

    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]],
        context: UserContext,
    ) -> Dict[str, Any]:
        QueryValidator.validate_collection(collection)
        QueryValidator.validate_pipeline(pipeline)

        scoped_pipeline = await self._inject_scope(pipeline, collection, context)
        cursor = self.db[collection].aggregate(scoped_pipeline)
        docs = await cursor.to_list(length=_MAX_RESULTS)
        return {"data": [_serialize(d) for d in docs], "count": len(docs)}

    async def _inject_scope(
        self,
        pipeline: List[Dict[str, Any]],
        collection: str,
        context: UserContext,
    ) -> List[Dict[str, Any]]:
        """Prepend scope $match and guarantee a $limit cap."""
        result = list(pipeline)
        if not context.has_state_access:
            scope_query = await self.scope_filter.apply(
                {}, context, _infer_scope_type(collection)
            )
            if scope_query:
                result.insert(0, {"$match": scope_query})
        if not any("$limit" in stage for stage in result):
            result.append({"$limit": _MAX_RESULTS})
        return result
