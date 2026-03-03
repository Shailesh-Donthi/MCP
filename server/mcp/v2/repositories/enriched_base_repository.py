"""Base repository for V2 relationship-aware enriched queries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.constants import Collections
from mcp.query_builder.filters import ScopeFilter
from mcp.utils.formatters import stringify_object_ids
from mcp.v2.repositories.pipeline_builder import PipelineBuilder
from mcp.v2.repositories.scope_context import ScopeContext


class EnrichedBaseRepository:
    """Shared repository implementation for enrichment + scoped access."""

    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str):
        self.db = db
        self.collection_name = collection_name
        self.pipeline_builder = PipelineBuilder()
        self.scope_filter = ScopeFilter(db)

    async def find_enriched(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        enrichments: Optional[List[str]] = None,
        scope_context: Optional[ScopeContext] = None,
        sort: Optional[Dict[str, int]] = None,
        page: int = 1,
        page_size: int = 50,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, int(page_size or 50))
        skip = (page - 1) * page_size

        scope_query = await self._build_scope_query(scope_context)
        data_pipeline = self.pipeline_builder.build_enriched_query(
            base_collection=self.collection_name,
            filters=filters or {},
            enrichments=enrichments,
            scope_filter=scope_query,
            sort=sort,
            skip=skip,
            limit=page_size,
            projection=projection,
        )
        data = await self.db[self.collection_name].aggregate(data_pipeline).to_list(length=None)

        count_query = self.pipeline_builder.build_count_query(
            filters=filters or {},
            scope_filter=scope_query,
        )
        total = await self.db[self.collection_name].count_documents(count_query)

        return {
            "data": [stringify_object_ids(row) for row in data],
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            },
        }

    async def find_one_enriched(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        enrichments: Optional[List[str]] = None,
        scope_context: Optional[ScopeContext] = None,
        sort: Optional[Dict[str, int]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload = await self.find_enriched(
            filters=filters or {},
            enrichments=enrichments,
            scope_context=scope_context,
            sort=sort,
            page=1,
            page_size=1,
            projection=projection,
        )
        rows = payload.get("data", [])
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        return row if isinstance(row, dict) else None

    async def _build_scope_query(self, scope_context: Optional[ScopeContext]) -> Dict[str, Any]:
        if scope_context is None or scope_context.has_state_access:
            return {}

        user_context = scope_context.to_user_context()

        collection_type = self._scope_collection_type()
        if collection_type:
            return await self.scope_filter.apply({}, user_context, collection_type)

        if self.collection_name == Collections.ASSIGNMENT_MASTER:
            accessible_units = await self.scope_filter.get_accessible_units(user_context)
            return {"unitId": {"$in": accessible_units}} if accessible_units else {"_id": {"$exists": False}}

        if self.collection_name == Collections.DISTRICT:
            district_ids = [
                ObjectId(did)
                for did in scope_context.get_accessible_district_ids()
                if ObjectId.is_valid(did)
            ]
            return {"_id": {"$in": district_ids}} if district_ids else {"_id": {"$exists": False}}

        return {}

    def _scope_collection_type(self) -> Optional[str]:
        if self.collection_name == Collections.PERSONNEL_MASTER:
            return "personnel"
        if self.collection_name == Collections.UNIT:
            return "unit"
        if self.collection_name == Collections.UNIT_VILLAGES:
            return "unit_villages"
        return None

