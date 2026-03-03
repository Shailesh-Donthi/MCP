"""
Scope Filters for MCP Queries

Applies unit/district-based access controls to queries based on user context.
"""

from typing import Any, Dict, List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.schemas.context_schema import UserContext
from mcp.constants import Collections


class ScopeFilter:
    """
    Apply scope-based filters to queries based on user context.
    Ensures users only see data they have access to.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def apply(
        self,
        base_query: Dict[str, Any],
        context: UserContext,
        collection_type: str = "personnel",
    ) -> Dict[str, Any]:
        """
        Apply scope filters to a query based on user context.

        Args:
            base_query: The base MongoDB query
            context: User context with access information
            collection_type: Type of collection being queried
                           ("personnel", "unit", "unit_villages")

        Returns:
            Modified query with scope filters applied
        """
        # If state-level access, no additional filters needed
        if context.has_state_access:
            return base_query

        # Build scope conditions based on collection type
        if collection_type == "personnel":
            scope_conditions = await self._build_personnel_scope(context)
        elif collection_type == "unit":
            scope_conditions = await self._build_unit_scope(context)
        elif collection_type == "unit_villages":
            scope_conditions = await self._build_unit_villages_scope(context)
        else:
            scope_conditions = {}

        # Merge with base query
        if scope_conditions:
            return self._merge_conditions(base_query, scope_conditions)

        return base_query

    def _merge_conditions(
        self, base_query: Dict[str, Any], scope_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge scope conditions with base query using $and"""
        if not scope_conditions:
            return base_query

        # If base query already has $and, extend it
        if "$and" in base_query:
            base_query = base_query.copy()
            base_query["$and"] = list(base_query["$and"]) + [scope_conditions]
            return base_query

        # If there's overlap, use $and
        overlapping_keys = set(base_query.keys()) & set(scope_conditions.keys())
        if overlapping_keys:
            return {"$and": [base_query, scope_conditions]}

        # Otherwise, merge directly
        return {**base_query, **scope_conditions}

    async def _build_personnel_scope(self, context: UserContext) -> Dict[str, Any]:
        """Build scope filter for personnel queries"""

        if context.has_district_access:
            district_ids = context.get_accessible_district_ids()
            if district_ids:
                # Get all units in user's districts
                unit_ids = await self._get_units_in_districts(district_ids)
                if unit_ids:
                    return {"units.unitId": {"$in": unit_ids}}

        # Unit-level access
        unit_ids = context.get_accessible_unit_ids()
        if unit_ids:
            # Include child units
            all_units = await self._get_units_with_children(unit_ids)
            return {"units.unitId": {"$in": all_units}}

        # No access - return impossible condition
        return {"_id": {"$exists": False}}

    async def _build_unit_scope(self, context: UserContext) -> Dict[str, Any]:
        """Build scope filter for unit queries"""

        if context.has_district_access:
            district_ids = context.get_accessible_district_ids()
            if district_ids:
                return {
                    "districtId": {
                        "$in": [ObjectId(did) for did in district_ids]
                    }
                }

        # Unit-level access - include units and their children
        unit_ids = context.get_accessible_unit_ids()
        if unit_ids:
            all_units = await self._get_units_with_children(unit_ids)
            return {"_id": {"$in": all_units}}

        # No access
        return {"_id": {"$exists": False}}

    async def _build_unit_villages_scope(
        self, context: UserContext
    ) -> Dict[str, Any]:
        """Build scope filter for unit_villages queries"""

        if context.has_district_access:
            district_ids = context.get_accessible_district_ids()
            if district_ids:
                unit_ids = await self._get_units_in_districts(district_ids)
                if unit_ids:
                    return {"unitId": {"$in": unit_ids}}

        # Unit-level access
        unit_ids = context.get_accessible_unit_ids()
        if unit_ids:
            return {"unitId": {"$in": [ObjectId(uid) for uid in unit_ids]}}

        # No access
        return {"_id": {"$exists": False}}

    async def _get_units_in_districts(
        self, district_ids: List[str]
    ) -> List[ObjectId]:
        """Get all unit IDs in given districts"""
        try:
            object_ids = [ObjectId(did) for did in district_ids]
            cursor = self.db[Collections.UNIT].find(
                {"districtId": {"$in": object_ids}, "isDelete": False},
                {"_id": 1},
            )
            units = await cursor.to_list(length=None)
            return [u["_id"] for u in units]
        except Exception:
            return []

    async def _get_units_with_children(
        self, unit_ids: List[str]
    ) -> List[ObjectId]:
        """Get unit IDs including all child units"""
        try:
            # Convert to ObjectIds
            object_ids = [ObjectId(uid) for uid in unit_ids]
            all_units = set(object_ids)

            # Get direct children using parentUnitId
            cursor = self.db[Collections.UNIT].find(
                {
                    "parentUnitId": {"$in": list(object_ids)},
                    "isDelete": False,
                },
                {"_id": 1},
            )
            children = await cursor.to_list(length=None)

            for child in children:
                all_units.add(child["_id"])

            # Recursively get children of children (up to 3 levels deep)
            for _ in range(2):  # 2 more levels
                if not children:
                    break
                child_ids = [c["_id"] for c in children]
                cursor = self.db[Collections.UNIT].find(
                    {
                        "parentUnitId": {"$in": child_ids},
                        "isDelete": False,
                    },
                    {"_id": 1},
                )
                children = await cursor.to_list(length=None)
                for child in children:
                    all_units.add(child["_id"])

            return list(all_units)
        except Exception:
            return [ObjectId(uid) for uid in unit_ids]

    async def get_accessible_units(self, context: UserContext) -> List[ObjectId]:
        """Get all accessible unit ObjectIds for a user context"""
        if context.has_state_access:
            # Return all units
            cursor = self.db[Collections.UNIT].find(
                {"isDelete": False}, {"_id": 1}
            )
            units = await cursor.to_list(length=None)
            return [u["_id"] for u in units]

        if context.has_district_access:
            district_ids = context.get_accessible_district_ids()
            if district_ids:
                return await self._get_units_in_districts(district_ids)

        unit_ids = context.get_accessible_unit_ids()
        if unit_ids:
            return await self._get_units_with_children(unit_ids)

        return []

