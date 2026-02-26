"""
Unit Query Tools for MCP

Tools for querying unit hierarchy and structure.
"""

from typing import Any, Dict, List, Optional
import re
from bson import ObjectId

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.query_builder.aggregation_builder import AggregationBuilder, AggregationHelpers
from mcp.constants import Collections
from mcp.query_builder.builder import SafeQueryBuilder
from mcp.router.extractors import normalize_common_entity_aliases, fuzzy_best_match


def _clean_district_name(district_name: str) -> str:
    """Normalize district name by trimming and removing optional suffix."""
    return re.sub(r"\s*district\s*$", "", district_name, flags=re.IGNORECASE).strip()


async def _find_district_by_name(db: Any, district_name: str) -> Optional[Dict[str, Any]]:
    """Find district by exact name, then partial match."""
    clean_name = _clean_district_name(district_name)
    clean_name = _clean_district_name(normalize_common_entity_aliases(clean_name))
    district = await db[Collections.DISTRICT].find_one(
        {"name": {"$regex": f"^{clean_name}$", "$options": "i"}, "isDelete": False}
    )
    if district:
        return district

    escaped_name = SafeQueryBuilder.escape_regex(clean_name)
    district = await db[Collections.DISTRICT].find_one(
        {"name": {"$regex": escaped_name, "$options": "i"}, "isDelete": False}
    )
    if district:
        return district

    district_names = await db[Collections.DISTRICT].distinct("name", {"isDelete": False})
    best = fuzzy_best_match(clean_name, district_names, cutoff=0.76)
    if not best:
        return None
    return await db[Collections.DISTRICT].find_one(
        {"name": {"$regex": f"^{re.escape(best)}$", "$options": "i"}, "isDelete": False}
    )


class GetUnitHierarchyTool(BaseTool):
    """Get unit hierarchy tree with personnel counts"""

    name = "get_unit_hierarchy"
    description = (
        "Get the unit hierarchy tree starting from a specific unit or district. "
        "Shows parent-child relationships and personnel counts at each level."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "root_unit_id": {
                    "type": "string",
                    "description": "Root unit ID to start hierarchy from",
                },
                "root_unit_name": {
                    "type": "string",
                    "description": "Root unit name (case insensitive)",
                },
                "district_id": {
                    "type": "string",
                    "description": "Get all top-level units in a district",
                },
                "district_name": {
                    "type": "string",
                    "description": "District name (case insensitive)",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth of hierarchy to return",
                    "default": 3,
                },
                "include_personnel_count": {
                    "type": "boolean",
                    "description": "Include personnel count for each unit",
                    "default": True,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        root_unit_id = arguments.get("root_unit_id")
        root_unit_name = arguments.get("root_unit_name")
        district_id = arguments.get("district_id")
        district_name = arguments.get("district_name")
        max_depth = min(arguments.get("max_depth", 3), 5)  # Cap at 5
        include_personnel_count = arguments.get("include_personnel_count", True)

        resolved = await self._resolve_hierarchy_targets(
            root_unit_id=root_unit_id,
            root_unit_name=root_unit_name,
            district_id=district_id,
            district_name=district_name,
        )
        root_unit_id = resolved["root_unit_id"]
        district_id = resolved["district_id"]
        original_root_unit_name = resolved["original_root_unit_name"]
        original_district_name = resolved["original_district_name"]

        not_found = self._validate_hierarchy_resolution(
            original_root_unit_name=original_root_unit_name,
            original_district_name=original_district_name,
            root_unit_id=root_unit_id,
            district_id=district_id,
        )
        if not_found:
            return not_found

        # Build base query for root units
        if root_unit_id:
            # Get hierarchy starting from specific unit
            return await self._get_hierarchy_from_unit(
                root_unit_id, max_depth, include_personnel_count, context
            )
        elif district_id:
            # Get all top-level units in district
            return await self._get_district_units(
                district_id, max_depth, include_personnel_count, context
            )
        else:
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide root_unit_id, root_unit_name, district_id, or district_name",
            )

    async def _resolve_hierarchy_targets(
        self,
        root_unit_id: Optional[str],
        root_unit_name: Optional[str],
        district_id: Optional[str],
        district_name: Optional[str],
    ) -> Dict[str, Optional[str]]:
        original_root_unit_name = root_unit_name
        original_district_name = district_name

        if not root_unit_id and root_unit_name:
            root_unit_id = await self._resolve_unit_name(root_unit_name)

        if not district_id and district_name:
            district_id = await self._resolve_district_name(district_name)
            if not district_id and not root_unit_name:
                root_unit_name = district_name
                root_unit_id = await self._resolve_unit_name(root_unit_name)

        if not district_id and not root_unit_id and root_unit_name and not district_name:
            district_id = await self._resolve_district_name(root_unit_name)

        return {
            "root_unit_id": root_unit_id,
            "district_id": district_id,
            "original_root_unit_name": original_root_unit_name,
            "original_district_name": original_district_name,
        }

    def _validate_hierarchy_resolution(
        self,
        original_root_unit_name: Optional[str],
        original_district_name: Optional[str],
        root_unit_id: Optional[str],
        district_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if (
            original_root_unit_name
            and not root_unit_id
            and not district_id
            and not original_district_name
        ):
            return self.format_error_response(
                "NOT_FOUND",
                f"Unit not found: {original_root_unit_name}",
            )

        if original_district_name and not district_id and not root_unit_id:
            return self.format_error_response(
                "NOT_FOUND",
                f"District not found: {original_district_name}",
            )
        return None

    async def _resolve_unit_name(self, unit_name: str) -> Optional[str]:
        """Resolve unit name to ID"""
        probe = normalize_common_entity_aliases(unit_name or "")
        unit = await self.db[Collections.UNIT].find_one(
            {
                "name": {"$regex": f"^{re.escape(probe)}$", "$options": "i"},
                "isDelete": False,
            }
        )
        if unit:
            return str(unit["_id"])

        unit = await self.db[Collections.UNIT].find_one(
            {
                "name": {"$regex": SafeQueryBuilder.escape_regex(probe), "$options": "i"},
                "isDelete": False,
            },
            {"_id": 1},
        )
        if unit:
            return str(unit["_id"])

        unit_names = await self.db[Collections.UNIT].distinct("name", {"isDelete": False})
        best = fuzzy_best_match(probe, unit_names, cutoff=0.78)
        if not best:
            return None
        unit = await self.db[Collections.UNIT].find_one(
            {"name": {"$regex": f"^{re.escape(best)}$", "$options": "i"}, "isDelete": False},
            {"_id": 1},
        )
        return str(unit["_id"]) if unit else None

    async def _resolve_district_name(self, district_name: str) -> Optional[str]:
        """Resolve district name to ID"""
        district = await _find_district_by_name(self.db, district_name)
        return str(district["_id"]) if district else None

    async def _get_hierarchy_from_unit(
        self,
        root_unit_id: str,
        max_depth: int,
        include_personnel_count: bool,
        context: UserContext,
    ) -> Dict[str, Any]:
        """Get hierarchy starting from a specific unit"""
        if not ObjectId.is_valid(root_unit_id):
            return self.format_error_response(
                "VALIDATION_ERROR",
                f"Invalid unit id: {root_unit_id}",
            )

        # Get root unit
        root_query = await self.apply_scope_filter(
            {"_id": ObjectId(root_unit_id), "isDelete": False},
            context,
            "unit",
        )
        root_unit = await self.db[Collections.UNIT].find_one(root_query)

        if not root_unit:
            return self.format_error_response(
                "NOT_FOUND", f"Unit not found: {root_unit_id}"
            )

        # Build hierarchy tree
        tree = await self._build_unit_tree(
            root_unit, max_depth, include_personnel_count, 0, context
        )

        return self.format_success_response(
            query_type="unit_hierarchy",
            data=tree,
            metadata={
                "root_unit_id": root_unit_id,
                "max_depth": max_depth,
            },
        )

    async def _get_district_units(
        self,
        district_id: str,
        max_depth: int,
        include_personnel_count: bool,
        context: UserContext,
    ) -> Dict[str, Any]:
        """Get all units in a district"""
        # Get district info
        district = await self.db[Collections.DISTRICT].find_one(
            {"_id": ObjectId(district_id), "isDelete": False}
        )

        if not district:
            return self.format_error_response(
                "NOT_FOUND", f"District not found: {district_id}"
            )

        # Build base query for all units in district
        base_query = {
            "districtId": ObjectId(district_id),
            "isDelete": False,
        }

        # Apply scope filter
        query = await self.apply_scope_filter(base_query, context, "unit")

        all_units = await self.db[Collections.UNIT].find(query).to_list(length=None)
        top_units = self._derive_top_units(all_units)
        trees = await self._build_forest(top_units, max_depth, include_personnel_count, context)

        return self.format_success_response(
            query_type="district_unit_hierarchy",
            data={
                "district_id": district_id,
                "district_name": district.get("name"),
                "units": trees,
                "total_units": len(trees),
            },
            metadata={
                "district_id": district_id,
                "max_depth": max_depth,
            },
        )

    def _derive_top_units(self, all_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Derive district root units with fallback when no explicit root exists."""
        unit_ids = {u["_id"] for u in all_units}
        top_units = [
            unit
            for unit in all_units
            if not unit.get("parentUnitId") or unit.get("parentUnitId") not in unit_ids
        ]
        source = top_units if top_units else all_units
        return sorted(source, key=lambda unit: (unit.get("name") or "").lower())

    async def _build_forest(
        self,
        units: List[Dict[str, Any]],
        max_depth: int,
        include_personnel_count: bool,
        context: UserContext,
    ) -> List[Dict[str, Any]]:
        trees: List[Dict[str, Any]] = []
        for unit in units:
            tree = await self._build_unit_tree(unit, max_depth, include_personnel_count, 0, context)
            trees.append(tree)
        return trees

    async def _build_unit_tree(
        self,
        unit: Dict[str, Any],
        max_depth: int,
        include_personnel_count: bool,
        current_depth: int,
        context: UserContext,
    ) -> Dict[str, Any]:
        """Recursively build unit tree"""
        unit_id = unit["_id"]

        unit_type_name = await self._get_unit_type_name(unit.get("unitTypeId"))
        node: Dict[str, Any] = {
            "_id": str(unit_id),
            "name": unit.get("name", ""),
            "unitType": unit_type_name,
            "policeReferenceId": unit.get("policeReferenceId"),
        }

        if include_personnel_count:
            node["personnelCount"] = await self._count_active_personnel(unit_id)

        if current_depth < max_depth:
            children = await self._get_child_units(unit_id, context)
            if children:
                node["children"] = []
                for child in children:
                    child_tree = await self._build_unit_tree(
                        child, max_depth, include_personnel_count, current_depth + 1, context
                    )
                    node["children"].append(child_tree)

        return node

    async def _get_unit_type_name(self, unit_type_id: Optional[ObjectId]) -> Optional[str]:
        if not unit_type_id:
            return None
        unit_type = await self.db[Collections.UNIT_TYPE].find_one(
            {"_id": unit_type_id}, {"name": 1}
        )
        return unit_type.get("name") if unit_type else None

    async def _count_active_personnel(self, unit_id: ObjectId) -> int:
        return await self.db[Collections.PERSONNEL_MASTER].count_documents(
            {"units.unitId": unit_id, "isDelete": False, "isActive": True}
        )

    async def _get_child_units(
        self,
        unit_id: ObjectId,
        context: UserContext,
    ) -> List[Dict[str, Any]]:
        child_query = await self.apply_scope_filter(
            {"parentUnitId": unit_id, "isDelete": False},
            context,
            "unit",
        )
        cursor = self.db[Collections.UNIT].find(child_query).sort("name", 1)
        return await cursor.to_list(length=None)


class ListUnitsInDistrictTool(BaseTool):
    """List all units in a district with basic info"""

    name = "list_units_in_district"
    description = (
        "List all units in a specific district with personnel counts "
        "and responsible officer information."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "district_id": {
                    "type": "string",
                    "description": "District ID",
                },
                "district_name": {
                    "type": "string",
                    "description": "District name (case insensitive)",
                },
                "unit_type_id": {
                    "type": "string",
                    "description": "Filter by unit type ID",
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "default": 50,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        district_id = arguments.get("district_id")
        district_name = arguments.get("district_name")
        unit_type_id = arguments.get("unit_type_id")
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve district name
        if not district_id and district_name:
            district = await _find_district_by_name(self.db, district_name)
            if district:
                district_id = str(district["_id"])
            else:
                return self.format_error_response(
                    "NOT_FOUND",
                    f"District not found: {district_name}",
                )

        if not district_id:
            return self.format_error_response(
                "MISSING_PARAMETER",
                "District ID or name is required",
            )

        # Build query
        base_query: Dict[str, Any] = {
            "districtId": ObjectId(district_id),
            "isDelete": False,
        }

        if unit_type_id and ObjectId.is_valid(unit_type_id):
            base_query["unitTypeId"] = ObjectId(unit_type_id)

        # Apply scope filter
        query = await self.apply_scope_filter(base_query, context, "unit")

        # Build pipeline
        pipeline = (
            AggregationBuilder()
            .match(query)
            .lookup(
                from_collection=Collections.UNIT_TYPE,
                local_field="unitTypeId",
                foreign_field="_id",
                as_field="unitTypeData",
            )
            .unwind("unitTypeData")
            .lookup(
                from_collection=Collections.PERSONNEL_MASTER,
                local_field="responsibleUserId",
                foreign_field="_id",
                as_field="responsibleUserData",
            )
            .unwind("responsibleUserData")
            .add_fields({
                "personnelCount": AggregationHelpers.array_size("unitPersonnelList"),
            })
            .project({
                "_id": {"$toString": "$_id"},
                "name": 1,
                "policeReferenceId": 1,
                "unitType": "$unitTypeData.name",
                "personnelCount": 1,
                "responsibleUserName": "$responsibleUserData.name",
                "city": 1,
            })
            .sort({"name": 1})
            .skip(skip)
            .limit(page_size)
            .build()
        )

        results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(
            length=None
        )
        total = await self.db[Collections.UNIT].count_documents(query)

        # Get district name
        district_doc = await self.db[Collections.DISTRICT].find_one(
            {"_id": ObjectId(district_id)}, {"name": 1}
        )

        return self.format_success_response(
            query_type="units_in_district",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "district_id": district_id,
                "district_name": district_doc.get("name") if district_doc else None,
            },
        )


class ListDistrictsTool(BaseTool):
    """List available districts in accessible scope"""

    name = "list_districts"
    description = (
        "List available districts in the database. "
        "Supports optional name search and pagination."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional district name search (case insensitive)",
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "default": 50,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        page, page_size, skip = self.get_pagination_params(arguments)
        name = arguments.get("name")

        query: Dict[str, Any] = {"isDelete": False}

        if name:
            query["name"] = {"$regex": name, "$options": "i"}

        # Scope by district access where possible.
        if not context.has_state_access:
            district_ids = context.get_accessible_district_ids()
            if district_ids:
                query["_id"] = {"$in": [ObjectId(d) for d in district_ids if ObjectId.is_valid(d)]}
            else:
                accessible_units = await self.scope_filter.get_accessible_units(context)
                if accessible_units:
                    unit_district_ids = await self.db[Collections.UNIT].distinct(
                        "districtId",
                        {"_id": {"$in": accessible_units}, "isDelete": False},
                    )
                    query["_id"] = {"$in": unit_district_ids}
                else:
                    query["_id"] = {"$exists": False}

        cursor = (
            self.db[Collections.DISTRICT]
            .find(query, {"name": 1})
            .sort("name", 1)
            .skip(skip)
            .limit(page_size)
        )
        docs = await cursor.to_list(length=None)
        total = await self.db[Collections.DISTRICT].count_documents(query)

        data = [{"district_id": str(d["_id"]), "district_name": d.get("name")} for d in docs]

        return self.format_success_response(
            query_type="district_list",
            data=data,
            total=total,
            page=page,
            page_size=page_size,
        )
