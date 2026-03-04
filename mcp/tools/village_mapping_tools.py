"""
Village Mapping Tools for MCP

Tools for querying village coverage and identifying gaps.
"""

from typing import Any, Dict, List, Optional
from bson import ObjectId

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.constants import Collections


class FindMissingVillageMappingsTool(BaseTool):
    """Find units or personnel with missing village mappings"""

    name = "find_missing_village_mappings"
    description = (
        "Find units that have no village mappings, or personnel assigned "
        "to units without village coverage. Helps identify gaps in "
        "jurisdictional coverage."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "district_id": {
                    "type": "string",
                    "description": "Filter by district ID",
                },
                "district_name": {
                    "type": "string",
                    "description": "Filter by district name (case insensitive)",
                },
                "unit_id": {
                    "type": "string",
                    "description": "Check specific unit ID",
                },
                "check_type": {
                    "type": "string",
                    "enum": [
                        "units_without_villages",
                        "personnel_in_unmapped_units",
                    ],
                    "description": "What to check for",
                    "default": "units_without_villages",
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
        check_type = arguments.get("check_type", "units_without_villages")
        district_id = arguments.get("district_id")
        district_name = arguments.get("district_name")
        unit_id = arguments.get("unit_id")
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve district name
        if not district_id and district_name:
            district = await self.db[Collections.DISTRICT].find_one(
                {
                    "name": {"$regex": f"^{district_name}$", "$options": "i"},
                    "isDelete": False,
                }
            )
            if district:
                district_id = str(district["_id"])

        if check_type == "units_without_villages":
            return await self._find_units_without_villages(
                context, district_id, unit_id, page, page_size, skip
            )
        else:
            return await self._find_personnel_in_unmapped_units(
                context, district_id, unit_id, page, page_size, skip
            )

    async def _find_units_without_villages(
        self,
        context: UserContext,
        district_id: Optional[str],
        unit_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Find units that have no village mappings"""

        # Get all unit IDs that have village mappings
        units_with_villages = await self.db[Collections.UNIT_VILLAGES].distinct(
            "unitId", {"isDelete": False}
        )

        # Build match for units without mappings
        unit_match: Dict[str, Any] = {
            "isDelete": False,
            "_id": {"$nin": units_with_villages},
        }

        if district_id and ObjectId.is_valid(district_id):
            unit_match["districtId"] = ObjectId(district_id)

        if unit_id and ObjectId.is_valid(unit_id):
            unit_match["_id"] = ObjectId(unit_id)

        # Apply scope filter
        unit_match = await self.apply_scope_filter(unit_match, context, "unit")

        # Query units
        pipeline = [
            {"$match": unit_match},
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "districtId",
                    "foreignField": "_id",
                    "as": "districtData",
                }
            },
            {
                "$unwind": {
                    "path": "$districtData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$lookup": {
                    "from": Collections.UNIT_TYPE,
                    "localField": "unitTypeId",
                    "foreignField": "_id",
                    "as": "unitTypeData",
                }
            },
            {
                "$unwind": {
                    "path": "$unitTypeData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$addFields": {
                    "personnelCount": {
                        "$size": {"$ifNull": ["$unitPersonnelList", []]}
                    }
                }
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "districtName": "$districtData.name",
                    "unitType": "$unitTypeData.name",
                    "personnelCount": 1,
                    "policeReferenceId": 1,
                }
            },
            {"$sort": {"name": 1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]

        results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(
            length=None
        )
        total = await self.db[Collections.UNIT].count_documents(unit_match)

        # Get summary stats
        total_units = await self.db[Collections.UNIT].count_documents(
            {"isDelete": False}
        )
        units_with_mapping = len(units_with_villages)

        return self.format_success_response(
            query_type="units_without_village_mappings",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "summary": {
                    "totalUnits": total_units,
                    "unitsWithMappings": units_with_mapping,
                    "unitsWithoutMappings": total,
                    "coveragePercentage": round(
                        (units_with_mapping / total_units * 100)
                        if total_units > 0
                        else 0,
                        2,
                    ),
                },
                "district_id": district_id,
            },
        )

    async def _find_personnel_in_unmapped_units(
        self,
        context: UserContext,
        district_id: Optional[str],
        unit_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Find personnel assigned to units that have no village mappings"""

        # Get unit IDs with village mappings
        units_with_villages = await self.db[Collections.UNIT_VILLAGES].distinct(
            "unitId", {"isDelete": False}
        )

        # Get unit IDs without village mappings (within scope)
        unit_query: Dict[str, Any] = {
            "isDelete": False,
            "_id": {"$nin": units_with_villages},
        }

        if district_id and ObjectId.is_valid(district_id):
            unit_query["districtId"] = ObjectId(district_id)

        # Apply scope filter to get accessible unmapped units
        unit_query = await self.apply_scope_filter(unit_query, context, "unit")

        unmapped_unit_ids = await self.db[Collections.UNIT].distinct(
            "_id", unit_query
        )

        if not unmapped_unit_ids:
            return self.format_success_response(
                query_type="personnel_in_unmapped_units",
                data=[],
                total=0,
                page=page,
                page_size=page_size,
                metadata={
                    "message": "All accessible units have village mappings",
                },
            )

        # Build personnel match
        personnel_match: Dict[str, Any] = {
            "isDelete": False,
            "isActive": True,
            "units.unitId": {"$in": unmapped_unit_ids},
        }

        # Apply scope filter
        personnel_match = await self.apply_scope_filter(
            personnel_match, context, "personnel"
        )

        # Query personnel
        pipeline = [
            {"$match": personnel_match},
            {"$unwind": "$units"},
            {"$match": {"units.unitId": {"$in": unmapped_unit_ids}}},
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "units.unitId",
                    "foreignField": "_id",
                    "as": "unitData",
                }
            },
            {
                "$unwind": {
                    "path": "$unitData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$lookup": {
                    "from": Collections.RANK_MASTER,
                    "localField": "rankId",
                    "foreignField": "_id",
                    "as": "rankData",
                }
            },
            {
                "$unwind": {
                    "path": "$rankData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "userId": 1,
                    "badgeNo": 1,
                    "rankName": "$rankData.name",
                    "unitId": {"$toString": "$units.unitId"},
                    "unitName": "$unitData.name",
                }
            },
            {"$sort": {"unitName": 1, "name": 1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total count
        total = await self.db[Collections.PERSONNEL_MASTER].count_documents(
            personnel_match
        )

        return self.format_success_response(
            query_type="personnel_in_unmapped_units",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "unmappedUnitsCount": len(unmapped_unit_ids),
                "district_id": district_id,
            },
        )


class GetVillageCoverageTool(BaseTool):
    """Get village coverage statistics for units"""

    name = "get_village_coverage"
    description = (
        "Get statistics on village coverage for units in a district, "
        "showing how many villages each unit covers."
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
                "unit_id": {
                    "type": "string",
                    "description": "Get coverage for specific unit",
                },
                "unit_name": {
                    "type": "string",
                    "description": "Unit name (case insensitive)",
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
        unit_id = arguments.get("unit_id")
        unit_name = arguments.get("unit_name")
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve unit name to ID
        if not unit_id and unit_name:
            from mcp.query_builder.builder import SafeQueryBuilder
            escaped_name = SafeQueryBuilder.escape_regex(unit_name)
            unit = await self.db[Collections.UNIT].find_one(
                {
                    "name": {"$regex": escaped_name, "$options": "i"},
                    "isDelete": False,
                }
            )
            if unit:
                unit_id = str(unit["_id"])
            else:
                return self.format_error_response(
                    "NOT_FOUND",
                    f"Unit not found: {unit_name}",
                )

        # Resolve district name
        if not district_id and district_name:
            district = await self.db[Collections.DISTRICT].find_one(
                {
                    "name": {"$regex": f"^{district_name}$", "$options": "i"},
                    "isDelete": False,
                }
            )
            if district:
                district_id = str(district["_id"])

        # Build query
        village_match: Dict[str, Any] = {"isDelete": False}

        if unit_id and ObjectId.is_valid(unit_id):
            village_match["unitId"] = ObjectId(unit_id)

        # Get units in district for filtering
        if district_id and ObjectId.is_valid(district_id):
            district_unit_query = await self.apply_scope_filter(
                {"districtId": ObjectId(district_id), "isDelete": False},
                context,
                "unit",
            )
            unit_ids = await self.db[Collections.UNIT].distinct(
                "_id",
                district_unit_query,
            )
            if "unitId" not in village_match:
                village_match["unitId"] = {"$in": unit_ids}

        village_match = await self.apply_scope_filter(
            village_match,
            context,
            "unit_villages",
        )

        # Group by unit
        pipeline = [
            {"$match": village_match},
            {
                "$group": {
                    "_id": "$unitId",
                    "villageCount": {"$sum": 1},
                    "villages": {"$push": "$villageName"},
                    "mandals": {"$addToSet": "$mandalId"},
                }
            },
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "unitData",
                }
            },
            {
                "$unwind": {
                    "path": "$unitData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "unitData.districtId",
                    "foreignField": "_id",
                    "as": "districtData",
                }
            },
            {
                "$unwind": {
                    "path": "$districtData",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$project": {
                    "unitId": {"$toString": "$_id"},
                    "unitName": "$unitData.name",
                    "districtName": "$districtData.name",
                    "villageCount": 1,
                    "mandalCount": {"$size": "$mandals"},
                    "villages": {"$slice": ["$villages", 10]},
                    "_id": 0,
                }
            },
            {"$sort": {"villageCount": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]

        results = await self.db[Collections.UNIT_VILLAGES].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total
        count_pipeline = [
            {"$match": village_match},
            {"$group": {"_id": "$unitId"}},
            {"$count": "total"},
        ]
        count_result = await self.db[Collections.UNIT_VILLAGES].aggregate(
            count_pipeline
        ).to_list(length=1)
        total = count_result[0]["total"] if count_result else len(results)

        # Get overall stats
        total_villages = await self.db[Collections.UNIT_VILLAGES].count_documents(
            village_match
        )

        return self.format_success_response(
            query_type="village_coverage",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "totalVillages": total_villages,
                "unitsWithCoverage": total,
                "district_id": district_id,
            },
        )

