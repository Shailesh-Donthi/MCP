"""
Vacancy Analysis Tools for MCP

Tools for analyzing staffing and vacancies across units and ranks.
"""

from typing import Any, Dict, List, Optional
from bson import ObjectId

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.query_builder.aggregation_builder import AggregationBuilder
from mcp.constants import Collections


class CountVacanciesByUnitRankTool(BaseTool):
    """Count personnel and analyze potential vacancies by unit and rank"""

    name = "count_vacancies_by_unit_rank"
    description = (
        "Analyze personnel distribution by unit and rank. Shows current "
        "personnel counts which can be compared to sanctioned strength. "
        "Note: Actual vacancy calculation requires sanctioned strength data."
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
                    "description": "Filter by specific unit ID",
                },
                "unit_name": {
                    "type": "string",
                    "description": "Filter by unit name (case insensitive)",
                },
                "rank_id": {
                    "type": "string",
                    "description": "Filter by specific rank ID",
                },
                "show_empty_units": {
                    "type": "boolean",
                    "description": "Include units with zero personnel",
                    "default": False,
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
        rank_id = arguments.get("rank_id")
        show_empty_units = arguments.get("show_empty_units", False)
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve names to IDs
        if not district_id and district_name:
            district_id = await self._resolve_name(
                Collections.DISTRICT, district_name
            )

        if not unit_id and unit_name:
            unit_id = await self._resolve_name(Collections.UNIT, unit_name)

        # Build unit match conditions
        unit_match: Dict[str, Any] = {"isDelete": False}

        if district_id and ObjectId.is_valid(district_id):
            unit_match["districtId"] = ObjectId(district_id)

        if unit_id and ObjectId.is_valid(unit_id):
            unit_match["_id"] = ObjectId(unit_id)

        # Apply scope filter to unit match
        unit_match = await self.apply_scope_filter(unit_match, context, "unit")

        # Get units with personnel counts by rank
        pipeline = [
            {"$match": unit_match},
            # Lookup personnel for each unit
            {
                "$lookup": {
                    "from": Collections.PERSONNEL_MASTER,
                    "let": {"unitId": "$_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {
                                            "$in": [
                                                "$$unitId",
                                                {"$ifNull": ["$units.unitId", []]},
                                            ]
                                        },
                                        {"$eq": ["$isDelete", False]},
                                        {"$eq": ["$isActive", True]},
                                    ]
                                }
                            }
                        },
                        {"$group": {"_id": "$rankId", "count": {"$sum": 1}}},
                    ],
                    "as": "personnelByRank",
                }
            },
            # Lookup district
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
            # Lookup unit type
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
            # Calculate total personnel
            {
                "$addFields": {
                    "totalPersonnel": {"$sum": "$personnelByRank.count"}
                }
            },
        ]

        # Filter out empty units if not requested
        if not show_empty_units:
            pipeline.append({"$match": {"totalPersonnel": {"$gt": 0}}})

        # Filter by rank if specified
        if rank_id and ObjectId.is_valid(rank_id):
            pipeline.append(
                {
                    "$match": {
                        "personnelByRank._id": ObjectId(rank_id)
                    }
                }
            )

        # Project final fields
        pipeline.extend([
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "unitName": "$name",
                    "districtName": "$districtData.name",
                    "unitType": "$unitTypeData.name",
                    "totalPersonnel": 1,
                    "personnelByRank": 1,
                }
            },
            {"$sort": {"totalPersonnel": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ])

        results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(
            length=None
        )

        # Get all ranks for reference
        ranks = await self.db[Collections.RANK_MASTER].find(
            {"isDelete": False}, {"_id": 1, "name": 1, "shortCode": 1}
        ).to_list(length=None)
        ranks_map = {str(r["_id"]): r for r in ranks}

        # Format results with rank names
        formatted_results = []
        total_personnel_all = 0

        for unit in results:
            rank_counts: Dict[str, Any] = {}

            for pr in unit.get("personnelByRank", []):
                rank_id_str = str(pr["_id"]) if pr["_id"] else "unknown"
                rank_info = ranks_map.get(rank_id_str, {})
                rank_counts[rank_id_str] = {
                    "rankName": rank_info.get("name", "Unknown"),
                    "rankShortCode": rank_info.get("shortCode"),
                    "actualCount": pr["count"],
                }

            total = unit.get("totalPersonnel", 0)
            total_personnel_all += total

            formatted_results.append({
                "unitId": unit["_id"],
                "unitName": unit["unitName"],
                "districtName": unit.get("districtName"),
                "unitType": unit.get("unitType"),
                "totalPersonnel": total,
                "byRank": rank_counts,
            })

        # Get total unit count
        count_pipeline = pipeline[:-3]  # Remove skip, limit, and project
        count_pipeline.append({"$count": "total"})
        count_result = await self.db[Collections.UNIT].aggregate(
            count_pipeline
        ).to_list(length=1)
        total_units = count_result[0]["total"] if count_result else len(formatted_results)

        return self.format_success_response(
            query_type="vacancies_by_unit_rank",
            data={
                "units": formatted_results,
                "summary": {
                    "totalUnits": total_units,
                    "totalPersonnel": total_personnel_all,
                },
            },
            total=total_units,
            page=page,
            page_size=page_size,
            metadata={
                "district_id": district_id,
                "unit_id": unit_id,
                "note": "Vacancy calculation requires sanctioned strength data",
            },
        )

    async def _resolve_name(
        self, collection: str, name: str
    ) -> Optional[str]:
        """Resolve a name to an ID"""
        doc = await self.db[collection].find_one(
            {
                "name": {"$regex": f"^{name}$", "$options": "i"},
                "isDelete": False,
            }
        )
        return str(doc["_id"]) if doc else None


class GetPersonnelDistributionTool(BaseTool):
    """Get overall personnel distribution across ranks and units"""

    name = "get_personnel_distribution"
    description = (
        "Get summary statistics of personnel distribution across "
        "ranks and unit types within accessible scope."
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
                "unit_name": {
                    "type": "string",
                    "description": "Filter by unit name (case insensitive)",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["rank", "unit_type", "district"],
                    "description": "How to group the distribution",
                    "default": "rank",
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
        unit_name = arguments.get("unit_name")
        group_by = arguments.get("group_by", "rank")

        # Resolve district name to ID
        if not district_id and district_name:
            district = await self.db[Collections.DISTRICT].find_one(
                {
                    "name": {"$regex": f"^{district_name}$", "$options": "i"},
                    "isDelete": False,
                }
            )
            if district:
                district_id = str(district["_id"])

        # Resolve unit name to get its district
        if not district_id and unit_name:
            from mcp.query_builder.builder import SafeQueryBuilder
            escaped_name = SafeQueryBuilder.escape_regex(unit_name)
            unit = await self.db[Collections.UNIT].find_one(
                {
                    "name": {"$regex": escaped_name, "$options": "i"},
                    "isDelete": False,
                }
            )
            if unit and unit.get("districtId"):
                district_id = str(unit["districtId"])

        # Build base query
        base_query: Dict[str, Any] = {
            "isDelete": False,
            "isActive": True,
        }

        # Apply scope filter
        query = await self.apply_scope_filter(base_query, context, "personnel")

        # If district filter, resolve personnel through unit_master.unitPersonnelList.
        # Some datasets do not store units on personnel documents.
        if district_id and ObjectId.is_valid(district_id):
            personnel_ids = await self._get_personnel_ids_in_district(district_id)
            if personnel_ids:
                query["_id"] = {"$in": personnel_ids}
            else:
                # Explicitly force no-match to avoid returning statewide totals
                query["_id"] = {"$exists": False}

        if group_by == "rank":
            return await self._group_by_rank(query, district_id)
        elif group_by == "unit_type":
            return await self._group_by_unit_type(query, district_id)
        elif group_by == "district":
            return await self._group_by_district(query)
        else:
            return self.format_error_response(
                "INVALID_PARAMETER",
                f"Invalid group_by value: {group_by}",
            )

    async def _get_units_in_district(self, district_id: str) -> List[ObjectId]:
        """Get unit IDs in a district"""
        cursor = self.db[Collections.UNIT].find(
            {"districtId": ObjectId(district_id), "isDelete": False},
            {"_id": 1},
        )
        units = await cursor.to_list(length=None)
        return [u["_id"] for u in units]

    async def _get_personnel_ids_in_district(self, district_id: str) -> List[ObjectId]:
        """Get personnel ObjectIds mapped to units in a district."""
        try:
            return await self.db[Collections.UNIT].distinct(
                "unitPersonnelList",
                {"districtId": ObjectId(district_id), "isDelete": False},
            )
        except Exception:
            return []

    async def _group_by_rank(
        self, query: dict, district_id: Optional[str]
    ) -> Dict[str, Any]:
        """Group personnel by rank"""
        pipeline = [
            {"$match": query},
            {"$group": {"_id": "$rankId", "count": {"$sum": 1}}},
            {
                "$lookup": {
                    "from": Collections.RANK_MASTER,
                    "localField": "_id",
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
                    "rankId": {"$toString": "$_id"},
                    "rankName": "$rankData.name",
                    "rankShortCode": "$rankData.shortCode",
                    "count": 1,
                    "_id": 0,
                }
            },
            {"$sort": {"count": -1}},
        ]

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        total = sum(r["count"] for r in results)

        return self.format_success_response(
            query_type="personnel_distribution_by_rank",
            data={
                "distribution": results,
                "total": total,
            },
            metadata={"district_id": district_id},
        )

    async def _group_by_unit_type(
        self, query: dict, district_id: Optional[str]
    ) -> Dict[str, Any]:
        """Group personnel by unit type"""
        pipeline = [
            {"$match": query},
            {"$unwind": "$units"},
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "units.unitId",
                    "foreignField": "_id",
                    "as": "unitData",
                }
            },
            {"$unwind": "$unitData"},
            {"$group": {"_id": "$unitData.unitTypeId", "count": {"$sum": 1}}},
            {
                "$lookup": {
                    "from": Collections.UNIT_TYPE,
                    "localField": "_id",
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
                "$project": {
                    "unitTypeId": {"$toString": "$_id"},
                    "unitTypeName": "$unitTypeData.name",
                    "count": 1,
                    "_id": 0,
                }
            },
            {"$sort": {"count": -1}},
        ]

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        total = sum(r["count"] for r in results)

        return self.format_success_response(
            query_type="personnel_distribution_by_unit_type",
            data={
                "distribution": results,
                "total": total,
            },
            metadata={"district_id": district_id},
        )

    async def _group_by_district(self, query: dict) -> Dict[str, Any]:
        """Group personnel by district"""
        pipeline = [
            {"$match": query},
            {"$unwind": "$units"},
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "units.unitId",
                    "foreignField": "_id",
                    "as": "unitData",
                }
            },
            {"$unwind": "$unitData"},
            {"$group": {"_id": "$unitData.districtId", "count": {"$sum": 1}}},
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "_id",
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
                    "districtId": {"$toString": "$_id"},
                    "districtName": "$districtData.name",
                    "count": 1,
                    "_id": 0,
                }
            },
            {"$sort": {"count": -1}},
        ]

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        total = sum(r["count"] for r in results)

        return self.format_success_response(
            query_type="personnel_distribution_by_district",
            data={
                "distribution": results,
                "total": total,
            },
        )

