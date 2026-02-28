"""
Personnel Query Tools for MCP

Tools for querying personnel data with various filters and groupings.
"""

from typing import Any, Dict, List, Optional
from bson import ObjectId
import re

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.query_builder.aggregation_builder import AggregationBuilder, AggregationHelpers
from mcp.utils.formatters import format_personnel_list, stringify_object_ids
from mcp.constants import Collections
from mcp.router.extractors import normalize_common_entity_aliases, fuzzy_best_match
from mcp.query_builder.builder import SafeQueryBuilder


class QueryPersonnelByUnitTool(BaseTool):
    """Query personnel in a specific unit, optionally grouped by rank"""

    name = "query_personnel_by_unit"
    description = (
        "Show all personnel in a specific unit. Can filter by rank and "
        "optionally group results by rank. Supports pagination."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "Unit ID to query personnel for",
                },
                "unit_name": {
                    "type": "string",
                    "description": "Unit name (if ID not known) - case insensitive search",
                },
                "group_by_rank": {
                    "type": "boolean",
                    "description": "Whether to group results by rank",
                    "default": False,
                },
                "rank_id": {
                    "type": "string",
                    "description": "Filter by specific rank ID",
                },
                "include_inactive": {
                    "type": "boolean",
                    "description": "Include inactive personnel",
                    "default": False,
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "description": "Results per page (max 100)",
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
        unit_id = arguments.get("unit_id")
        unit_name = arguments.get("unit_name")
        group_by_rank = arguments.get("group_by_rank", False)
        rank_id = arguments.get("rank_id")
        include_inactive = arguments.get("include_inactive", False)
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve unit_name to unit_id if needed
        if not unit_id and unit_name:
            unit_id = await self._resolve_unit_name(unit_name)

        if not unit_id:
            if unit_name:
                return self.format_error_response(
                    "NOT_FOUND",
                    f"Unit not found: {unit_name}",
                    {"hint": "Provide an exact unit name or unit_id."},
                )
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Unit ID or name is required",
                {"hint": "Provide either unit_id or unit_name parameter"},
            )
        if not ObjectId.is_valid(unit_id):
            return self.format_error_response(
                "VALIDATION_ERROR",
                f"Invalid unit id: {unit_id}",
            )

        unit_oid = ObjectId(unit_id)

        # Scope guard for non-state users before executing unit-scoped lookups.
        if not context.has_state_access:
            accessible_units = await self.scope_filter.get_accessible_units(context)
            if unit_oid not in set(accessible_units):
                return self.format_success_response(
                    query_type="personnel_by_unit_grouped_by_rank" if group_by_rank else "personnel_by_unit",
                    data={"total_personnel": 0, "groups": []} if group_by_rank else [],
                    total=0,
                    page=page,
                    page_size=page_size,
                    metadata={"unit_id": unit_id, "unit_name": unit_name},
                )

        base_query = self._build_personnel_base_query(include_inactive, rank_id)

        # Prefer legacy path when personnel.units is populated for this unit.
        # Otherwise fall back to assignment_master mapping (current DB shape).
        legacy_probe = dict(base_query)
        legacy_probe["units.unitId"] = unit_oid
        legacy_count = await self.db[Collections.PERSONNEL_MASTER].count_documents(legacy_probe)
        assignment_count = await self.db[Collections.ASSIGNMENT_MASTER].count_documents(
            {"unitId": unit_oid, "isDelete": False, "isActive": True}
        )

        if legacy_count == 0 and assignment_count > 0:
            if group_by_rank:
                return await self._execute_grouped_by_rank_with_assignments(
                    base_query, unit_id, page, page_size, skip
                )
            return await self._execute_flat_with_assignments(
                base_query, unit_id, page, page_size, skip
            )

        # Legacy personnel.units path
        legacy_query = dict(base_query)
        legacy_query["units.unitId"] = unit_oid
        query = await self.apply_scope_filter(legacy_query, context, "personnel")

        if group_by_rank:
            return await self._execute_grouped_by_rank(
                query, unit_id, page, page_size, skip
            )
        else:
            return await self._execute_flat(query, unit_id, page, page_size, skip)

    def _build_personnel_base_query(
        self,
        include_inactive: bool,
        rank_id: Optional[str],
    ) -> Dict[str, Any]:
        base_query: Dict[str, Any] = {"isDelete": False}
        if not include_inactive:
            base_query["isActive"] = True
        if rank_id and ObjectId.is_valid(rank_id):
            base_query["rankId"] = ObjectId(rank_id)
        return base_query

    async def _get_unit_name(self, unit_id: str) -> Optional[str]:
        unit_doc = await self.db[Collections.UNIT].find_one(
            {"_id": ObjectId(unit_id)}, {"name": 1}
        )
        return unit_doc.get("name") if unit_doc else None

    def _format_rank_groups(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for row in results:
            groups.append(
                {
                    "rank_id": str(row["_id"]) if row.get("_id") else None,
                    "rank_name": row.get("rankName", "Unknown"),
                    "rank_short_code": row.get("rankShortCode"),
                    "count": row.get("count", 0),
                    "personnel": row.get("personnel", [])[:10],
                }
            )
        return groups

    def _build_assignment_lookup_stage(self) -> Dict[str, Any]:
        return {
            "$lookup": {
                "from": Collections.ASSIGNMENT_MASTER,
                "localField": "_id",
                "foreignField": "userId",
                "as": "assignmentData",
            }
        }

    def _build_unit_assignment_match_stage(self, unit_oid: ObjectId) -> Dict[str, Any]:
        return {
            "$match": {
                "assignmentData.unitId": unit_oid,
                "assignmentData.isDelete": False,
                "assignmentData.isActive": True,
            }
        }

    async def _resolve_unit_name(self, unit_name: str) -> Optional[str]:
        """Resolve unit name with exact + tolerant matching (aligned with command-history tool)."""
        raw = normalize_common_entity_aliases((unit_name or "").strip())
        if not raw:
            return None
        raw = re.sub(r"\bSPDO\b", "SDPO", raw, flags=re.IGNORECASE)

        def normalize_spacing(value: str) -> str:
            v = re.sub(r"\s+", " ", value).strip()
            v = re.sub(r"\s*,\s*", ", ", v)
            v = re.sub(r"\s*\(\s*", " (", v)
            v = re.sub(r"\s*\)\s*", ")", v)
            v = re.sub(r"\s+", " ", v).strip()
            return v

        # 1) Exact case-insensitive match (including punctuation-spacing variants)
        exact_variants = [raw, normalize_spacing(raw), raw.replace(", ", ","), raw.replace(",", ", ")]
        seen_exact: set[str] = set()
        for candidate in exact_variants:
            key = candidate.lower().strip()
            if not key or key in seen_exact:
                continue
            seen_exact.add(key)
            exact = await self.db[Collections.UNIT].find_one(
                {
                    "name": {"$regex": f"^{re.escape(candidate)}$", "$options": "i"},
                    "isDelete": False,
                },
                {"_id": 1},
            )
            if exact:
                return str(exact["_id"])

        # 2) Common suffix expansions for station queries
        variants = [raw, normalize_spacing(raw), raw.replace(", ", ","), raw.replace(",", ", ")]
        if not re.search(r"\b(ps|police\s+station|station)\b", raw, re.IGNORECASE):
            variants.extend([f"{raw} PS", f"{raw} Police Station", f"{raw} Station"])
        normalized_variants: List[str] = []
        for variant in variants:
            nv = normalize_spacing(variant)
            if nv and nv.lower() not in {x.lower() for x in normalized_variants}:
                normalized_variants.append(nv)
        variants = normalized_variants
        for name in variants:
            doc = await self.db[Collections.UNIT].find_one(
                {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}, "isDelete": False},
                {"_id": 1},
            )
            if doc:
                return str(doc["_id"])

        # 3) Contains-all-tokens fallback with lightweight scoring
        cleaned = re.sub(r"\b(police\s+station|station|ps)\b", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip() or re.sub(r"[^a-z0-9\s]", " ", raw.lower()).strip()
        tokens = [re.escape(t) for t in re.findall(r"[a-z0-9]+", cleaned) if t]
        if not tokens:
            return None
        pattern = "".join([rf"(?=.*{t})" for t in tokens]) + r".*"
        docs = await self.db[Collections.UNIT].find(
            {"name": {"$regex": pattern, "$options": "i"}, "isDelete": False},
            {"_id": 1, "name": 1},
        ).limit(20).to_list(length=20)
        if not docs:
            # 4) Fuzzy fallback against available unit names
            unit_names = await self.db[Collections.UNIT].distinct("name", {"isDelete": False})
            best = fuzzy_best_match(raw, unit_names, cutoff=0.74)
            if best:
                best_doc = await self.db[Collections.UNIT].find_one(
                    {"name": {"$regex": f"^{re.escape(best)}$", "$options": "i"}, "isDelete": False},
                    {"_id": 1},
                )
                if best_doc:
                    return str(best_doc["_id"])
            return None

        cleaned_lower = cleaned.lower()

        def score(doc: Dict[str, Any]) -> int:
            name = re.sub(r"[^a-z0-9\s]", " ", str(doc.get("name", "")).lower())
            name = re.sub(r"\s+", " ", name).strip()
            s = 0
            if name.startswith(cleaned_lower):
                s += 6
            if re.search(r"\b(ps|ups|police station|station)\b", name):
                s += 4
            if cleaned_lower in name:
                s += 2
            overlap = len(set(cleaned_lower.split()) & set(name.split()))
            s += overlap * 2
            s -= min(len(name), 200) // 25
            return s

        best = sorted(docs, key=score, reverse=True)[0]
        return str(best["_id"])

    async def _execute_grouped_by_rank(
        self,
        query: dict,
        unit_id: str,
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Execute query with results grouped by rank"""
        pipeline = (
            AggregationBuilder()
            .match(query)
            .lookup(
                from_collection=Collections.RANK_MASTER,
                local_field="rankId",
                foreign_field="_id",
                as_field="rankData",
            )
            .unwind("rankData")
            .group(
                group_by="$rankId",
                aggregations={
                    "rankName": AggregationHelpers.first("rankData.name"),
                    "rankShortCode": AggregationHelpers.first("rankData.shortCode"),
                    "count": AggregationHelpers.count_docs(),
                    "personnel": {
                        "$push": {
                            "_id": AggregationHelpers.to_string("_id"),
                            "name": "$name",
                            "userId": "$userId",
                            "badgeNo": "$badgeNo",
                            "mobile": "$mobile",
                        }
                    },
                },
            )
            .sort({"count": -1})
            .skip(skip)
            .limit(page_size)
            .build()
        )

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total count of distinct ranks
        count_pipeline = [
            {"$match": query},
            {"$group": {"_id": "$rankId"}},
            {"$count": "total"},
        ]
        count_result = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            count_pipeline
        ).to_list(length=1)
        total_ranks = count_result[0]["total"] if count_result else 0

        # Get total personnel count
        total_personnel = await self.db[Collections.PERSONNEL_MASTER].count_documents(
            query
        )

        unit_name = await self._get_unit_name(unit_id)
        groups = self._format_rank_groups(results)

        return self.format_success_response(
            query_type="personnel_by_unit_grouped_by_rank",
            data={
                "total_personnel": total_personnel,
                "groups": groups,
            },
            total=total_ranks,
            page=page,
            page_size=page_size,
            metadata={
                "unit_id": unit_id,
                "unit_name": unit_name,
            },
        )

    async def _execute_flat(
        self,
        query: dict,
        unit_id: str,
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Execute query with flat results"""
        pipeline = (
            AggregationBuilder()
            .match(query)
            .sort({"name": 1})
            .skip(skip)
            .limit(page_size)
            .lookup(
                from_collection=Collections.RANK_MASTER,
                local_field="rankId",
                foreign_field="_id",
                as_field="rankData",
            )
            .unwind("rankData")
            .project({
                "_id": AggregationHelpers.to_string("_id"),
                "name": 1,
                "userId": 1,
                "badgeNo": 1,
                "mobile": 1,
                "email": 1,
                "rankName": "$rankData.name",
                "rankShortCode": "$rankData.shortCode",
                "isActive": 1,
            })
            .build()
        )

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total count
        total = await self.db[Collections.PERSONNEL_MASTER].count_documents(query)

        unit_name = await self._get_unit_name(unit_id)

        return self.format_success_response(
            query_type="personnel_by_unit",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "unit_id": unit_id,
                "unit_name": unit_name,
            },
        )

    async def _execute_grouped_by_rank_with_assignments(
        self,
        base_query: dict,
        unit_id: str,
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        unit_oid = ObjectId(unit_id)
        pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_unit_assignment_match_stage(unit_oid),
            {
                "$lookup": {
                    "from": Collections.RANK_MASTER,
                    "localField": "rankId",
                    "foreignField": "_id",
                    "as": "rankData",
                }
            },
            {"$unwind": {"path": "$rankData", "preserveNullAndEmptyArrays": True}},
            {
                "$group": {
                    "_id": "$rankId",
                    "rankName": {"$first": "$rankData.name"},
                    "rankShortCode": {"$first": "$rankData.shortCode"},
                    "personnelSet": {
                        "$addToSet": {
                            "_id": {"$toString": "$_id"},
                            "name": "$name",
                            "userId": "$userId",
                            "badgeNo": "$badgeNo",
                            "mobile": "$mobile",
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "rankName": 1,
                    "rankShortCode": 1,
                    "count": {"$size": "$personnelSet"},
                    "personnel": {"$slice": ["$personnelSet", 10]},
                }
            },
            {"$sort": {"count": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]
        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=None)

        count_pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_unit_assignment_match_stage(unit_oid),
            {"$group": {"_id": "$rankId"}},
            {"$count": "total"},
        ]
        count_result = await self.db[Collections.PERSONNEL_MASTER].aggregate(count_pipeline).to_list(length=1)
        total_ranks = count_result[0]["total"] if count_result else 0
        total_personnel = await self._count_distinct_personnel_in_unit_assignments(base_query, unit_oid)

        unit_name = await self._get_unit_name(unit_id)
        groups = self._format_rank_groups(results)

        return self.format_success_response(
            query_type="personnel_by_unit_grouped_by_rank",
            data={"total_personnel": total_personnel, "groups": groups},
            total=total_ranks,
            page=page,
            page_size=page_size,
            metadata={
                "unit_id": unit_id,
                "unit_name": unit_name,
            },
        )

    async def _execute_flat_with_assignments(
        self,
        base_query: dict,
        unit_id: str,
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        unit_oid = ObjectId(unit_id)
        pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_unit_assignment_match_stage(unit_oid),
            {
                "$lookup": {
                    "from": Collections.RANK_MASTER,
                    "localField": "rankId",
                    "foreignField": "_id",
                    "as": "rankData",
                }
            },
            {"$unwind": {"path": "$rankData", "preserveNullAndEmptyArrays": True}},
            {
                "$group": {
                    "_id": "$_id",
                    "name": {"$first": "$name"},
                    "userId": {"$first": "$userId"},
                    "badgeNo": {"$first": "$badgeNo"},
                    "mobile": {"$first": "$mobile"},
                    "email": {"$first": "$email"},
                    "rankName": {"$first": "$rankData.name"},
                    "rankShortCode": {"$first": "$rankData.shortCode"},
                    "isActive": {"$first": "$isActive"},
                }
            },
            {"$sort": {"name": 1}},
            {"$skip": skip},
            {"$limit": page_size},
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "userId": 1,
                    "badgeNo": 1,
                    "mobile": 1,
                    "email": 1,
                    "rankName": 1,
                    "rankShortCode": 1,
                    "isActive": 1,
                }
            },
        ]
        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=None)
        total = await self._count_distinct_personnel_in_unit_assignments(base_query, unit_oid)
        unit_name = await self._get_unit_name(unit_id)
        return self.format_success_response(
            query_type="personnel_by_unit",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "unit_id": unit_id,
                "unit_name": unit_name,
            },
        )

    async def _count_distinct_personnel_in_unit_assignments(
        self,
        base_query: dict,
        unit_oid: ObjectId,
    ) -> int:
        pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_unit_assignment_match_stage(unit_oid),
            {"$group": {"_id": "$_id"}},
            {"$count": "total"},
        ]
        result = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=1)
        return result[0]["total"] if result else 0


class QueryPersonnelByRankTool(BaseTool):
    """Query personnel filtered by rank across units or districts"""

    name = "query_personnel_by_rank"
    description = (
        "Query personnel by rank across all accessible units. "
        "Can filter by district and optionally group by unit."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "rank_id": {
                    "type": "string",
                    "description": "Rank ID to filter by",
                },
                "rank_name": {
                    "type": "string",
                    "description": "Rank name (if ID not known) - case insensitive",
                },
                "rank_relation": {
                    "type": "string",
                    "description": "Rank comparison mode relative to rank_name/rank_id",
                    "enum": ["exact", "above", "below", "at_or_above", "at_or_below"],
                    "default": "exact",
                },
                "district_id": {
                    "type": "string",
                    "description": "Filter by district ID",
                },
                "district_name": {
                    "type": "string",
                    "description": "Filter by district name - case insensitive",
                },
                "district_names": {
                    "type": "array",
                    "description": "Optional list of district names for combined multi-district queries",
                    "items": {"type": "string"},
                },
                "group_by_unit": {
                    "type": "boolean",
                    "description": "Group results by unit",
                    "default": False,
                },
                "include_inactive": {
                    "type": "boolean",
                    "description": "Include inactive personnel",
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
        rank_id = arguments.get("rank_id")
        rank_name = arguments.get("rank_name")
        rank_relation = str(arguments.get("rank_relation") or "exact").strip().lower()
        district_id = arguments.get("district_id")
        district_name = arguments.get("district_name")
        raw_district_names = arguments.get("district_names")
        group_by_unit = arguments.get("group_by_unit", False)
        include_inactive = arguments.get("include_inactive", False)
        page, page_size, skip = self.get_pagination_params(arguments)
        allowed_relations = {"exact", "above", "below", "at_or_above", "at_or_below"}
        if rank_relation not in allowed_relations:
            rank_relation = "exact"

        district_names: List[str] = []
        if isinstance(raw_district_names, list):
            district_names = [
                str(item).strip()
                for item in raw_district_names
                if isinstance(item, str) and item.strip()
            ]
        elif isinstance(raw_district_names, str) and raw_district_names.strip():
            district_names = [
                part.strip()
                for part in re.split(r",|\band\b", raw_district_names, flags=re.IGNORECASE)
                if part and part.strip()
            ]

        if district_names and not district_name and not district_id and len(district_names) == 1:
            district_name = district_names[0]
            district_names = []

        # Resolve rank_name to rank_id if needed
        if not rank_id and rank_name:
            rank_id = await self._resolve_rank_name(rank_name)
            if not rank_id:
                return self.format_error_response(
                    "NOT_FOUND",
                    f"Rank not found: {rank_name}",
                    {"hint": "Try a known rank like Circle Inspector, Sub Inspector, Head Constable, or Police Constable."},
                )

        relation_rank_ids: Optional[List[str]] = None
        if rank_relation != "exact":
            if not rank_id:
                return self.format_error_response(
                    "MISSING_PARAMETER",
                    "rank_name or rank_id is required when using rank_relation",
                )
            relation_rank_ids = await self._resolve_rank_ids_by_relation(rank_id, rank_relation)

        # Resolve district_name to district_id if needed
        if not district_id and district_name:
            district_id = await self._resolve_district_name(district_name)
            if not district_id:
                return self.format_error_response(
                    "NOT_FOUND",
                    f"District not found: {district_name}",
                    {
                        "hint": "Please verify the district name or use 'list districts' to see available values.",
                    },
                )

        # Multi-district rank query (e.g., "Circle Inspectors in Guntur and Chittoor")
        if district_names and len(district_names) >= 2:
            return await self._execute_multi_district_flat(
                base_query=self._build_personnel_base_query(
                    include_inactive,
                    rank_id if rank_relation == "exact" else None,
                    relation_rank_ids,
                ),
                rank_id=rank_id,
                district_names=district_names,
                context=context,
                page=page,
                page_size=page_size,
            )

        # Build effective unit scope for district/context filtering.
        # This DB schema stores person->unit mapping in assignment_master.
        unit_ids = await self._get_effective_unit_ids(context, district_id)

        # Build base query on personnel records
        base_query = self._build_personnel_base_query(
            include_inactive,
            rank_id if rank_relation == "exact" else None,
            relation_rank_ids,
        )

        # If unit scoping is required (district filter and/or non-state context),
        # use assignment_master mapping instead of personnel.units.
        if unit_ids is not None:
            if not unit_ids:
                return self.format_success_response(
                    query_type="personnel_by_rank_grouped_by_unit" if group_by_unit else "personnel_by_rank",
                    data={"total_personnel": 0, "groups": []} if group_by_unit else [],
                    total=0,
                    page=page,
                    page_size=page_size,
                    metadata={
                        "rank_id": rank_id,
                        "rank_name": await self._get_rank_name(rank_id),
                    },
                )
            if group_by_unit:
                return await self._execute_grouped_by_unit_with_assignments(
                    base_query, unit_ids, rank_id, page, page_size, skip
                )
            return await self._execute_flat_with_assignments(
                base_query, unit_ids, rank_id, page, page_size, skip
            )

        # State-wide query with no unit/district scope can use the legacy fast path.
        query = await self.apply_scope_filter(base_query, context, "personnel")
        if group_by_unit:
            return await self._execute_grouped_by_unit(query, rank_id, page, page_size, skip)
        return await self._execute_flat(query, rank_id, page, page_size, skip)

    def _build_personnel_base_query(
        self,
        include_inactive: bool,
        rank_id: Optional[str],
        rank_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        base_query: Dict[str, Any] = {"isDelete": False}
        if not include_inactive:
            base_query["isActive"] = True
        valid_rank_oids = [
            ObjectId(rid) for rid in (rank_ids or []) if isinstance(rid, str) and ObjectId.is_valid(rid)
        ]
        if valid_rank_oids:
            base_query["rankId"] = {"$in": valid_rank_oids}
        elif rank_id and ObjectId.is_valid(rank_id):
            base_query["rankId"] = ObjectId(rank_id)
        return base_query

    def _build_assignment_lookup_stage(self) -> Dict[str, Any]:
        return {
            "$lookup": {
                "from": Collections.ASSIGNMENT_MASTER,
                "localField": "_id",
                "foreignField": "userId",
                "as": "assignmentData",
            }
        }

    def _build_assignment_match_stage(self, unit_ids: List[ObjectId]) -> Dict[str, Any]:
        return {
            "$match": {
                "assignmentData.isDelete": False,
                "assignmentData.isActive": True,
                "assignmentData.unitId": {"$in": unit_ids},
            }
        }

    async def _get_effective_unit_ids(
        self,
        context: UserContext,
        district_id: Optional[str],
    ) -> Optional[List[ObjectId]]:
        """
        Resolve effective unit IDs combining district filter and access scope.
        Returns:
            - None: no unit scoping needed (state-level with no district filter)
            - []: scoped query with no accessible units
            - [ObjectId, ...]: scoped units
        """
        district_units: Optional[List[ObjectId]] = None
        if district_id and ObjectId.is_valid(district_id):
            district_units = await self._get_units_in_district(district_id)
        elif district_id:
            district_units = []

        scope_units: Optional[List[ObjectId]] = None
        if not context.has_state_access:
            scope_units = await self.scope_filter.get_accessible_units(context)

        if district_units is None and scope_units is None:
            return None
        if district_units is None:
            return scope_units or []
        if scope_units is None:
            return district_units

        scope_set = {u for u in scope_units}
        return [u for u in district_units if u in scope_set]

    async def _get_rank_name(self, rank_id: Optional[str]) -> Optional[str]:
        if not rank_id or not ObjectId.is_valid(rank_id):
            return None
        rank_doc = await self.db[Collections.RANK_MASTER].find_one(
            {"_id": ObjectId(rank_id)}, {"name": 1}
        )
        return rank_doc.get("name") if rank_doc else None

    async def _resolve_rank_name(self, rank_name: str) -> Optional[str]:
        """Resolve rank name to ID with aliases, short-codes, and partial matching."""
        raw = (rank_name or "").strip()
        if not raw:
            return None

        norm = re.sub(r"[\s\-_]+", " ", raw).strip().lower()
        if norm.endswith("s"):
            norm = norm[:-1].strip()

        aliases = {
            "ci": "Circle Inspector",
            "circle inspector": "Circle Inspector",
            "inspector general": "Inspector General of Police",
            "igp": "Inspector General of Police",
            "inspector of police": "Inspector Of Police",
            "iop": "Inspector Of Police",
            "sub inspector": "Sub Inspector",
            "si": "Sub Inspector",
            "assistant sub inspector": "Assistant SubInspector",
            "assistant subinspector": "Assistant SubInspector",
            "asi": "Assistant SubInspector",
            "head constable": "Head Constable",
            "hc": "Head Constable",
            "police constable": "Police Constable",
            "pc": "Police Constable",
            "constable": "Constable",
            "sp": "Superintendent of Police",
            "dsp": "Deputy Superintendent of Police",
            "dysp": "Deputy Superintendent of Police",
        }
        canonical = aliases.get(norm, raw)

        # 1) Exact name match
        rank = await self.db[Collections.RANK_MASTER].find_one(
            {"name": {"$regex": f"^{re.escape(canonical)}$", "$options": "i"}, "isDelete": False}
        )
        if rank:
            return str(rank["_id"])

        # 2) Exact short code match
        rank = await self.db[Collections.RANK_MASTER].find_one(
            {"shortCode": {"$regex": f"^{re.escape(norm.upper())}$", "$options": "i"}, "isDelete": False}
        )
        if rank:
            return str(rank["_id"])

        # 3) Partial name contains match
        escaped = SafeQueryBuilder.escape_regex(canonical)
        rank = await self.db[Collections.RANK_MASTER].find_one(
            {"name": {"$regex": escaped, "$options": "i"}, "isDelete": False}
        )
        if rank:
            return str(rank["_id"])

        return None

    async def _resolve_district_name(self, district_name: str) -> Optional[str]:
        """Resolve district name to ID"""
        district = await self.db[Collections.DISTRICT].find_one(
            {
                "name": {"$regex": f"^{district_name}$", "$options": "i"},
                "isDelete": False,
            }
        )
        return str(district["_id"]) if district else None

    async def _resolve_rank_ids_by_relation(
        self,
        anchor_rank_id: str,
        relation: str,
    ) -> List[str]:
        """Resolve rank IDs using rank_master.level relation against an anchor rank."""
        if not ObjectId.is_valid(anchor_rank_id):
            return []
        anchor = await self.db[Collections.RANK_MASTER].find_one(
            {"_id": ObjectId(anchor_rank_id), "isDelete": False},
            {"_id": 1, "level": 1},
        )
        if not anchor:
            return []
        anchor_level = anchor.get("level")
        if not isinstance(anchor_level, (int, float)):
            return []

        if relation == "above":
            level_clause: Dict[str, Any] = {"$lt": anchor_level}
        elif relation == "below":
            level_clause = {"$gt": anchor_level}
        elif relation == "at_or_above":
            level_clause = {"$lte": anchor_level}
        elif relation == "at_or_below":
            level_clause = {"$gte": anchor_level}
        else:
            return [anchor_rank_id]

        cursor = self.db[Collections.RANK_MASTER].find(
            {"isDelete": False, "level": level_clause},
            {"_id": 1},
        )
        rows = await cursor.to_list(length=None)
        return [str(row["_id"]) for row in rows]

    async def _get_units_in_district(self, district_id: str) -> List[ObjectId]:
        """Get all unit IDs in a district"""
        cursor = self.db[Collections.UNIT].find(
            {"districtId": ObjectId(district_id), "isDelete": False},
            {"_id": 1},
        )
        units = await cursor.to_list(length=None)
        return [u["_id"] for u in units]

    async def _build_rank_metadata(self, rank_id: Optional[str]) -> Dict[str, Optional[str]]:
        return {
            "rank_id": rank_id,
            "rank_name": await self._get_rank_name(rank_id),
        }

    async def _execute_grouped_by_unit(
        self,
        query: dict,
        rank_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Execute query grouped by unit"""
        pipeline = [
            {"$match": query},
            {"$unwind": "$units"},
            {
                "$group": {
                    "_id": "$units.unitId",
                    "count": {"$sum": 1},
                    "personnel": {
                        "$push": {
                            "_id": {"$toString": "$_id"},
                            "name": "$name",
                            "userId": "$userId",
                            "badgeNo": "$badgeNo",
                        }
                    },
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
            {"$unwind": {"path": "$unitData", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "unit_id": {"$toString": "$_id"},
                    "unit_name": "$unitData.name",
                    "count": 1,
                    "personnel": {"$slice": ["$personnel", 10]},
                }
            },
            {"$sort": {"count": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total units count
        count_pipeline = [
            {"$match": query},
            {"$unwind": "$units"},
            {"$group": {"_id": "$units.unitId"}},
            {"$count": "total"},
        ]
        count_result = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            count_pipeline
        ).to_list(length=1)
        total_units = count_result[0]["total"] if count_result else 0

        # Get total personnel count
        total_personnel = await self.db[Collections.PERSONNEL_MASTER].count_documents(
            query
        )

        metadata = await self._build_rank_metadata(rank_id)

        return self.format_success_response(
            query_type="personnel_by_rank_grouped_by_unit",
            data={
                "total_personnel": total_personnel,
                "groups": results,
            },
            total=total_units,
            page=page,
            page_size=page_size,
            metadata=metadata,
        )

    async def _execute_flat(
        self,
        query: dict,
        rank_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        """Execute flat query"""
        pipeline = (
            AggregationBuilder()
            .match(query)
            .sort({"name": 1})
            .skip(skip)
            .limit(page_size)
            .lookup(
                from_collection=Collections.RANK_MASTER,
                local_field="rankId",
                foreign_field="_id",
                as_field="rankData",
            )
            .unwind("rankData")
            .project({
                "_id": {"$toString": "$_id"},
                "name": 1,
                "userId": 1,
                "badgeNo": 1,
                "mobile": 1,
                "email": 1,
                "dateOfBirth": 1,
                "rankName": "$rankData.name",
                "rankShortCode": "$rankData.shortCode",
                "isActive": 1,
            })
            .build()
        )

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        total = await self.db[Collections.PERSONNEL_MASTER].count_documents(query)

        metadata = await self._build_rank_metadata(rank_id)

        return self.format_success_response(
            query_type="personnel_by_rank",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata=metadata,
        )

    async def _execute_multi_district_flat(
        self,
        *,
        base_query: dict,
        rank_id: Optional[str],
        district_names: List[str],
        context: UserContext,
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        combined: List[Dict[str, Any]] = []
        resolved_names: List[str] = []
        missing_names: List[str] = []
        seen_ids: set[str] = set()

        for district_name in district_names:
            district_id = await self._resolve_district_name(district_name)
            if not district_id:
                missing_names.append(district_name)
                continue
            unit_ids = await self._get_effective_unit_ids(context, district_id)
            if not unit_ids:
                resolved_names.append(district_name)
                continue
            district_result = await self._execute_flat_with_assignments(
                dict(base_query), unit_ids, rank_id, 1, self.max_results, 0
            )
            rows = district_result.get("data", []) if isinstance(district_result, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    person_id = str(row.get("_id") or "")
                    if person_id and person_id in seen_ids:
                        continue
                    if person_id:
                        seen_ids.add(person_id)
                    if not row.get("districtName"):
                        row["districtName"] = district_name
                    combined.append(row)
            resolved_names.append(district_name)

        if not combined and missing_names and not resolved_names:
            return self.format_error_response(
                "NOT_FOUND",
                f"District not found: {', '.join(missing_names)}",
                {"missing_district_names": missing_names},
            )

        combined.sort(key=lambda row: (str(row.get("name") or "").lower(), str(row.get("userId") or "")))
        total = len(combined)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        page_rows = combined[start:end]

        metadata = await self._build_rank_metadata(rank_id)
        metadata.update(
            {
                "district_names": resolved_names,
                "missing_district_names": missing_names,
            }
        )

        return self.format_success_response(
            query_type="personnel_by_rank",
            data=page_rows,
            total=total,
            page=page,
            page_size=page_size,
            metadata=metadata,
        )

    async def _execute_grouped_by_unit_with_assignments(
        self,
        base_query: dict,
        unit_ids: List[ObjectId],
        rank_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_assignment_match_stage(unit_ids),
            {
                "$group": {
                    "_id": "$assignmentData.unitId",
                    "count": {"$sum": 1},
                    "personnel": {
                        "$addToSet": {
                            "_id": {"$toString": "$_id"},
                            "name": "$name",
                            "userId": "$userId",
                            "badgeNo": "$badgeNo",
                        }
                    },
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
            {"$unwind": {"path": "$unitData", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "unit_id": {"$toString": "$_id"},
                    "unit_name": "$unitData.name",
                    "count": 1,
                    "personnel": {"$slice": ["$personnel", 10]},
                }
            },
            {"$sort": {"count": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]
        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=None)

        count_pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_assignment_match_stage(unit_ids),
            {"$group": {"_id": "$assignmentData.unitId"}},
            {"$count": "total"},
        ]
        count_result = await self.db[Collections.PERSONNEL_MASTER].aggregate(count_pipeline).to_list(length=1)
        total_units = count_result[0]["total"] if count_result else 0

        total_personnel = await self._count_distinct_personnel_with_assignments(base_query, unit_ids)

        return self.format_success_response(
            query_type="personnel_by_rank_grouped_by_unit",
            data={"total_personnel": total_personnel, "groups": results},
            total=total_units,
            page=page,
            page_size=page_size,
            metadata=await self._build_rank_metadata(rank_id),
        )

    async def _execute_flat_with_assignments(
        self,
        base_query: dict,
        unit_ids: List[ObjectId],
        rank_id: Optional[str],
        page: int,
        page_size: int,
        skip: int,
    ) -> Dict[str, Any]:
        pipeline = [
            {"$match": base_query},
            {
                "$lookup": {
                    "from": Collections.RANK_MASTER,
                    "localField": "rankId",
                    "foreignField": "_id",
                    "as": "rankData",
                }
            },
            {"$unwind": "$rankData"},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_assignment_match_stage(unit_ids),
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "assignmentData.unitId",
                    "foreignField": "_id",
                    "as": "unitData",
                }
            },
            {"$unwind": {"path": "$unitData", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "unitData.districtId",
                    "foreignField": "_id",
                    "as": "districtData",
                }
            },
            {"$unwind": {"path": "$districtData", "preserveNullAndEmptyArrays": True}},
            {
                "$group": {
                    "_id": "$_id",
                    "name": {"$first": "$name"},
                    "userId": {"$first": "$userId"},
                    "badgeNo": {"$first": "$badgeNo"},
                    "mobile": {"$first": "$mobile"},
                    "email": {"$first": "$email"},
                    "dateOfBirth": {"$first": "$dateOfBirth"},
                    "rankName": {"$first": "$rankData.name"},
                    "rankShortCode": {"$first": "$rankData.shortCode"},
                    "unitName": {"$first": "$unitData.name"},
                    "districtName": {"$first": "$districtData.name"},
                    "isActive": {"$first": "$isActive"},
                }
            },
            {"$sort": {"name": 1}},
            {"$skip": skip},
            {"$limit": page_size},
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "userId": 1,
                    "badgeNo": 1,
                    "mobile": 1,
                    "email": 1,
                    "dateOfBirth": 1,
                    "rankName": 1,
                    "rankShortCode": 1,
                    "unitName": 1,
                    "districtName": 1,
                    "isActive": 1,
                }
            },
        ]
        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=None)
        total = await self._count_distinct_personnel_with_assignments(base_query, unit_ids)

        return self.format_success_response(
            query_type="personnel_by_rank",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata=await self._build_rank_metadata(rank_id),
        )

    async def _count_distinct_personnel_with_assignments(
        self,
        base_query: dict,
        unit_ids: List[ObjectId],
    ) -> int:
        pipeline = [
            {"$match": base_query},
            self._build_assignment_lookup_stage(),
            {"$unwind": "$assignmentData"},
            self._build_assignment_match_stage(unit_ids),
            {"$group": {"_id": "$_id"}},
            {"$count": "total"},
        ]
        result = await self.db[Collections.PERSONNEL_MASTER].aggregate(pipeline).to_list(length=1)
        return result[0]["total"] if result else 0
