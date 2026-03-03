"""V2 relationship-aware search tools for personnel/unit/assignment queries."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bson import ObjectId

from mcp.constants import Collections
from mcp.router.extractors import fuzzy_best_match, normalize_common_entity_aliases
from mcp.schemas.context_schema import UserContext
from mcp.tools.base_tool import BaseTool
from mcp.v2.repositories import AssignmentRepository, PersonnelRepository, ScopeContext, UnitRepository


class SearchPersonnelTool(BaseTool):
    """Search personnel with relationship enrichment enabled by default."""

    name = "search_personnel"
    description = (
        "Search for a person by name, userId, badge number, mobile, or email. "
        "Returns enriched person details with assignments, unit, district, rank, "
        "and designation context."
    )
    _PERSON_TITLE_PREFIX_RE = re.compile(
        r"^\s*(?:mr|mrs|ms|miss|dr|sri|shri|smt|mt)\.?\s+",
        re.IGNORECASE,
    )

    def __init__(self, db):
        super().__init__(db)
        self._repo = PersonnelRepository(db)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person name"},
                "user_id": {"type": "string", "description": "Police User ID"},
                "badge_no": {"type": "string", "description": "Badge number"},
                "mobile": {"type": "string", "description": "Mobile number"},
                "email": {"type": "string", "description": "Email address"},
                "designation_name": {"type": "string", "description": "Designation name"},
                "district_name": {"type": "string", "description": "District name"},
                "include_inactive": {"type": "boolean", "default": False},
                "page": {"type": "integer", "default": 1},
                "page_size": {"type": "integer", "default": 20},
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        name = arguments.get("name")
        user_id = arguments.get("user_id")
        badge_no = arguments.get("badge_no")
        mobile = arguments.get("mobile")
        email = arguments.get("email")
        designation_name = arguments.get("designation_name")
        district_name = arguments.get("district_name")
        include_inactive = bool(arguments.get("include_inactive", False))
        page, page_size, _ = self.get_pagination_params(arguments)

        if not any([name, user_id, badge_no, mobile, email, designation_name, district_name]):
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide at least one search parameter: name, user_id, badge_no, mobile, email, designation_name, or district_name",
            )

        designation_filter_text = self._normalize_label(designation_name)
        designation_id = await self._resolve_designation_id(designation_name)
        designation_resolution = "resolved" if designation_id is not None else ("unresolved" if designation_name else None)

        district_id = await self._resolve_district_id(district_name)
        if district_name and district_id is None:
            return self.format_error_response(
                "NOT_FOUND",
                f"District not found: {district_name}",
                {"hint": "Try a valid district value (for example: Guntur)."},
            )
        unit_ids_in_district: Optional[List[Any]] = None
        if district_id:
            unit_ids_in_district = await self._resolve_unit_ids_for_district(district_id)
            if not unit_ids_in_district:
                return self.format_success_response(
                    query_type="search_personnel",
                    data=[],
                    total=0,
                    page=page,
                    page_size=page_size,
                    metadata={"search_params": self._non_empty_params({"district_name": district_name})},
                )

        # When filters depend on enrichment (district-unit or designation text/id),
        # pull a wider first page and post-filter deterministically.
        needs_post_filter = bool(unit_ids_in_district or designation_name)
        if needs_post_filter:
            repo_page = 1
            repo_page_size = min(self.max_results, max(page * page_size * 3, 200))
        else:
            repo_page = page
            repo_page_size = page_size

        scope_context = ScopeContext.from_user_context(context)
        result = await self._repo.search(
            name=name,
            user_id=user_id,
            badge_no=badge_no,
            mobile=mobile,
            email=email,
            designation_id=None,
            include_inactive=include_inactive,
            page=repo_page,
            page_size=repo_page_size,
            scope_context=scope_context,
        )

        rows = result.get("data", [])
        rows = self._filter_rows_by_unit_ids(rows, unit_ids_in_district)
        if designation_name:
            rows = [
                row
                for row in rows
                if self._row_matches_designation(
                    row,
                    designation_id=designation_id,
                    designation_text=designation_filter_text,
                )
            ]

        if needs_post_filter:
            total = len(rows)
            start = max(0, (page - 1) * page_size)
            end = start + page_size
            rows = rows[start:end]
        else:
            pagination = result.get("pagination", {})
            total = int(pagination.get("total", 0) or 0)

        if (
            total == 0
            and isinstance(name, str)
            and name.strip()
            and not any([user_id, badge_no, mobile, email])
        ):
            best_name = await self._resolve_best_person_name(
                name,
                include_inactive=include_inactive,
            )
            if best_name and best_name.strip().lower() != name.strip().lower():
                retry = await self._repo.search(
                    name=best_name,
                    user_id=user_id,
                    badge_no=badge_no,
                    mobile=mobile,
                    email=email,
                    designation_id=None,
                    include_inactive=include_inactive,
                    page=repo_page,
                    page_size=repo_page_size,
                    scope_context=scope_context,
                )
                rows = retry.get("data", [])
                rows = self._filter_rows_by_unit_ids(rows, unit_ids_in_district)
                if designation_name:
                    rows = [
                        row
                        for row in rows
                        if self._row_matches_designation(
                            row,
                            designation_id=designation_id,
                            designation_text=designation_filter_text,
                        )
                    ]
                if needs_post_filter:
                    total = len(rows)
                    start = max(0, (page - 1) * page_size)
                    end = start + page_size
                    rows = rows[start:end]
                else:
                    pagination = retry.get("pagination", {})
                    total = int(pagination.get("total", 0) or 0)

        formatted = [
            self._normalize_person_row(row)
            for row in rows
            if isinstance(row, dict)
        ]

        return self.format_success_response(
            query_type="search_personnel",
            data=formatted,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "search_params": self._non_empty_params(
                    {
                        "name": name,
                        "user_id": user_id,
                        "badge_no": badge_no,
                        "mobile": mobile,
                        "email": email,
                        "designation_name": designation_name,
                        "district_name": district_name,
                    }
                ),
                "designation_resolution": designation_resolution,
            },
        )

    async def _resolve_designation_id(self, designation_name: Optional[str]) -> Optional[Any]:
        if not designation_name:
            return None
        normalized = normalize_common_entity_aliases(str(designation_name).strip())
        if not normalized:
            return None
        normalized = re.sub(r"\bSPDO\b", "SDPO", normalized, flags=re.IGNORECASE)
        designation = await self.db[Collections.DESIGNATION_MASTER].find_one(
            {
                "name": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"},
                "isDelete": False,
            },
            {"_id": 1},
        )
        if designation:
            return designation["_id"]
        names = await self.db[Collections.DESIGNATION_MASTER].distinct("name", {"isDelete": False})
        best = fuzzy_best_match(normalized, names, cutoff=0.76)
        if not best:
            return None
        designation = await self.db[Collections.DESIGNATION_MASTER].find_one(
            {
                "name": {"$regex": f"^{re.escape(best)}$", "$options": "i"},
                "isDelete": False,
            },
            {"_id": 1},
        )
        return designation["_id"] if designation else None

    async def _resolve_district_id(self, district_name: Optional[str]) -> Optional[str]:
        if not district_name:
            return None
        normalized = normalize_common_entity_aliases(str(district_name).strip())
        if not normalized:
            return None
        district = await self.db[Collections.DISTRICT].find_one(
            {"name": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}, "isDelete": False},
            {"_id": 1},
        )
        if district:
            return str(district["_id"])
        names = await self.db[Collections.DISTRICT].distinct("name", {"isDelete": False})
        best = fuzzy_best_match(normalized, names, cutoff=0.76)
        if not best:
            return None
        district = await self.db[Collections.DISTRICT].find_one(
            {"name": {"$regex": f"^{re.escape(best)}$", "$options": "i"}, "isDelete": False},
            {"_id": 1},
        )
        return str(district["_id"]) if district else None

    async def _resolve_unit_ids_for_district(self, district_id: str) -> List[ObjectId]:
        if not ObjectId.is_valid(district_id):
            return []
        rows = await self.db[Collections.UNIT].find(
            {"districtId": ObjectId(district_id), "isDelete": False},
            {"_id": 1},
        ).to_list(length=None)
        return [row["_id"] for row in rows if isinstance(row, dict) and isinstance(row.get("_id"), ObjectId)]

    def _filter_rows_by_unit_ids(
        self,
        rows: Any,
        unit_ids: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        dict_rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        if not unit_ids:
            return dict_rows

        unit_id_str = {str(value) for value in unit_ids if value is not None}
        if not unit_id_str:
            return []

        filtered: List[Dict[str, Any]] = []
        for row in dict_rows:
            assignments = row.get("assignments", [])
            if isinstance(assignments, list):
                if any(str((item or {}).get("unitId") or "") in unit_id_str for item in assignments if isinstance(item, dict)):
                    filtered.append(row)
                    continue

            units = row.get("units", [])
            if isinstance(units, list):
                if any(
                    str((item or {}).get("unitId") or (item or {}).get("_id") or "") in unit_id_str
                    for item in units
                    if isinstance(item, dict)
                ):
                    filtered.append(row)
        return filtered

    def _normalize_label(self, value: Optional[Any]) -> str:
        if value is None:
            return ""
        normalized = normalize_common_entity_aliases(str(value))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\bSPDO\b", "SDPO", normalized, flags=re.IGNORECASE)
        return normalized.lower()

    def _designation_text_match(self, candidate: Optional[Any], target: str) -> bool:
        if not target:
            return False
        text = self._normalize_label(candidate)
        if not text:
            return False
        return (
            text == target
            or text.startswith(f"{target} ")
            or target.startswith(f"{text} ")
            or (len(target) >= 4 and target in text)
        )

    def _row_matches_designation(
        self,
        row: Dict[str, Any],
        *,
        designation_id: Optional[Any],
        designation_text: str,
    ) -> bool:
        if designation_id is None and not designation_text:
            return True

        def _match_dict(item: Dict[str, Any]) -> bool:
            if designation_id is not None:
                if str(item.get("designationId") or "") == str(designation_id):
                    return True

            for field in ("designationName", "designation_name", "postCode", "post_code"):
                if self._designation_text_match(item.get(field), designation_text):
                    return True

            designation_value = item.get("designation")
            if isinstance(designation_value, dict):
                if designation_id is not None and str(designation_value.get("_id") or "") == str(designation_id):
                    return True
                if self._designation_text_match(designation_value.get("name"), designation_text):
                    return True
                if self._designation_text_match(designation_value.get("shortCode"), designation_text):
                    return True
            elif isinstance(designation_value, list):
                for nested in designation_value:
                    if isinstance(nested, dict):
                        if designation_id is not None and str(nested.get("_id") or "") == str(designation_id):
                            return True
                        if self._designation_text_match(nested.get("name"), designation_text):
                            return True
                        if self._designation_text_match(nested.get("shortCode"), designation_text):
                            return True
            return False

        if _match_dict(row):
            return True

        rank = row.get("rank")
        if isinstance(rank, dict):
            if self._designation_text_match(rank.get("name"), designation_text):
                return True
            if self._designation_text_match(rank.get("shortCode"), designation_text):
                return True

        for list_key in ("assignments", "units"):
            payload = row.get(list_key)
            if isinstance(payload, list):
                for nested in payload:
                    if isinstance(nested, dict) and _match_dict(nested):
                        return True

        return False

    def _canonical_person_name(self, raw_name: str) -> str:
        value = normalize_common_entity_aliases(str(raw_name or ""))
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(self._PERSON_TITLE_PREFIX_RE, "", value)
        value = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        return re.sub(r"\s+", " ", value)

    async def _resolve_best_person_name(
        self,
        raw_name: str,
        *,
        include_inactive: bool,
    ) -> Optional[str]:
        probe = self._canonical_person_name(raw_name)
        if not probe:
            return None
        names_filter: Dict[str, Any] = {"isDelete": False}
        if not include_inactive:
            names_filter["isActive"] = True
        names = await self.db[Collections.PERSONNEL_MASTER].distinct("name", names_filter)
        canonical_to_original: Dict[str, str] = {}
        for item in names:
            if isinstance(item, str):
                key = self._canonical_person_name(item)
                if key and key not in canonical_to_original:
                    canonical_to_original[key] = item
        if probe in canonical_to_original:
            return canonical_to_original[probe]
        best = fuzzy_best_match(probe, canonical_to_original.keys(), cutoff=0.78)
        if not best:
            return None
        return canonical_to_original.get(best)

    def _normalize_person_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(row)
        normalized["_id"] = str(normalized.get("_id") or "")
        rank = normalized.get("rank")
        if isinstance(rank, list):
            rank = rank[0] if rank else {}
        rank_obj = rank if isinstance(rank, dict) else {}
        normalized["rank"] = {
            "name": rank_obj.get("name") or normalized.get("rankName"),
            "shortCode": rank_obj.get("shortCode") or rank_obj.get("short_code") or normalized.get("rankShortCode"),
        }
        return normalized

    def _non_empty_params(self, params: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {k: v for k, v in params.items() if isinstance(v, str) and v.strip()}


class SearchUnitTool(BaseTool):
    """Search units with relationship enrichment enabled by default."""

    name = "search_unit"
    description = (
        "Search for a unit/station by name, police reference ID, or city. "
        "Returns enriched unit details with district, parent unit, and responsible officer."
    )

    def __init__(self, db):
        super().__init__(db)
        self._repo = UnitRepository(db)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unit name"},
                "police_reference_id": {"type": "string", "description": "Unit police reference ID"},
                "city": {"type": "string", "description": "City"},
                "district_name": {"type": "string", "description": "District name"},
                "page": {"type": "integer", "default": 1},
                "page_size": {"type": "integer", "default": 20},
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        name = arguments.get("name")
        police_ref_id = arguments.get("police_reference_id")
        city = arguments.get("city")
        district_name = arguments.get("district_name")
        page, page_size, _ = self.get_pagination_params(arguments)

        if not any([name, police_ref_id, city, district_name]):
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide at least one search parameter: name, police_reference_id, city, or district_name",
            )

        district_id = await self._resolve_district_id(district_name)
        scope_context = ScopeContext.from_user_context(context)
        result = await self._repo.search(
            name=name,
            police_reference_id=police_ref_id,
            city=city,
            district_id=district_id,
            page=page,
            page_size=page_size,
            scope_context=scope_context,
        )
        rows = result.get("data", [])
        pagination = result.get("pagination", {})
        total = int(pagination.get("total", 0) or 0)

        return self.format_success_response(
            query_type="search_unit",
            data=[row for row in rows if isinstance(row, dict)],
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "search_params": self._non_empty_params(
                    {
                        "name": name,
                        "police_reference_id": police_ref_id,
                        "city": city,
                        "district_name": district_name,
                    }
                )
            },
        )

    async def _resolve_district_id(self, district_name: Optional[str]) -> Optional[str]:
        if not district_name:
            return None
        normalized = normalize_common_entity_aliases(str(district_name).strip())
        district = await self.db[Collections.DISTRICT].find_one(
            {"name": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}, "isDelete": False},
            {"_id": 1},
        )
        if district:
            return str(district["_id"])
        return None

    def _non_empty_params(self, params: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {k: v for k, v in params.items() if isinstance(v, str) and v.strip()}


class CheckResponsibleUserTool(BaseTool):
    """Check whether a person is the responsible officer for any unit."""

    name = "check_responsible_user"
    description = (
        "Check if a person is the responsible officer (SHO/in-charge) of any unit "
        "and return matching units with district context."
    )

    def __init__(self, db):
        super().__init__(db)
        self._person_repo = PersonnelRepository(db)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person name"},
                "user_id": {"type": "string", "description": "Person user ID"},
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        name = arguments.get("name")
        user_id = arguments.get("user_id")
        if not name and not user_id:
            return self.format_error_response("MISSING_PARAMETER", "Provide name or user_id to check")

        scope_context = ScopeContext.from_user_context(context)
        person_rows = await self._person_repo.search(
            name=name,
            user_id=user_id,
            include_inactive=False,
            page=1,
            page_size=10,
            scope_context=scope_context,
        )
        people = person_rows.get("data", []) if isinstance(person_rows, dict) else []
        if not people:
            return self.format_success_response(
                query_type="check_responsible_user",
                data={
                    "person_found": False,
                    "is_responsible_user": False,
                    "message": f"No person found matching '{name or user_id}'",
                    "units": [],
                },
                total=0,
            )

        person_ids = [ObjectId(p["_id"]) for p in people if isinstance(p, dict) and ObjectId.is_valid(str(p.get("_id")))]
        unit_query = {"isDelete": False, "responsibleUserId": {"$in": person_ids}}
        scoped_query = await self.apply_scope_filter(unit_query, context, "unit")
        units = await self.db[Collections.UNIT].find(scoped_query).to_list(length=100)
        return self.format_success_response(
            query_type="check_responsible_user",
            data={
                "person_found": True,
                "person_name": str(people[0].get("name") or (name or user_id)),
                "is_responsible_user": len(units) > 0,
                "units": units,
                "unit_count": len(units),
            },
            total=len(units),
        )


class SearchAssignmentTool(BaseTool):
    """Search assignment records with personnel/unit enrichment."""

    name = "search_assignment"
    description = (
        "Search assignment records by personnel (name/user_id), unit (unit_id/unit_name), "
        "or post code."
    )

    def __init__(self, db):
        super().__init__(db)
        self._repo = AssignmentRepository(db)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "user_id": {"type": "string"},
                "unit_id": {"type": "string"},
                "unit_name": {"type": "string"},
                "post_code": {"type": "string"},
                "include_inactive": {"type": "boolean", "default": False},
                "page": {"type": "integer", "default": 1},
                "page_size": {"type": "integer", "default": 20},
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        name = arguments.get("name")
        user_id = arguments.get("user_id")
        unit_id = arguments.get("unit_id")
        unit_name = arguments.get("unit_name")
        post_code = arguments.get("post_code")
        include_inactive = bool(arguments.get("include_inactive", False))
        page, page_size, _ = self.get_pagination_params(arguments)

        if not any([name, user_id, unit_id, unit_name, post_code]):
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide at least one search parameter: name, user_id, unit_id, unit_name, or post_code",
            )

        personnel_id = await self._resolve_personnel_id(name=name, user_id=user_id)
        resolved_unit_id = await self._resolve_unit_id(unit_id=unit_id, unit_name=unit_name, context=context)
        scope_context = ScopeContext.from_user_context(context)

        result = await self._repo.search(
            personnel_id=personnel_id,
            unit_id=resolved_unit_id,
            post_code=post_code,
            include_inactive=include_inactive,
            page=page,
            page_size=page_size,
            scope_context=scope_context,
        )
        rows = result.get("data", [])
        pagination = result.get("pagination", {})
        total = int(pagination.get("total", 0) or 0)

        return self.format_success_response(
            query_type="search_assignment",
            data=[row for row in rows if isinstance(row, dict)],
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "search_params": self._non_empty_params(
                    {"name": name, "user_id": user_id, "unit_id": unit_id, "unit_name": unit_name, "post_code": post_code}
                )
            },
        )

    async def _resolve_personnel_id(
        self,
        *,
        name: Optional[str],
        user_id: Optional[str],
    ) -> Optional[str]:
        if user_id:
            row = await self.db[Collections.PERSONNEL_MASTER].find_one(
                {"userId": str(user_id).strip(), "isDelete": False},
                {"_id": 1},
            )
            return str(row["_id"]) if row else None
        if name:
            escaped = re.escape(str(name).strip())
            row = await self.db[Collections.PERSONNEL_MASTER].find_one(
                {"isDelete": False, "name": {"$regex": escaped, "$options": "i"}},
                {"_id": 1},
            )
            return str(row["_id"]) if row else None
        return None

    async def _resolve_unit_id(
        self,
        *,
        unit_id: Optional[str],
        unit_name: Optional[str],
        context: UserContext,
    ) -> Optional[str]:
        if unit_id and ObjectId.is_valid(unit_id):
            return unit_id
        if unit_name:
            unit_query = {"name": {"$regex": re.escape(str(unit_name).strip()), "$options": "i"}, "isDelete": False}
            scoped_query = await self.apply_scope_filter(unit_query, context, "unit")
            row = await self.db[Collections.UNIT].find_one(scoped_query, {"_id": 1})
            return str(row["_id"]) if row else None
        return None

    def _non_empty_params(self, params: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {k: v for k, v in params.items() if isinstance(v, str) and v.strip()}
