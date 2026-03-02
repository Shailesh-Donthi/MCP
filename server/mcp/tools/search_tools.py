"""
Search Tools for MCP

Tools for searching personnel, units, and other entities by name or attributes.
"""

from typing import Any, Dict, List, Optional, Tuple
from bson import ObjectId
import re
import copy

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.query_builder.builder import SafeQueryBuilder
from mcp.constants import Collections
from mcp.router.extractors import normalize_common_entity_aliases, fuzzy_best_match


class SearchPersonnelTool(BaseTool):
    """
    Search for personnel by name, userId, badgeNo, mobile, or email.
    Returns personnel details including their unit assignments.
    """

    name = "search_personnel"
    description = (
        "Search for a person by name, userId, badge number, mobile, or email. "
        "Returns their details including which unit(s) they belong to, their rank, "
        "and designation. Use this to find 'Which unit does person X belong to?'"
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Person's name (partial match, case insensitive)",
                },
                "user_id": {
                    "type": "string",
                    "description": "Police User ID (8 digits)",
                },
                "badge_no": {
                    "type": "string",
                    "description": "Badge number",
                },
                "mobile": {
                    "type": "string",
                    "description": "Mobile number",
                },
                "email": {
                    "type": "string",
                    "description": "Email address",
                },
                "designation_name": {
                    "type": "string",
                    "description": "Designation name (case insensitive), e.g. SDPO",
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
                    "default": 20,
                },
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
        include_inactive = arguments.get("include_inactive", False)
        page, page_size, skip = self.get_pagination_params(arguments)

        # Validate at least one search param
        if not any([name, user_id, badge_no, mobile, email, designation_name]):
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide at least one search parameter: name, user_id, badge_no, mobile, email, or designation_name",
            )

        search_params = {
            "name": name,
            "user_id": user_id,
            "badge_no": badge_no,
            "mobile": mobile,
            "email": email,
            "designation_name": designation_name,
        }
        designation_condition = await self._resolve_designation_condition(designation_name)
        if designation_name and not designation_condition:
            return self.format_error_response(
                "NOT_FOUND",
                f"Designation not found: {designation_name}",
                {"hint": "Try a valid designation value (for example: SDPO)."},
            )
        search_conditions = self._build_search_conditions(
            name=name,
            user_id=user_id,
            badge_no=badge_no,
            mobile=mobile,
            email=email,
        )
        if designation_condition:
            search_conditions.append(designation_condition)
        base_query = self._build_base_query(search_conditions, include_inactive)

        # Apply scope filter
        query = await self.apply_scope_filter(base_query, context, "personnel")

        pipeline = self._build_personnel_search_pipeline(query, skip, page_size)

        results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            pipeline
        ).to_list(length=None)

        # Get total count
        total = await self.db[Collections.PERSONNEL_MASTER].count_documents(query)

        formatted_results = self._format_personnel_results(results)
        metadata: Dict[str, Any] = {
            "search_params": self._non_empty_params(search_params),
        }

        # If the user gave a name, prefer exact full-name matches over broader partial matches.
        # When duplicates share the same exact name, return all of those exact-name matches.
        exact_override = await self._get_exact_name_override(
            name=name,
            user_id=user_id,
            badge_no=badge_no,
            mobile=mobile,
            email=email,
            scoped_query=query,
        )
        if exact_override is not None:
            exact_results, exact_total, partial_total, exact_truncated = exact_override
            formatted_results = self._format_personnel_results(exact_results)
            metadata.update({
                "exact_name_match_applied": True,
                "exact_name_match_count": exact_total,
                "partial_name_match_count": total,
                "exact_name_match_truncated": exact_truncated,
            })
            total = exact_total
            page = 1
            page_size = max(1, len(formatted_results))

        return self.format_success_response(
            query_type="search_personnel",
            data=formatted_results,
            total=total,
            page=page,
            page_size=page_size,
            metadata=metadata,
        )

    def _build_search_conditions(
        self,
        name: Optional[str],
        user_id: Optional[str],
        badge_no: Optional[str],
        mobile: Optional[str],
        email: Optional[str],
    ) -> List[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        if name:
            escaped_name = SafeQueryBuilder.escape_regex(name)
            conditions.append(
                {
                    "$or": [
                        {"name": {"$regex": escaped_name, "$options": "i"}},
                        {"firstName": {"$regex": escaped_name, "$options": "i"}},
                        {"lastName": {"$regex": escaped_name, "$options": "i"}},
                    ]
                }
            )
        if user_id:
            conditions.append({"userId": user_id})
        if badge_no:
            conditions.append({"badgeNo": badge_no})
        if mobile:
            mobile_clean = re.sub(r"\D", "", mobile)
            if len(mobile_clean) == 10:
                conditions.append(
                    {
                        "$or": [
                            {"mobile": mobile},
                            {"mobile": {"$regex": mobile_clean}},
                        ]
                    }
                )
            else:
                conditions.append({"mobile": {"$regex": mobile_clean}})
        if email:
            conditions.append({"email": {"$regex": email, "$options": "i"}})
        return conditions

    async def _resolve_designation_condition(
        self,
        designation_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not designation_name:
            return None
        normalized = normalize_common_entity_aliases(str(designation_name).strip())
        if not normalized:
            return None
        if re.fullmatch(r"[A-Za-z]{2,10}", normalized) and normalized.lower().endswith("s"):
            normalized = normalized[:-1]
        # Common typo/alias normalization for this domain.
        normalized = re.sub(r"\bSPDO\b", "SDPO", normalized, flags=re.IGNORECASE)
        designation = await self.db[Collections.DESIGNATION_MASTER].find_one(
            {
                "name": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"},
                "isDelete": False,
            },
            {"_id": 1},
        )
        if not designation:
            names = await self.db[Collections.DESIGNATION_MASTER].distinct("name", {"isDelete": False})
            best = fuzzy_best_match(normalized, names, cutoff=0.76)
            if best:
                designation = await self.db[Collections.DESIGNATION_MASTER].find_one(
                    {
                        "name": {"$regex": f"^{re.escape(best)}$", "$options": "i"},
                        "isDelete": False,
                    },
                    {"_id": 1},
                )
        if not designation:
            return None
        return {"units.designationId": designation["_id"]}

    async def _get_exact_name_override(
        self,
        *,
        name: Optional[str],
        user_id: Optional[str],
        badge_no: Optional[str],
        mobile: Optional[str],
        email: Optional[str],
        scoped_query: Dict[str, Any],
    ) -> Optional[Tuple[List[Dict[str, Any]], int, int, bool]]:
        """Return exact full-name matches when user searched by name only and exacts exist."""
        if not name or any([user_id, badge_no, mobile, email]):
            return None

        exact_name = re.sub(r"\s+", " ", str(name).strip())
        if not exact_name:
            return None

        exact_query = copy.deepcopy(scoped_query)
        exact_condition = {
            "name": {"$regex": f"^{re.escape(exact_name)}$", "$options": "i"},
        }
        and_conditions = exact_query.get("$and")
        if isinstance(and_conditions, list):
            and_conditions.append(exact_condition)
        else:
            exact_query["$and"] = [exact_condition]

        exact_total = await self.db[Collections.PERSONNEL_MASTER].count_documents(exact_query)
        if exact_total <= 0:
            return None

        fetch_limit = min(max(1, exact_total), self.max_results)
        exact_results = await self.db[Collections.PERSONNEL_MASTER].aggregate(
            self._build_personnel_search_pipeline(exact_query, 0, fetch_limit)
        ).to_list(length=None)

        partial_total = await self.db[Collections.PERSONNEL_MASTER].count_documents(scoped_query)
        return exact_results, exact_total, partial_total, exact_total > fetch_limit

    def _build_base_query(
        self,
        search_conditions: List[Dict[str, Any]],
        include_inactive: bool,
    ) -> Dict[str, Any]:
        base_query: Dict[str, Any] = {"isDelete": False, "$and": search_conditions}
        if not include_inactive:
            base_query["isActive"] = True
        return base_query

    def _build_personnel_search_pipeline(
        self,
        query: Dict[str, Any],
        skip: int,
        page_size: int,
    ) -> List[Dict[str, Any]]:
        return [
            {"$match": query},
            # Add stable tie-breakers so ambiguous-name searches return a consistent first match.
            {"$sort": {"name": 1, "userId": 1, "_id": 1}},
            {"$skip": skip},
            {"$limit": page_size},
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
                "$lookup": {
                    "from": Collections.DEPARTMENT,
                    "localField": "departmentId",
                    "foreignField": "_id",
                    "as": "departmentData",
                }
            },
            {"$unwind": {"path": "$departmentData", "preserveNullAndEmptyArrays": True}},
            {"$unwind": {"path": "$units", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "units.unitId",
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
                "$lookup": {
                    "from": Collections.DESIGNATION_MASTER,
                    "localField": "units.designationId",
                    "foreignField": "_id",
                    "as": "designationData",
                }
            },
            {"$unwind": {"path": "$designationData", "preserveNullAndEmptyArrays": True}},
            {
                "$group": {
                    "_id": "$_id",
                    "name": {"$first": "$name"},
                    "firstName": {"$first": "$firstName"},
                    "lastName": {"$first": "$lastName"},
                    "userId": {"$first": "$userId"},
                    "badgeNo": {"$first": "$badgeNo"},
                    "mobile": {"$first": "$mobile"},
                    "email": {"$first": "$email"},
                    "gender": {"$first": "$gender"},
                    "dateOfBirth": {"$first": "$dateOfBirth"},
                    "fatherName": {"$first": "$fatherName"},
                    "address": {"$first": "$address"},
                    "bloodGroup": {"$first": "$bloodGroup"},
                    "dateOfJoining": {"$first": "$dateOfJoining"},
                    "dateOfRetirement": {"$first": "$dateOfRetirement"},
                    "isActive": {"$first": "$isActive"},
                    "rankName": {"$first": "$rankData.name"},
                    "rankShortCode": {"$first": "$rankData.shortCode"},
                    "departmentName": {"$first": "$departmentData.name"},
                    "units": {
                        "$push": {
                            "unitId": {"$toString": "$unitData._id"},
                            "unitName": "$unitData.name",
                            "districtName": "$districtData.name",
                            "designationName": "$designationData.name",
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "firstName": 1,
                    "lastName": 1,
                    "userId": 1,
                    "badgeNo": 1,
                    "mobile": 1,
                    "email": 1,
                    "gender": 1,
                    "dateOfBirth": 1,
                    "fatherName": 1,
                    "address": 1,
                    "bloodGroup": 1,
                    "dateOfJoining": 1,
                    "dateOfRetirement": 1,
                    "isActive": 1,
                    "rank": {"name": "$rankName", "shortCode": "$rankShortCode"},
                    "department": "$departmentName",
                    "units": {
                        "$filter": {
                            "input": "$units",
                            "cond": {"$ne": ["$$this.unitId", None]},
                        }
                    },
                }
            },
            # $group does not guarantee preserving prior sort order; sort again on final rows.
            {"$sort": {"name": 1, "userId": 1, "_id": 1}},
        ]

    def _format_personnel_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for person in results:
            unit_summary = self._build_unit_summary(person.get("units", []))
            formatted.append(
                {
                    **person,
                    "unit_summary": unit_summary,
                    "primary_unit": unit_summary[0] if unit_summary else "Not assigned",
                }
            )
        return formatted

    def _build_unit_summary(self, units: List[Dict[str, Any]]) -> List[str]:
        summary: List[str] = []
        for unit in units:
            if not unit.get("unitName"):
                continue
            text = unit["unitName"]
            if unit.get("districtName"):
                text += f" ({unit['districtName']})"
            if unit.get("designationName"):
                text += f" - {unit['designationName']}"
            summary.append(text)
        return summary

    def _non_empty_params(self, params: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {k: v for k, v in params.items() if v}


class SearchUnitTool(BaseTool):
    """Search for units by name, code, or location"""

    name = "search_unit"
    description = (
        "Search for a unit/station by name, police reference ID, or city. "
        "Returns unit details including district, personnel count, and responsible officer."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unit name (partial match, case insensitive)",
                },
                "police_reference_id": {
                    "type": "string",
                    "description": "Police reference/code",
                },
                "city": {
                    "type": "string",
                    "description": "City name",
                },
                "district_name": {
                    "type": "string",
                    "description": "Filter by district name",
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "default": 20,
                },
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
        page, page_size, skip = self.get_pagination_params(arguments)

        # Validate at least one search param
        if not any([name, police_ref_id, city, district_name]):
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide at least one search parameter: name, police_reference_id, city, or district_name",
            )

        search_params = {
            "name": name,
            "police_reference_id": police_ref_id,
            "city": city,
            "district_name": district_name,
        }
        search_conditions = self._build_unit_search_conditions(name, police_ref_id, city)
        district_condition = await self._resolve_district_condition(district_name)
        if district_condition:
            search_conditions.append(district_condition)
        base_query = self._build_unit_base_query(search_conditions)

        # Apply scope filter
        query = await self.apply_scope_filter(base_query, context, "unit")

        pipeline = self._build_unit_search_pipeline(query, skip, page_size)

        results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(length=None)
        total = await self.db[Collections.UNIT].count_documents(query)

        # Fuzzy retry for common misspellings/aliases (e.g., Arunelpet -> Arundelpet).
        if (
            total == 0
            and isinstance(name, str)
            and name.strip()
            and not police_ref_id
            and not city
        ):
            unit_names = await self.db[Collections.UNIT].distinct("name", {"isDelete": False})
            best = fuzzy_best_match(name, unit_names, cutoff=0.76)
            if isinstance(best, str) and best.strip() and best.strip().lower() != name.strip().lower():
                search_conditions = self._build_unit_search_conditions(best, police_ref_id, city)
                if district_condition:
                    search_conditions.append(district_condition)
                query = await self.apply_scope_filter(self._build_unit_base_query(search_conditions), context, "unit")
                pipeline = self._build_unit_search_pipeline(query, skip, page_size)
                results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(length=None)
                total = await self.db[Collections.UNIT].count_documents(query)

        return self.format_success_response(
            query_type="search_unit",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "search_params": self._non_empty_params(search_params),
            },
        )

    def _build_unit_search_conditions(
        self,
        name: Optional[str],
        police_ref_id: Optional[str],
        city: Optional[str],
    ) -> List[Dict[str, Any]]:
        conditions: List[Dict[str, Any]] = []
        if name:
            name = normalize_common_entity_aliases(name)
            escaped_name = SafeQueryBuilder.escape_regex(name)
            conditions.append({"name": {"$regex": escaped_name, "$options": "i"}})
        if police_ref_id:
            conditions.append({"policeReferenceId": {"$regex": police_ref_id, "$options": "i"}})
        if city:
            escaped_city = SafeQueryBuilder.escape_regex(city)
            conditions.append({"city": {"$regex": escaped_city, "$options": "i"}})
        return conditions

    async def _resolve_district_condition(
        self,
        district_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not district_name:
            return None
        district_name = normalize_common_entity_aliases(district_name)
        district = await self.db[Collections.DISTRICT].find_one(
            {"name": {"$regex": f"^{district_name}$", "$options": "i"}, "isDelete": False}
        )
        if not district:
            names = await self.db[Collections.DISTRICT].distinct("name", {"isDelete": False})
            best = fuzzy_best_match(district_name, names, cutoff=0.76)
            if best:
                district = await self.db[Collections.DISTRICT].find_one(
                    {"name": {"$regex": f"^{re.escape(best)}$", "$options": "i"}, "isDelete": False}
                )
        if not district:
            return None
        return {"districtId": district["_id"]}

    def _build_unit_base_query(self, search_conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        base_query: Dict[str, Any] = {"isDelete": False}
        if search_conditions:
            base_query["$and"] = search_conditions
        return base_query

    def _build_unit_search_pipeline(
        self,
        query: Dict[str, Any],
        skip: int,
        page_size: int,
    ) -> List[Dict[str, Any]]:
        return [
            {"$match": query},
            {"$sort": {"name": 1}},
            {"$skip": skip},
            {"$limit": page_size},
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "districtId",
                    "foreignField": "_id",
                    "as": "districtData",
                }
            },
            {"$unwind": {"path": "$districtData", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": Collections.UNIT_TYPE,
                    "localField": "unitTypeId",
                    "foreignField": "_id",
                    "as": "unitTypeData",
                }
            },
            {"$unwind": {"path": "$unitTypeData", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": Collections.PERSONNEL_MASTER,
                    "localField": "responsibleUserId",
                    "foreignField": "_id",
                    "as": "responsibleUserData",
                }
            },
            {"$unwind": {"path": "$responsibleUserData", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": Collections.UNIT,
                    "localField": "parentUnitId",
                    "foreignField": "_id",
                    "as": "parentUnitData",
                }
            },
            {"$unwind": {"path": "$parentUnitData", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "policeReferenceId": 1,
                    "city": 1,
                    "address1": 1,
                    "phone": 1,
                    "email": 1,
                    "district": {
                        "id": {"$toString": "$districtData._id"},
                        "name": "$districtData.name",
                    },
                    "unitType": "$unitTypeData.name",
                    "parentUnit": {
                        "id": {"$toString": "$parentUnitData._id"},
                        "name": "$parentUnitData.name",
                    },
                    "responsibleOfficer": {
                        "id": {"$toString": "$responsibleUserData._id"},
                        "name": "$responsibleUserData.name",
                        "userId": "$responsibleUserData.userId",
                    },
                    "personnelCount": {"$size": {"$ifNull": ["$unitPersonnelList", []]}},
                    "isActive": 1,
                }
            },
        ]

    def _non_empty_params(self, params: Dict[str, Optional[str]]) -> Dict[str, str]:
        return {k: v for k, v in params.items() if v}


class CheckResponsibleUserTool(BaseTool):
    """
    Check if a person is a responsible user (SHO/in-charge) of any unit.
    Returns the units where they are the responsible officer.
    """

    name = "check_responsible_user"
    description = (
        "Check if a person is a responsible user (SHO/Station House Officer/in-charge) "
        "of any unit. Use this when asked 'Is X a responsible user?', 'Which unit does X head?', "
        "'Is X an SHO?', 'What unit is X in charge of?'"
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Person's name to check",
                },
                "user_id": {
                    "type": "string",
                    "description": "Person's user ID",
                },
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
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Provide name or user_id to check",
            )

        # Step 1: Find the person first
        person_query: Dict[str, Any] = {"isDelete": False, "isActive": True}

        if name:
            escaped_name = SafeQueryBuilder.escape_regex(name)
            person_query["$or"] = [
                {"name": {"$regex": escaped_name, "$options": "i"}},
                {"firstName": {"$regex": escaped_name, "$options": "i"}},
            ]

        if user_id:
            person_query["userId"] = user_id

        # Find matching personnel
        personnel = await self.db[Collections.PERSONNEL_MASTER].find(
            person_query,
            {"_id": 1, "name": 1, "userId": 1}
        ).to_list(length=10)

        if not personnel:
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

        # Step 2: Check if any of these personnel are responsible users of units
        personnel_ids = [p["_id"] for p in personnel]
        person_names = {str(p["_id"]): p.get("name", "Unknown") for p in personnel}

        # Find units where these personnel are responsible users
        unit_query: Dict[str, Any] = {
            "isDelete": False,
            "responsibleUserId": {"$in": personnel_ids},
        }

        # Apply scope filter
        unit_query = await self.apply_scope_filter(unit_query, context, "unit")

        pipeline = [
            {"$match": unit_query},
            # Lookup district
            {
                "$lookup": {
                    "from": Collections.DISTRICT,
                    "localField": "districtId",
                    "foreignField": "_id",
                    "as": "districtData",
                }
            },
            {"$unwind": {"path": "$districtData", "preserveNullAndEmptyArrays": True}},
            # Lookup unit type
            {
                "$lookup": {
                    "from": Collections.UNIT_TYPE,
                    "localField": "unitTypeId",
                    "foreignField": "_id",
                    "as": "unitTypeData",
                }
            },
            {"$unwind": {"path": "$unitTypeData", "preserveNullAndEmptyArrays": True}},
            # Project
            {
                "$project": {
                    "_id": {"$toString": "$_id"},
                    "name": 1,
                    "policeReferenceId": 1,
                    "city": 1,
                    "responsibleUserId": {"$toString": "$responsibleUserId"},
                    "district": "$districtData.name",
                    "unitType": "$unitTypeData.name",
                    "phone": 1,
                }
            },
        ]

        units = await self.db[Collections.UNIT].aggregate(pipeline).to_list(length=None)

        # Add person name to each unit
        for unit in units:
            resp_id = unit.get("responsibleUserId")
            unit["responsibleUserName"] = person_names.get(resp_id, "Unknown")

        is_responsible = len(units) > 0
        person_name = personnel[0].get("name", name)

        if is_responsible:
            message = f"Yes, {person_name} is the responsible user (in-charge) of {len(units)} unit(s)."
        else:
            message = f"No, {person_name} is not a responsible user of any unit."

        return self.format_success_response(
            query_type="check_responsible_user",
            data={
                "person_found": True,
                "person_name": person_name,
                "is_responsible_user": is_responsible,
                "message": message,
                "units": units,
                "unit_count": len(units),
            },
            total=len(units),
            metadata={
                "search_name": name,
                "search_user_id": user_id,
            },
        )

