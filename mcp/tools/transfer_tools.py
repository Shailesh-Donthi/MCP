"""
Transfer Query Tools for MCP

Tools for querying personnel transfer history.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from bson import ObjectId
import re
import logging

from mcp.tools.base_tool import BaseTool
from mcp.schemas.context_schema import UserContext
from mcp.utils.date_parser import parse_date_range, get_date_range_description
from mcp.constants import Collections
from mcp.core.logging_config import log_structured


logger = logging.getLogger(__name__)


class QueryRecentTransfersTool(BaseTool):
    """Query personnel transfers within a date range"""

    name = "query_recent_transfers"
    description = (
        "List transfers in the last N days for a district or unit. "
        "Shows changes in unit responsible officers over time. "
        "Supports relative dates like 'last 30 days' or 'last week'."
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
                    "description": "Filter by unit ID",
                },
                "unit_name": {
                    "type": "string",
                    "description": "Filter by unit name (case insensitive)",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back",
                    "default": 30,
                },
                "from_date": {
                    "type": "string",
                    "description": "Start date (ISO format or relative like 'last week')",
                },
                "to_date": {
                    "type": "string",
                    "description": "End date (ISO format or relative)",
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
        days = arguments.get("days", 30)
        from_date = arguments.get("from_date")
        to_date = arguments.get("to_date")
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve names to IDs
        if not district_id and district_name:
            district_id = await self._resolve_name(
                Collections.DISTRICT, district_name
            )

        if not unit_id and unit_name:
            unit_id = await self._resolve_name(Collections.UNIT, unit_name)

        # Parse date range
        start_date, end_date = parse_date_range(from_date, to_date, days)

        # Build unit match conditions
        unit_match: Dict[str, Any] = {"isDelete": False}

        if district_id and ObjectId.is_valid(district_id):
            unit_match["districtId"] = ObjectId(district_id)

        if unit_id and ObjectId.is_valid(unit_id):
            unit_match["_id"] = ObjectId(unit_id)

        # Apply scope filter
        unit_match = await self.apply_scope_filter(unit_match, context, "unit")

        # Query units with responsibleUserHistory changes in date range
        pipeline = [
            {"$match": unit_match},
            # Unwind the history array
            {
                "$unwind": {
                    "path": "$responsibleUserHistory",
                    "preserveNullAndEmptyArrays": False,
                }
            },
            # Filter by date range
            {
                "$match": {
                    "$or": [
                        {
                            "responsibleUserHistory.changedAt": {
                                "$gte": start_date,
                                "$lte": end_date,
                            }
                        },
                        {
                            "responsibleUserHistory.fromDate": {
                                "$gte": start_date,
                                "$lte": end_date,
                            }
                        },
                    ]
                }
            },
            # Lookup personnel data
            {
                "$lookup": {
                    "from": Collections.PERSONNEL_MASTER,
                    "localField": "responsibleUserHistory.userId",
                    "foreignField": "_id",
                    "as": "personnelData",
                }
            },
            {
                "$unwind": {
                    "path": "$personnelData",
                    "preserveNullAndEmptyArrays": True,
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
            # Project final fields
            {
                "$project": {
                    "_id": 0,
                    "unitId": {"$toString": "$_id"},
                    "unitName": "$name",
                    "districtName": "$districtData.name",
                    "personnelId": {"$toString": "$personnelData._id"},
                    "personnelName": "$personnelData.name",
                    "personnelUserId": "$personnelData.userId",
                    "title": "$responsibleUserHistory.title",
                    "reason": "$responsibleUserHistory.reason",
                    "fromDate": "$responsibleUserHistory.fromDate",
                    "toDate": "$responsibleUserHistory.toDate",
                    "changedAt": "$responsibleUserHistory.changedAt",
                    "changedBy": "$responsibleUserHistory.changedBy",
                }
            },
            # Sort by most recent first
            {"$sort": {"changedAt": -1, "fromDate": -1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]

        results = await self.db[Collections.UNIT].aggregate(pipeline).to_list(
            length=None
        )

        # Format dates in results
        for r in results:
            if r.get("fromDate") and isinstance(r["fromDate"], datetime):
                r["fromDate"] = r["fromDate"].isoformat()
            if r.get("toDate") and isinstance(r["toDate"], datetime):
                r["toDate"] = r["toDate"].isoformat()
            if r.get("changedAt") and isinstance(r["changedAt"], datetime):
                r["changedAt"] = r["changedAt"].isoformat()

        # Get total count
        count_pipeline = pipeline[:-3]  # Remove skip, limit, project
        count_pipeline.append({"$count": "total"})
        count_result = await self.db[Collections.UNIT].aggregate(
            count_pipeline
        ).to_list(length=1)
        total = count_result[0]["total"] if count_result else len(results)

        # Get metadata
        date_range_desc = get_date_range_description(start_date, end_date)

        return self.format_success_response(
            query_type="recent_transfers",
            data=results,
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "date_range": {
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat(),
                    "description": date_range_desc,
                },
                "district_id": district_id,
                "unit_id": unit_id,
                "note": "Transfers based on responsibleUserHistory in unit_master",
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


class GetUnitCommandHistoryTool(BaseTool):
    """Get the command history for a specific unit"""

    name = "get_unit_command_history"
    description = (
        "Get the complete history of responsible officers (commanders) "
        "for a specific unit."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "unit_id": {
                    "type": "string",
                    "description": "Unit ID",
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
        unit_id = arguments.get("unit_id")
        unit_name = arguments.get("unit_name")
        page, page_size, skip = self.get_pagination_params(arguments)

        # Resolve unit name
        if not unit_id and unit_name:
            unit_id = await self._resolve_unit_name(unit_name)

        if not unit_id:
            if unit_name:
                log_structured(
                    logger,
                    "warning",
                    "unit_command_history_unit_resolution_failed",
                    requested_unit_name=unit_name,
                    tool=self.name,
                    user_id=getattr(context, "user_id", None),
                    has_state_access=getattr(context, "has_state_access", None),
                )
            else:
                log_structured(
                    logger,
                    "warning",
                    "unit_command_history_missing_unit_identifier",
                    tool=self.name,
                    user_id=getattr(context, "user_id", None),
                    has_state_access=getattr(context, "has_state_access", None),
                    arguments={k: v for k, v in (arguments or {}).items() if k in {"unit_id", "unit_name"}},
                )
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Unit ID or name is required",
            )

        if not ObjectId.is_valid(unit_id):
            return self.format_error_response(
                "VALIDATION_ERROR",
                f"Invalid unit id: {unit_id}",
            )

        # Get unit with history
        unit_query = await self.apply_scope_filter(
            {"_id": ObjectId(unit_id), "isDelete": False},
            context,
            "unit",
        )
        unit = await self.db[Collections.UNIT].find_one(unit_query)

        if not unit:
            log_structured(
                logger,
                "warning",
                "unit_command_history_unit_not_found",
                tool=self.name,
                unit_id=unit_id,
                requested_unit_name=unit_name,
                user_id=getattr(context, "user_id", None),
            )
            return self.format_error_response(
                "NOT_FOUND", f"Unit not found: {unit_id}"
            )

        # Get history entries
        history = unit.get("responsibleUserHistory", [])

        # Sort by date descending
        history = sorted(
            history,
            key=lambda x: x.get("fromDate") or x.get("changedAt") or datetime.min,
            reverse=True,
        )

        total = len(history)

        # Paginate
        history = history[skip : skip + page_size]

        # Enrich with personnel names
        personnel_ids = [
            h.get("userId") for h in history if h.get("userId")
        ]
        personnel_map = {}

        if personnel_ids:
            cursor = self.db[Collections.PERSONNEL_MASTER].find(
                {"_id": {"$in": personnel_ids}},
                {"_id": 1, "name": 1, "userId": 1},
            )
            personnel = await cursor.to_list(length=None)
            personnel_map = {str(p["_id"]): p for p in personnel}
            missing_history_refs = [
                str(pid) for pid in personnel_ids
                if pid and str(pid) not in personnel_map
            ]
            if missing_history_refs:
                log_structured(
                    logger,
                    "warning",
                    "unit_command_history_missing_personnel_refs",
                    tool=self.name,
                    unit_id=unit_id,
                    unit_name=unit.get("name"),
                    missing_ref_count=len(set(missing_history_refs)),
                    sample_missing_refs=list(dict.fromkeys(missing_history_refs))[:5],
                    history_entry_count=len(history),
                )

        # Format results
        results = []
        for h in history:
            user_id = h.get("userId")
            personnel = personnel_map.get(str(user_id), {}) if user_id else {}

            from_date = h.get("fromDate")
            to_date = h.get("toDate")
            changed_at = h.get("changedAt")

            results.append({
                "personnelId": str(user_id) if user_id else None,
                "personnelName": personnel.get("name"),
                "personnelUserId": personnel.get("userId"),
                "title": h.get("title"),
                "reason": h.get("reason"),
                "fromDate": from_date.isoformat() if isinstance(from_date, datetime) else from_date,
                "toDate": to_date.isoformat() if isinstance(to_date, datetime) else to_date,
                "changedAt": changed_at.isoformat() if isinstance(changed_at, datetime) else changed_at,
            })

        # Get current responsible user
        current_user = None
        if unit.get("responsibleUserId"):
            current = await self.db[Collections.PERSONNEL_MASTER].find_one(
                {"_id": unit["responsibleUserId"]},
                {"_id": 1, "name": 1, "userId": 1},
            )
            if current:
                current_user = {
                    "personnelId": str(current["_id"]),
                    "personnelName": current.get("name"),
                    "personnelUserId": current.get("userId"),
                }
            else:
                log_structured(
                    logger,
                    "warning",
                    "unit_command_history_current_user_reference_missing",
                    tool=self.name,
                    unit_id=unit_id,
                    unit_name=unit.get("name"),
                    responsible_user_id=str(unit.get("responsibleUserId")),
                )

        command_data_missing = (not current_user) and (total == 0)
        if command_data_missing:
            log_structured(
                logger,
                "warning",
                "unit_command_history_data_missing",
                tool=self.name,
                unit_id=unit_id,
                unit_name=unit.get("name"),
                requested_unit_name=unit_name,
                responsible_user_id=str(unit.get("responsibleUserId")) if unit.get("responsibleUserId") else None,
                has_responsible_history=False,
                history_entry_count=0,
                user_id=getattr(context, "user_id", None),
                note="Unit record exists but responsibleUserId and responsibleUserHistory are empty/missing",
            )

        return self.format_success_response(
            query_type="unit_command_history",
            data={
                "unitId": unit_id,
                "unitName": unit.get("name"),
                "currentResponsibleUser": current_user,
                "history": results,
            },
            total=total,
            page=page,
            page_size=page_size,
            metadata={
                "requested_unit_name": unit_name,
                "resolved_unit_name": unit.get("name"),
                "resolved_unit_id": unit_id,
                "command_data_missing": command_data_missing,
                "current_responsible_user_missing": current_user is None,
                "history_entry_count": total,
            },
        )

    async def _resolve_unit_name(self, unit_name: str) -> Optional[str]:
        """Resolve unit name with exact + tolerant matching (e.g., 'kuppam' -> 'Kuppam PS')."""
        raw = (unit_name or "").strip()
        if not raw:
            return None
        # Normalize common typo alias used by users/UI prompts.
        raw = re.sub(r"\bSPDO\b", "SDPO", raw, flags=re.IGNORECASE)

        # 1) Exact case-insensitive match
        exact = await self.db[Collections.UNIT].find_one(
            {"name": {"$regex": f"^{re.escape(raw)}$", "$options": "i"}, "isDelete": False},
            {"_id": 1},
        )
        if exact:
            return str(exact["_id"])

        # 2) Try common police-station suffix expansions
        variants = [raw]
        if not re.search(r"\b(ps|police\s+station|station)\b", raw, re.IGNORECASE):
            variants.extend([f"{raw} PS", f"{raw} Police Station", f"{raw} Station"])
        for name in variants:
            doc = await self.db[Collections.UNIT].find_one(
                {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}, "isDelete": False},
                {"_id": 1},
            )
            if doc:
                return str(doc["_id"])

        # 3) Contains-all-tokens fallback with stopword stripping and ranking.
        cleaned = re.sub(r"\b(police\s+station|station|ps)\b", " ", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip() or raw
        tokens = [re.escape(t) for t in re.split(r"\s+", cleaned) if t]
        if not tokens:
            return None
        pattern = "".join([rf"(?=.*\b{t}\b)" for t in tokens]) + r".*"
        cursor = self.db[Collections.UNIT].find(
            {"name": {"$regex": pattern, "$options": "i"}, "isDelete": False},
            {"_id": 1, "name": 1},
        ).limit(20)
        docs = await cursor.to_list(length=20)
        if docs:
            cleaned_lower = cleaned.lower()

            def score(doc: Dict[str, Any]) -> int:
                name = str(doc.get("name", "")).lower()
                s = 0
                if name.startswith(cleaned_lower):
                    s += 6
                if re.search(r"\b(ps|ups|police station|station)\b", name):
                    s += 4
                if cleaned_lower in name:
                    s += 2
                # Prefer shorter names when otherwise similar.
                s -= min(len(name), 200) // 25
                return s

            best = sorted(docs, key=score, reverse=True)[0]
            return str(best["_id"])
        return None
