"""Entity repositories built on top of V2 EnrichedBaseRepository."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.constants import Collections
from mcp.v2.repositories.enriched_base_repository import EnrichedBaseRepository
from mcp.v2.repositories.scope_context import ScopeContext


class PersonnelRepository(EnrichedBaseRepository):
    """Relationship-aware queries for personnel."""

    DEFAULT_ENRICHMENTS = [
        "rank",
        "department",
        "assignments",
        "assignments.unit",
        "assignments.unit.district",
        "assignments.designation",
    ]

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, Collections.PERSONNEL_MASTER)

    async def search(
        self,
        *,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        badge_no: Optional[str] = None,
        mobile: Optional[str] = None,
        email: Optional[str] = None,
        designation_id: Optional[Any] = None,
        include_inactive: bool = False,
        page: int = 1,
        page_size: int = 50,
        scope_context: Optional[ScopeContext] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"isDelete": False}
        if not include_inactive:
            filters["isActive"] = True

        and_conditions: List[Dict[str, Any]] = []
        if name:
            escaped = re.escape(str(name).strip())
            if escaped:
                and_conditions.append(
                    {
                        "$or": [
                            {"name": {"$regex": escaped, "$options": "i"}},
                            {"firstName": {"$regex": escaped, "$options": "i"}},
                            {"lastName": {"$regex": escaped, "$options": "i"}},
                        ]
                    }
                )
        if user_id:
            and_conditions.append({"userId": str(user_id)})
        if badge_no:
            and_conditions.append({"badgeNo": str(badge_no)})
        if mobile:
            mobile_value = str(mobile).strip()
            and_conditions.append({"mobile": {"$regex": re.escape(mobile_value)}})
        if email:
            and_conditions.append({"email": {"$regex": re.escape(str(email).strip()), "$options": "i"}})
        if designation_id is not None:
            and_conditions.append(
                {
                    "$or": [
                        {"designationId": designation_id},
                        {"units.designationId": designation_id},
                    ]
                }
            )

        if and_conditions:
            filters["$and"] = and_conditions

        return await self.find_enriched(
            filters=filters,
            enrichments=self.DEFAULT_ENRICHMENTS,
            scope_context=scope_context,
            sort={"name": 1, "userId": 1, "_id": 1},
            page=page,
            page_size=page_size,
        )

    async def find_by_user_id(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
        scope_context: Optional[ScopeContext] = None,
    ) -> Optional[Dict[str, Any]]:
        filters: Dict[str, Any] = {"userId": str(user_id), "isDelete": False}
        if not include_inactive:
            filters["isActive"] = True
        return await self.find_one_enriched(
            filters=filters,
            enrichments=self.DEFAULT_ENRICHMENTS,
            scope_context=scope_context,
        )


class AssignmentRepository(EnrichedBaseRepository):
    """Relationship-aware queries for assignments."""

    DEFAULT_ENRICHMENTS = [
        "personnel",
        "unit",
        "unit.district",
        "designation",
    ]

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, Collections.ASSIGNMENT_MASTER)

    async def search(
        self,
        *,
        personnel_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        post_code: Optional[str] = None,
        include_inactive: bool = False,
        page: int = 1,
        page_size: int = 50,
        scope_context: Optional[ScopeContext] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"isDelete": False}
        if not include_inactive:
            filters["isActive"] = True

        if personnel_id and ObjectId.is_valid(personnel_id):
            filters["userId"] = ObjectId(personnel_id)
        if unit_id and ObjectId.is_valid(unit_id):
            filters["unitId"] = ObjectId(unit_id)
        if post_code:
            filters["postCode"] = {"$regex": re.escape(str(post_code).strip()), "$options": "i"}

        return await self.find_enriched(
            filters=filters,
            enrichments=self.DEFAULT_ENRICHMENTS,
            scope_context=scope_context,
            sort={"startDate": -1, "_id": -1},
            page=page,
            page_size=page_size,
        )


class UnitRepository(EnrichedBaseRepository):
    """Relationship-aware queries for units."""

    DEFAULT_ENRICHMENTS = [
        "district",
        "unit_type",
        "parent_unit",
        "responsible_user",
    ]

    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, Collections.UNIT)

    async def search(
        self,
        *,
        name: Optional[str] = None,
        police_reference_id: Optional[str] = None,
        city: Optional[str] = None,
        district_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
        scope_context: Optional[ScopeContext] = None,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"isDelete": False}
        and_conditions: List[Dict[str, Any]] = []

        if name:
            and_conditions.append({"name": {"$regex": re.escape(str(name).strip()), "$options": "i"}})
        if police_reference_id:
            and_conditions.append(
                {"policeReferenceId": {"$regex": re.escape(str(police_reference_id).strip()), "$options": "i"}}
            )
        if city:
            and_conditions.append({"city": {"$regex": re.escape(str(city).strip()), "$options": "i"}})
        if district_id and ObjectId.is_valid(district_id):
            and_conditions.append({"districtId": ObjectId(district_id)})

        if and_conditions:
            filters["$and"] = and_conditions

        return await self.find_enriched(
            filters=filters,
            enrichments=self.DEFAULT_ENRICHMENTS,
            scope_context=scope_context,
            sort={"name": 1, "_id": 1},
            page=page,
            page_size=page_size,
        )

