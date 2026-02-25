"""
Tool Input/Output Schemas for MCP Tools

Defines the structured response formats for MCP tool outputs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime

T = TypeVar("T")


class PaginationInfo(BaseModel):
    """Pagination metadata for paginated results"""

    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Items per page")
    total: int = Field(0, description="Total number of items")
    total_pages: int = Field(0, description="Total number of pages")

    @classmethod
    def create(cls, page: int, page_size: int, total: int) -> "PaginationInfo":
        """Factory method to create pagination info"""
        return cls(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size if page_size > 0 else 0,
        )


class ToolResponse(BaseModel):
    """Standard response wrapper for all MCP tools"""

    success: bool = Field(True, description="Whether the query was successful")
    query_type: str = Field(..., description="Type of query executed")
    data: Any = Field(None, description="Query results")
    pagination: Optional[PaginationInfo] = Field(
        None, description="Pagination info if applicable"
    )
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the query"
    )


class PersonnelResult(BaseModel):
    """Personnel record in query results"""

    id: str = Field(..., alias="_id", description="Personnel MongoDB ID")
    name: str = Field(..., description="Full name")
    user_id: Optional[str] = Field(None, alias="userId", description="Police User ID")
    badge_no: Optional[str] = Field(None, alias="badgeNo", description="Badge number")
    mobile: Optional[str] = Field(None, description="Mobile number")
    email: Optional[str] = Field(None, description="Email address")
    rank_name: Optional[str] = Field(None, alias="rankName", description="Rank name")
    rank_short_code: Optional[str] = Field(
        None, alias="rankShortCode", description="Rank short code"
    )
    unit_name: Optional[str] = Field(None, alias="unitName", description="Unit name")
    designation_name: Optional[str] = Field(
        None, alias="designationName", description="Designation"
    )
    is_active: bool = Field(True, alias="isActive", description="Active status")

    class Config:
        populate_by_name = True


class PersonnelGroupedByRank(BaseModel):
    """Personnel grouped by rank"""

    rank_id: Optional[str] = Field(None, description="Rank MongoDB ID")
    rank_name: str = Field("Unknown", description="Rank name")
    rank_short_code: Optional[str] = Field(None, description="Rank short code")
    count: int = Field(0, description="Number of personnel in this rank")
    personnel: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sample personnel (limited)"
    )


class UnitResult(BaseModel):
    """Unit record in query results"""

    id: str = Field(..., alias="_id", description="Unit MongoDB ID")
    name: str = Field(..., description="Unit name")
    district_name: Optional[str] = Field(
        None, alias="districtName", description="District name"
    )
    unit_type: Optional[str] = Field(None, alias="unitType", description="Unit type")
    parent_unit_name: Optional[str] = Field(
        None, alias="parentUnitName", description="Parent unit name"
    )
    personnel_count: int = Field(
        0, alias="personnelCount", description="Personnel count"
    )
    responsible_user_name: Optional[str] = Field(
        None, alias="responsibleUserName", description="Responsible officer name"
    )

    class Config:
        populate_by_name = True


class VacancyResult(BaseModel):
    """Vacancy analysis result"""

    unit_id: str = Field(..., description="Unit MongoDB ID")
    unit_name: str = Field(..., description="Unit name")
    district_name: Optional[str] = Field(None, description="District name")
    total_personnel: int = Field(0, description="Total personnel count")
    by_rank: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Breakdown by rank"
    )


class TransferResult(BaseModel):
    """Transfer record in query results"""

    unit_id: str = Field(..., description="Unit MongoDB ID")
    unit_name: str = Field(..., description="Unit name")
    district_name: Optional[str] = Field(None, description="District name")
    personnel_id: Optional[str] = Field(None, description="Personnel ID")
    personnel_name: Optional[str] = Field(None, description="Personnel name")
    change_type: Optional[str] = Field(None, description="Type of change")
    changed_at: Optional[datetime] = Field(None, description="When change occurred")
    from_date: Optional[datetime] = Field(None, description="Effective from date")
    to_date: Optional[datetime] = Field(None, description="Effective to date")


class VillageMappingResult(BaseModel):
    """Village mapping result"""

    id: str = Field(..., alias="_id", description="Record MongoDB ID")
    name: str = Field(..., description="Unit or personnel name")
    unit_id: Optional[str] = Field(None, alias="unitId", description="Unit ID")
    unit_name: Optional[str] = Field(None, alias="unitName", description="Unit name")
    district_name: Optional[str] = Field(
        None, alias="districtName", description="District name"
    )
    user_id: Optional[str] = Field(
        None, alias="userId", description="Personnel User ID (if personnel)"
    )

    class Config:
        populate_by_name = True


class UnitHierarchyNode(BaseModel):
    """Node in unit hierarchy tree"""

    id: str = Field(..., alias="_id", description="Unit MongoDB ID")
    name: str = Field(..., description="Unit name")
    unit_type: Optional[str] = Field(None, alias="unitType", description="Unit type")
    personnel_count: int = Field(
        0, alias="personnelCount", description="Personnel count"
    )
    children: List["UnitHierarchyNode"] = Field(
        default_factory=list, description="Child units"
    )

    class Config:
        populate_by_name = True


# Update forward references for recursive model
UnitHierarchyNode.model_rebuild()

