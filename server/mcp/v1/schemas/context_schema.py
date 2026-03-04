"""
User Context Schema for MCP Tool Execution

Defines the user context that determines access scope for queries.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class UserContext(BaseModel):
    """User context for MCP tool execution"""

    user_id: Optional[str] = Field(None, description="Personnel MongoDB ObjectId")
    unit_id: Optional[str] = Field(None, description="User's primary unit ID")
    unit_ids: List[str] = Field(
        default_factory=list, description="All unit IDs user has access to"
    )
    district_id: Optional[str] = Field(None, description="User's district ID")
    district_ids: List[str] = Field(
        default_factory=list, description="All district IDs user has access to"
    )
    role_id: Optional[str] = Field(None, description="User's role ID")

    # Access scope
    scope_level: str = Field(
        default="unit", description="Access scope: 'unit', 'district', 'state'"
    )

    @property
    def has_state_access(self) -> bool:
        """Check if user has state-level access (sees all data)"""
        return self.scope_level == "state"

    @property
    def has_district_access(self) -> bool:
        """Check if user has district-level access"""
        return self.scope_level in ("district", "state")

    @property
    def has_unit_access(self) -> bool:
        """Check if user has at least unit-level access"""
        return self.scope_level in ("unit", "district", "state")

    def get_accessible_unit_ids(self) -> List[str]:
        """Get all unit IDs the user can access"""
        if self.unit_ids:
            return self.unit_ids
        elif self.unit_id:
            return [self.unit_id]
        return []

    def get_accessible_district_ids(self) -> List[str]:
        """Get all district IDs the user can access"""
        if self.district_ids:
            return self.district_ids
        elif self.district_id:
            return [self.district_id]
        return []

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "507f1f77bcf86cd799439011",
                "unit_id": "507f1f77bcf86cd799439012",
                "unit_ids": [
                    "507f1f77bcf86cd799439012",
                    "507f1f77bcf86cd799439013",
                ],
                "district_id": "507f1f77bcf86cd799439014",
                "scope_level": "district",
            }
        }

