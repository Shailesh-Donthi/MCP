"""Repository scope context (V2) derived from request/user context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mcp.schemas.context_schema import UserContext


@dataclass
class ScopeContext:
    """Scope information used by V2 repository-layer filtering."""

    user_id: Optional[str] = None
    unit_id: Optional[str] = None
    unit_ids: List[str] = field(default_factory=list)
    district_id: Optional[str] = None
    district_ids: List[str] = field(default_factory=list)
    post_code: Optional[str] = None
    role_name: Optional[str] = None
    reports_to_post: Optional[Dict[str, Any]] = None
    scope_level: str = "state"

    @property
    def has_state_access(self) -> bool:
        return self.scope_level == "state"

    @property
    def has_district_access(self) -> bool:
        return self.scope_level in ("district", "state")

    @property
    def has_unit_access(self) -> bool:
        return self.scope_level in ("unit", "district", "state")

    def get_accessible_unit_ids(self) -> List[str]:
        if self.unit_ids:
            return [str(u) for u in self.unit_ids if u]
        if self.unit_id:
            return [str(self.unit_id)]
        return []

    def get_accessible_district_ids(self) -> List[str]:
        if self.district_ids:
            return [str(d) for d in self.district_ids if d]
        if self.district_id:
            return [str(self.district_id)]
        return []

    def to_user_context(self) -> UserContext:
        """Convert to existing UserContext model for compatibility."""
        return UserContext(
            user_id=self.user_id,
            unit_id=self.unit_id,
            unit_ids=self.get_accessible_unit_ids(),
            district_id=self.district_id,
            district_ids=self.get_accessible_district_ids(),
            role_id=None,
            scope_level=self.scope_level,
        )

    @classmethod
    def from_user_context(cls, context: Optional[UserContext]) -> "ScopeContext":
        if context is None:
            return cls(scope_level="state")
        return cls(
            user_id=context.user_id,
            unit_id=context.unit_id,
            unit_ids=list(context.unit_ids or []),
            district_id=context.district_id,
            district_ids=list(context.district_ids or []),
            post_code=getattr(context, "post_code", None),
            role_name=getattr(context, "role_name", None),
            reports_to_post=getattr(context, "reports_to_post", None),
            scope_level=context.scope_level or "state",
        )

