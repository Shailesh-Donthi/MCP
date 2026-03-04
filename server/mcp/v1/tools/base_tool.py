"""
Base Tool Class for MCP Tools

Provides common functionality for all MCP tools including
scope filtering, database access, and response formatting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from mcp.schemas.context_schema import UserContext
from mcp.query_builder.filters import ScopeFilter
from mcp.config import mcp_settings
from mcp.core.error_catalog import build_error_payload, normalize_error_code


class BaseTool(ABC):
    """
    Base class for all MCP tools.

    Provides:
    - Database access
    - Scope filtering based on user context
    - Standard response formatting
    - Input validation helpers
    """

    # Override in subclasses
    name: str = "base_tool"
    description: str = "Base tool description"

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize the tool with database connection.

        Args:
            db: Motor async database instance
        """
        self.db = db
        self.scope_filter = ScopeFilter(db)
        self.max_results = mcp_settings.MCP_MAX_RESULTS
        self.default_page_size = mcp_settings.MCP_DEFAULT_PAGE_SIZE

    @abstractmethod
    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Execute the tool with given arguments and context.

        Args:
            arguments: Tool input arguments
            context: User context with access information

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Return JSON Schema for tool inputs.

        Returns:
            JSON Schema dictionary defining expected inputs
        """
        pass

    async def apply_scope_filter(
        self,
        base_query: Dict[str, Any],
        context: UserContext,
        collection_type: str = "personnel",
    ) -> Dict[str, Any]:
        """
        Apply unit/district scope filters based on user context.

        Args:
            base_query: The base MongoDB query
            context: User context with access information
            collection_type: Type of collection ("personnel", "unit", "unit_villages")

        Returns:
            Modified query with scope filters applied
        """
        return await self.scope_filter.apply(base_query, context, collection_type)

    def get_pagination_params(
        self, arguments: Dict[str, Any]
    ) -> tuple[int, int, int]:
        """
        Extract and validate pagination parameters.

        Args:
            arguments: Tool input arguments

        Returns:
            Tuple of (page, page_size, skip)
        """
        page = max(1, arguments.get("page", 1))
        page_size = min(
            self.max_results,
            max(1, arguments.get("page_size", self.default_page_size)),
        )
        skip = (page - 1) * page_size
        return page, page_size, skip

    def format_success_response(
        self,
        query_type: str,
        data: Any,
        total: Optional[int] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a successful tool response.

        Args:
            query_type: Type of query executed
            data: Query results
            total: Total count for pagination
            page: Current page number
            page_size: Items per page
            metadata: Additional metadata

        Returns:
            Formatted response dictionary
        """
        response: Dict[str, Any] = {
            "success": True,
            "query_type": query_type,
            "data": data,
        }

        if total is not None:
            response["pagination"] = {
                "total": total,
                "page": page or 1,
                "page_size": page_size or self.default_page_size,
                "total_pages": (
                    (total + (page_size or self.default_page_size) - 1)
                    // (page_size or self.default_page_size)
                ),
            }

        if metadata:
            response["metadata"] = metadata

        return response

    def format_error_response(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error_code: Error code
            message: Error message
            details: Additional error details

        Returns:
            Formatted error response
        """
        normalized_code = normalize_error_code(error_code)
        return {
            "success": False,
            "error": build_error_payload(
                normalized_code,
                message=message,
                details=details,
                legacy_code=error_code if normalized_code != error_code else None,
            ),
        }

    def validate_required_params(
        self,
        arguments: Dict[str, Any],
        required: list[str],
    ) -> Optional[str]:
        """
        Validate that required parameters are present.

        Args:
            arguments: Tool input arguments
            required: List of required parameter names

        Returns:
            Error message if validation fails, None otherwise
        """
        missing = [param for param in required if not arguments.get(param)]
        if missing:
            return f"Missing required parameters: {', '.join(missing)}"
        return None

    def validate_at_least_one(
        self,
        arguments: Dict[str, Any],
        params: list[str],
    ) -> Optional[str]:
        """
        Validate that at least one of the parameters is provided.

        Args:
            arguments: Tool input arguments
            params: List of parameter names where at least one is required

        Returns:
            Error message if validation fails, None otherwise
        """
        if not any(arguments.get(param) for param in params):
            return f"At least one of these parameters is required: {', '.join(params)}"
        return None
