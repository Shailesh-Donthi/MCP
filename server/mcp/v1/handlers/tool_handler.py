"""
Tool Handler for MCP Server

Manages tool registration, execution, and error handling.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Type

from mcp.core.database import get_database
from mcp.core.error_catalog import build_error_payload, normalize_error_code
from mcp.core.logging_config import configure_logging, log_structured
from mcp.schemas.context_schema import UserContext
from mcp.tools.base_tool import BaseTool
from mcp.tools.personnel_tools import (
    QueryPersonnelByUnitTool,
    QueryPersonnelByRankTool,
)
from mcp.tools.unit_tools import (
    GetUnitHierarchyTool,
    ListUnitsInDistrictTool,
    ListDistrictsTool,
)
from mcp.tools.vacancy_tools import (
    CountVacanciesByUnitRankTool,
    GetPersonnelDistributionTool,
)
from mcp.tools.transfer_tools import (
    QueryRecentTransfersTool,
    GetUnitCommandHistoryTool,
)
from mcp.tools.village_mapping_tools import (
    FindMissingVillageMappingsTool,
    GetVillageCoverageTool,
)
from mcp.tools.search_tools import (
    SearchPersonnelTool,
    SearchUnitTool,
    CheckResponsibleUserTool,
)
from mcp.tools.master_data_tools import QueryLinkedMasterDataTool

logger = logging.getLogger(__name__)
configure_logging()


class MCPError(Exception):
    """Base exception for MCP errors"""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = normalize_error_code(error_code)
        self.details = details or {}
        self.legacy_error_code = error_code if self.error_code != error_code else None


class ValidationError(MCPError):
    """Input validation error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP-VALIDATION-001", details)


class QueryExecutionError(MCPError):
    """Query execution error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP-TOOL-002", details)


class ScopeAccessError(MCPError):
    """Scope/access error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP-SCOPE-001", details)


class ToolNotFoundError(MCPError):
    """Tool not found error"""

    def __init__(self, tool_name: str):
        super().__init__(
            f"Unknown tool: {tool_name}",
            "MCP-TOOL-001",
            {"tool_name": tool_name},
        )


# Registry of all available tool classes
TOOL_CLASSES: List[Type[BaseTool]] = [
    # Search tools (most commonly used)
    SearchPersonnelTool,
    SearchUnitTool,
    CheckResponsibleUserTool,
    # Personnel tools
    QueryPersonnelByUnitTool,
    QueryPersonnelByRankTool,
    # Unit tools
    GetUnitHierarchyTool,
    ListUnitsInDistrictTool,
    ListDistrictsTool,
    # Vacancy tools
    CountVacanciesByUnitRankTool,
    GetPersonnelDistributionTool,
    # Transfer tools
    QueryRecentTransfersTool,
    GetUnitCommandHistoryTool,
    # Village mapping tools
    FindMissingVillageMappingsTool,
    GetVillageCoverageTool,
    # Linked master-data tool
    QueryLinkedMasterDataTool,
]


class ToolHandler:
    """
    Handles MCP tool registration and execution.

    Provides:
    - Tool registration from classes
    - Input validation
    - Error handling and formatting
    - Response serialization
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize tools with database connection"""
        if self._initialized:
            return

        db = get_database()
        if db is None:
            raise RuntimeError("Database not connected")

        for tool_cls in TOOL_CLASSES:
            tool = tool_cls(db)
            self._tools[tool.name] = tool

        self._initialized = True
        log_structured(
            logger,
            "info",
            "tool_handler_initialized",
            tools_loaded=len(self._tools),
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools with their schemas.

        Returns:
            List of tool definitions for MCP server
        """
        if not self._initialized:
            self.initialize()

        tools = []
        for tool in self._tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.get_input_schema(),
            })
        return tools

    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        if not self._initialized:
            self.initialize()
        return list(self._tools.keys())

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool with comprehensive error handling.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool input arguments
            context: User context for scope filtering

        Returns:
            Tool execution result or error response
        """
        if not self._initialized:
            self.initialize()

        # Default context if none provided (internal/state-level access)
        if context is None:
            context = UserContext(scope_level="state")

        try:
            # Validate tool exists
            if tool_name not in self._tools:
                raise ToolNotFoundError(tool_name)

            tool = self._tools[tool_name]

            # Validate input schema
            validation_error = self._validate_arguments(tool, arguments)
            if validation_error:
                raise ValidationError(validation_error)

            # Execute tool
            log_structured(logger, "info", "tool_execute_started", tool_name=tool_name)
            result = await tool.execute(arguments, context)

            return result

        except ValidationError as e:
            log_structured(
                logger,
                "warning",
                "tool_execute_validation_error",
                tool_name=tool_name,
                error_code=e.error_code,
                message=e.message,
            )
            return self._format_error(e)

        except ScopeAccessError as e:
            log_structured(
                logger,
                "warning",
                "tool_execute_scope_error",
                tool_name=tool_name,
                error_code=e.error_code,
                message=e.message,
            )
            return self._format_error(e)

        except QueryExecutionError as e:
            log_structured(
                logger,
                "error",
                "tool_execute_query_error",
                tool_name=tool_name,
                error_code=e.error_code,
                message=e.message,
            )
            return self._format_error(e)

        except ToolNotFoundError as e:
            log_structured(
                logger,
                "warning",
                "tool_execute_tool_not_found",
                tool_name=tool_name,
                error_code=e.error_code,
                message=e.message,
            )
            return self._format_error(e)

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}: {str(e)}")
            error = MCPError(
                "An unexpected error occurred",
                "MCP-INTERNAL-001",
                {"original_error": str(e)},
            )
            return self._format_error(error)

    async def execute_json(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[UserContext] = None,
    ) -> str:
        """
        Execute a tool and return JSON string result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool input arguments
            context: User context for scope filtering

        Returns:
            JSON string of tool execution result
        """
        result = await self.execute(tool_name, arguments, context)
        return json.dumps(result, indent=2, default=str)

    def _validate_arguments(
        self, tool: BaseTool, arguments: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate arguments against tool schema.

        Args:
            tool: Tool instance
            arguments: Provided arguments

        Returns:
            Error message if validation fails, None otherwise
        """
        schema = tool.get_input_schema()
        required = schema.get("required", [])

        # Check required fields
        missing = [field for field in required if field not in arguments]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"

        # Validate types if properties defined
        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field in properties and value is not None:
                prop_schema = properties[field]
                expected_type = prop_schema.get("type")

                if expected_type == "string" and not isinstance(value, str):
                    return f"Field '{field}' must be a string"
                elif expected_type == "integer" and not isinstance(value, int):
                    return f"Field '{field}' must be an integer"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return f"Field '{field}' must be a boolean"

                # Validate enum values
                if "enum" in prop_schema and value not in prop_schema["enum"]:
                    return (
                        f"Field '{field}' must be one of: "
                        f"{', '.join(prop_schema['enum'])}"
                    )

        return None

    def _format_error(self, error: MCPError) -> Dict[str, Any]:
        """Format error as response dictionary"""
        return {
            "success": False,
            "error": build_error_payload(
                error.error_code,
                message=error.message,
                details=error.details,
                legacy_code=error.legacy_error_code,
            ),
        }


# Global handler instance
_handler: Optional[ToolHandler] = None


def get_tool_handler() -> ToolHandler:
    """Get the global tool handler instance"""
    global _handler
    if _handler is None:
        _handler = ToolHandler()
    return _handler

