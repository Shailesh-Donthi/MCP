"""V2 Tool Handler for relationship-aware MCP tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from mcp.core.database import get_database
from mcp.core.error_catalog import build_error_payload, normalize_error_code
from mcp.core.logging_config import configure_logging, log_structured
from mcp.schemas.context_schema import UserContext
from mcp.tools.base_tool import BaseTool
from mcp.tools.master_data_tools import QueryLinkedMasterDataTool
from mcp.tools.personnel_tools import QueryPersonnelByRankTool, QueryPersonnelByUnitTool
from mcp.tools.transfer_tools import GetUnitCommandHistoryTool, QueryRecentTransfersTool
from mcp.tools.unit_tools import GetUnitHierarchyTool, ListDistrictsTool, ListUnitsInDistrictTool
from mcp.tools.vacancy_tools import CountVacanciesByUnitRankTool, GetPersonnelDistributionTool
from mcp.tools.village_mapping_tools import FindMissingVillageMappingsTool, GetVillageCoverageTool
from mcp.v2.tools.search_tools import (
    CheckResponsibleUserTool,
    SearchAssignmentTool,
    SearchPersonnelTool,
    SearchUnitTool,
)

logger = logging.getLogger(__name__)
configure_logging()


class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = normalize_error_code(error_code)
        self.details = details or {}
        self.legacy_error_code = error_code if self.error_code != error_code else None


class ValidationError(MCPError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MCP-VALIDATION-001", details)


class ToolNotFoundError(MCPError):
    def __init__(self, tool_name: str):
        super().__init__(f"Unknown tool: {tool_name}", "MCP-TOOL-001", {"tool_name": tool_name})


TOOL_CLASSES: List[Type[BaseTool]] = [
    # V2 enhanced search tools
    SearchPersonnelTool,
    SearchUnitTool,
    CheckResponsibleUserTool,
    SearchAssignmentTool,
    # V1-compatible feature set for full parity
    QueryPersonnelByUnitTool,
    QueryPersonnelByRankTool,
    GetUnitHierarchyTool,
    ListUnitsInDistrictTool,
    ListDistrictsTool,
    CountVacanciesByUnitRankTool,
    GetPersonnelDistributionTool,
    QueryRecentTransfersTool,
    GetUnitCommandHistoryTool,
    FindMissingVillageMappingsTool,
    GetVillageCoverageTool,
    QueryLinkedMasterDataTool,
]


class ToolHandler:
    """Handles V2 tool registration and execution."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        db = get_database()
        if db is None:
            raise RuntimeError("Database not connected")
        for tool_cls in TOOL_CLASSES:
            tool = tool_cls(db)
            self._tools[tool.name] = tool
        self._initialized = True
        log_structured(logger, "info", "v2_tool_handler_initialized", tools_loaded=len(self._tools))

    def get_tools(self) -> List[Dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.get_input_schema()}
            for tool in self._tools.values()
        ]

    def get_tool_names(self) -> List[str]:
        if not self._initialized:
            self.initialize()
        return list(self._tools.keys())

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        if context is None:
            context = UserContext(scope_level="state")
        try:
            if tool_name not in self._tools:
                raise ToolNotFoundError(tool_name)
            tool = self._tools[tool_name]
            validation_error = self._validate_arguments(tool, arguments)
            if validation_error:
                raise ValidationError(validation_error)
            log_structured(logger, "info", "v2_tool_execute_started", tool_name=tool_name)
            return await tool.execute(arguments, context)
        except MCPError as error:
            return {
                "success": False,
                "error": build_error_payload(
                    error.error_code,
                    message=error.message,
                    details=error.details,
                    legacy_code=error.legacy_error_code,
                ),
            }
        except Exception as error:  # pragma: no cover - defensive
            logger.exception("Unexpected V2 error in %s: %s", tool_name, error)
            payload = MCPError(
                "An unexpected error occurred",
                "MCP-INTERNAL-001",
                {"original_error": str(error)},
            )
            return {
                "success": False,
                "error": build_error_payload(
                    payload.error_code,
                    message=payload.message,
                    details=payload.details,
                    legacy_code=payload.legacy_error_code,
                ),
            }

    async def execute_json(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[UserContext] = None,
    ) -> str:
        result = await self.execute(tool_name, arguments, context)
        return json.dumps(result, indent=2, default=str)

    def _validate_arguments(self, tool: BaseTool, arguments: Dict[str, Any]) -> Optional[str]:
        schema = tool.get_input_schema()
        required = schema.get("required", [])
        missing = [field for field in required if field not in arguments]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"

        properties = schema.get("properties", {})
        for field, value in arguments.items():
            if field in properties and value is not None:
                expected_type = properties[field].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return f"Field '{field}' must be a string"
                if expected_type == "integer" and not isinstance(value, int):
                    return f"Field '{field}' must be an integer"
                if expected_type == "boolean" and not isinstance(value, bool):
                    return f"Field '{field}' must be a boolean"
        return None


_handler: Optional[ToolHandler] = None


def get_tool_handler() -> ToolHandler:
    """Get V2 global tool handler instance."""
    global _handler
    if _handler is None:
        _handler = ToolHandler()
    return _handler
