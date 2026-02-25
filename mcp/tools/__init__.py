"""MCP Tools Package"""

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
from mcp.tools.vacancy_tools import CountVacanciesByUnitRankTool, GetPersonnelDistributionTool
from mcp.tools.transfer_tools import QueryRecentTransfersTool, GetUnitCommandHistoryTool
from mcp.tools.village_mapping_tools import FindMissingVillageMappingsTool, GetVillageCoverageTool
from mcp.tools.search_tools import SearchPersonnelTool, SearchUnitTool, CheckResponsibleUserTool

__all__ = [
    "BaseTool",
    # Personnel tools
    "QueryPersonnelByUnitTool",
    "QueryPersonnelByRankTool",
    # Unit tools
    "GetUnitHierarchyTool",
    "ListUnitsInDistrictTool",
    "ListDistrictsTool",
    # Vacancy tools
    "CountVacanciesByUnitRankTool",
    "GetPersonnelDistributionTool",
    # Transfer tools
    "QueryRecentTransfersTool",
    "GetUnitCommandHistoryTool",
    # Village mapping tools
    "FindMissingVillageMappingsTool",
    "GetVillageCoverageTool",
    # Search tools
    "SearchPersonnelTool",
    "SearchUnitTool",
    "CheckResponsibleUserTool",
]

