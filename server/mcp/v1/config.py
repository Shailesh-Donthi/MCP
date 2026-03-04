"""
MCP Server Configuration

Extends the existing settings pattern from app/core/config.py
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class MCPSettings(BaseSettings):
    """MCP Server specific configuration"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MCP Server Config
    MCP_SERVER_NAME: str = "personnel-reporting-mcp"
    MCP_SERVER_VERSION: str = "1.0.0"
    MCP_MAX_RESULTS: int = 1000
    MCP_DEFAULT_PAGE_SIZE: int = 50

    # Query Safety
    MCP_QUERY_TIMEOUT_MS: int = 30000
    MCP_MAX_AGGREGATION_STAGES: int = 10

    # Allowed collections for queries (whitelist)
    MCP_ALLOWED_COLLECTIONS: str = (
        "approval_flow_master,department_master,district_master,error_master,"
        "jobs_master,log_master,mandal_master,modules_master,notification_master,"
        "permissions_master,permissions_mapping_master,personnel_master,prompt_master,"
        "rank_master,roles_master,unit_master,unit_type_master,unit_villages_master,"
        "user_role_permissions_master,value_sets_master,designation_master,assignment_master"
    )

    @property
    def allowed_collections_list(self) -> List[str]:
        """Get allowed collections as a list"""
        return [c.strip() for c in self.MCP_ALLOWED_COLLECTIONS.split(",")]


mcp_settings = MCPSettings()

