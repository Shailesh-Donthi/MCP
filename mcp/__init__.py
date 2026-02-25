"""
MCP (Model Context Protocol) Server for Natural-Language Reporting

This module provides an MCP server that enables natural-language queries
against the MongoDB database with unit/district-based access controls.

Tools:
- query_personnel_by_unit: Personnel in unit, optionally grouped by rank
- query_personnel_by_rank: Personnel filtered by rank
- count_vacancies_by_unit_rank: Vacancy analysis
- query_recent_transfers: Transfer history by date range
- find_missing_village_mappings: Units/personnel without village coverage
- get_unit_hierarchy: Unit tree with personnel counts
"""

from mcp.config import mcp_settings

__all__ = ["mcp_settings"]

