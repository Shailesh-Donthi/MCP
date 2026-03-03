"""Minimal natural-language response formatters for V2 tools."""

from __future__ import annotations

from typing import Any, Dict
from mcp.utils.formatters import generate_natural_language_response as _legacy_generate_nl


def generate_natural_language_response(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    """Generate concise natural-language output for active V2 tools."""
    if not isinstance(result, dict):
        return "No response payload was returned."
    if not result.get("success", False):
        error = result.get("error", {}) if isinstance(result.get("error"), dict) else {}
        return str(error.get("message") or "The request could not be completed.")

    data = result.get("data")
    pagination = result.get("pagination", {}) if isinstance(result.get("pagination"), dict) else {}

    if tool_name == "search_personnel":
        rows = data if isinstance(data, list) else []
        total = int(pagination.get("total", len(rows)) or 0)
        if total == 0:
            return "No personnel records matched the query."
        if total == 1 and rows:
            person = rows[0] if isinstance(rows[0], dict) else {}
            name = str(person.get("name") or "Unknown")
            user_id = str(person.get("userId") or "N/A")
            rank = person.get("rank") if isinstance(person.get("rank"), dict) else {}
            rank_name = str(rank.get("name") or "Unknown")
            return f"{name} (User ID: {user_id}) is {rank_name}."
        return f"Found {total} personnel records."

    if tool_name == "search_unit":
        rows = data if isinstance(data, list) else []
        total = int(pagination.get("total", len(rows)) or 0)
        if total == 0:
            return "No unit records matched the query."
        if total == 1 and rows:
            unit = rows[0] if isinstance(rows[0], dict) else {}
            name = str(unit.get("name") or "Unknown unit")
            district = unit.get("district") if isinstance(unit.get("district"), dict) else {}
            district_name = str(district.get("name") or "Unknown district")
            return f"{name} is in {district_name}."
        return f"Found {total} unit records."

    if tool_name == "check_responsible_user":
        if isinstance(data, dict):
            if data.get("is_responsible_user") is True:
                return f"Yes, {data.get('person_name') or 'the person'} is a responsible user."
            return str(data.get("message") or "Responsible-user check completed.")
        return "Responsible-user check completed."

    if tool_name == "search_assignment":
        rows = data if isinstance(data, list) else []
        total = int(pagination.get("total", len(rows)) or 0)
        if total == 0:
            return "No assignment records matched the query."
        return f"Found {total} assignment records."

    # Reuse mature V1/compat formatter for parity on all non-search tools.
    return _legacy_generate_nl(query, tool_name, arguments, result)
