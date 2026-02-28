"""
Response Formatters for MCP Tools

Provides utilities to format MongoDB documents into structured responses.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from bson import ObjectId
import re
import math

from mcp.core.error_catalog import build_error_payload, normalize_error_code


def _stringify_id(value: Any) -> str:
    return str(value or "")


def _first_truthy(*values: Any) -> Any:
    if not values:
        return None
    for value in values:
        if value:
            return value
    return values[-1]


def _iso_if_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _build_pagination(total: int, page: Optional[int], page_size: Optional[int]) -> Dict[str, int]:
    effective_page = page or 1
    effective_page_size = page_size or 50
    return {
        "total": total,
        "page": effective_page,
        "page_size": effective_page_size,
        "total_pages": (
            (total + effective_page_size - 1) // effective_page_size if page_size else 1
        ),
    }


def stringify_object_ids(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all ObjectId fields in a document to strings"""
    result = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = stringify_object_ids(value)
        elif isinstance(value, list):
            result[key] = [
                stringify_object_ids(item) if isinstance(item, dict)
                else str(item) if isinstance(item, ObjectId)
                else item
                for item in value
            ]
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def format_response(
    query_type: str,
    data: Any,
    total: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format a standard MCP tool response.

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
        response["pagination"] = _build_pagination(total, page, page_size)

    if metadata:
        response["metadata"] = metadata

    return response


def format_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format an error response"""
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


def format_personnel(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a personnel document for response.

    Args:
        doc: Raw personnel document from MongoDB

    Returns:
        Formatted personnel record
    """
    rank_data = doc.get("rankData", {})
    unit_data = doc.get("unitData", {})
    designation_data = doc.get("designationData", {})
    return {
        "_id": _stringify_id(doc.get("_id")),
        "name": doc.get("name", ""),
        "userId": doc.get("userId"),
        "badgeNo": doc.get("badgeNo"),
        "mobile": doc.get("mobile"),
        "email": doc.get("email"),
        "rankName": _first_truthy(doc.get("rankName"), rank_data.get("name")),
        "rankShortCode": _first_truthy(doc.get("rankShortCode"), rank_data.get("shortCode")),
        "unitName": _first_truthy(doc.get("unitName"), unit_data.get("name")),
        "designationName": _first_truthy(doc.get("designationName"), designation_data.get("name")),
        "isActive": doc.get("isActive", True),
    }


def format_personnel_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of personnel documents"""
    return [format_personnel(doc) for doc in docs]


def format_unit(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a unit document for response.

    Args:
        doc: Raw unit document from MongoDB

    Returns:
        Formatted unit record
    """
    district_data = doc.get("districtData", {})
    unit_type_data = doc.get("unitTypeData", {})
    parent_unit_data = doc.get("parentUnitData", {})
    responsible_user_data = doc.get("responsibleUserData", {})
    return {
        "_id": _stringify_id(doc.get("_id")),
        "name": doc.get("name", ""),
        "policeReferenceId": doc.get("policeReferenceId"),
        "districtName": _first_truthy(doc.get("districtName"), district_data.get("name")),
        "unitType": _first_truthy(doc.get("unitType"), unit_type_data.get("name")),
        "parentUnitName": _first_truthy(doc.get("parentUnitName"), parent_unit_data.get("name")),
        "personnelCount": doc.get("personnelCount", 0),
        "responsibleUserName": _first_truthy(
            doc.get("responsibleUserName"),
            responsible_user_data.get("name"),
        ),
        "city": doc.get("city"),
        "isActive": doc.get("isActive", True),
    }


def format_unit_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of unit documents"""
    return [format_unit(doc) for doc in docs]


def format_transfer(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a transfer record for response.

    Args:
        doc: Raw transfer/history document

    Returns:
        Formatted transfer record
    """
    changed_at = _iso_if_datetime(_first_truthy(doc.get("changedAt"), doc.get("changed_at")))
    from_date = _iso_if_datetime(_first_truthy(doc.get("fromDate"), doc.get("from_date")))
    to_date = _iso_if_datetime(_first_truthy(doc.get("toDate"), doc.get("to_date")))

    return {
        "unitId": _stringify_id(_first_truthy(doc.get("unitId"), doc.get("unit_id"), doc.get("_id"))),
        "unitName": _first_truthy(doc.get("unitName"), doc.get("unit_name"), ""),
        "districtName": _first_truthy(doc.get("districtName"), doc.get("district_name")),
        "personnelId": _stringify_id(
            _first_truthy(doc.get("personnelId"), doc.get("personnel_id"), doc.get("userId"))
        ),
        "personnelName": _first_truthy(doc.get("personnelName"), doc.get("personnel_name")),
        "changeType": _first_truthy(doc.get("changeType"), doc.get("change_type")),
        "changedAt": changed_at,
        "fromDate": from_date,
        "toDate": to_date,
        "reason": doc.get("reason"),
        "title": doc.get("title"),
    }


def format_transfer_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of transfer records"""
    return [format_transfer(doc) for doc in docs]


def format_vacancy(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a vacancy analysis record.

    Args:
        doc: Vacancy analysis document

    Returns:
        Formatted vacancy record
    """
    by_rank: Dict[str, Dict[str, Any]] = {}
    for rank_id, rank_data in doc.get("by_rank", doc.get("byRank", {})).items():
        if isinstance(rank_data, dict):
            by_rank[rank_id] = {
                "rankName": _first_truthy(rank_data.get("rank_name"), rank_data.get("rankName")),
                "rankShortCode": _first_truthy(
                    rank_data.get("rank_short_code"),
                    rank_data.get("rankShortCode"),
                ),
                "actualCount": _first_truthy(rank_data.get("actual_count"), rank_data.get("actualCount", 0)),
                "sanctionedStrength": _first_truthy(
                    rank_data.get("sanctioned_strength"),
                    rank_data.get("sanctionedStrength"),
                ),
                "vacancy": rank_data.get("vacancy"),
            }

    return {
        "unitId": _stringify_id(_first_truthy(doc.get("unit_id"), doc.get("unitId"), doc.get("_id"))),
        "unitName": _first_truthy(doc.get("unit_name"), doc.get("unitName"), ""),
        "districtName": _first_truthy(doc.get("district_name"), doc.get("districtName")),
        "totalPersonnel": _first_truthy(doc.get("total_personnel"), doc.get("totalPersonnel", 0)),
        "byRank": by_rank,
    }


def format_vacancy_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of vacancy records"""
    return [format_vacancy(doc) for doc in docs]


def format_village_mapping(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a village mapping record.

    Args:
        doc: Village mapping document

    Returns:
        Formatted village mapping record
    """
    return {
        "_id": _stringify_id(doc.get("_id")),
        "name": doc.get("name", ""),
        "unitId": _stringify_id(_first_truthy(doc.get("unitId"), doc.get("unit_id"))),
        "unitName": _first_truthy(doc.get("unitName"), doc.get("unit_name")),
        "districtName": _first_truthy(doc.get("districtName"), doc.get("district_name")),
        "userId": _first_truthy(doc.get("userId"), doc.get("user_id")),
    }


def format_village_mapping_list(
    docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Format a list of village mapping records"""
    return [format_village_mapping(doc) for doc in docs]


# =============================================================================
# Natural Language Response Generator
# =============================================================================

def _format_empty_result(tool_name: str, arguments: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Generate appropriate message for empty results based on tool type"""

    district = arguments.get("district_name", "")
    unit = arguments.get("unit_name", "") or arguments.get("root_unit_name", "")
    name = arguments.get("name", "")

    if tool_name == "search_personnel":
        if name:
            return f"I couldn't find anyone named '{name}' in the database. Please check the spelling and try again."
        return "No personnel found matching your criteria."

    elif tool_name == "search_unit":
        if name or unit:
            return f"I couldn't find a unit named '{name or unit}'. Please check the name and try again."
        return "No units found matching your criteria."

    elif tool_name == "query_personnel_by_unit":
        if unit:
            return f"No personnel found in {unit}. The unit may be empty or the name might be incorrect."
        elif district:
            return f"No personnel found in {district} district."
        return "No personnel found for the specified unit."

    elif tool_name == "query_personnel_by_rank":
        rank = arguments.get("rank_name", "the specified rank")
        relation = str(arguments.get("rank_relation") or "exact").strip().lower()
        relation_prefix = {
            "above": "above",
            "below": "below",
            "at_or_above": "at or above",
            "at_or_below": "at or below",
        }.get(relation)
        rank_phrase = f"{relation_prefix} {rank} rank" if relation_prefix else rank
        if district:
            return f"No personnel found {rank_phrase} in {district} district."
        return f"No personnel found with rank filter: {rank_phrase}."

    elif tool_name == "get_unit_hierarchy":
        if district:
            return f"No unit hierarchy data available for {district} district."
        elif unit:
            return f"No hierarchy data found for unit '{unit}'."
        return "No unit hierarchy data available."

    elif tool_name == "list_units_in_district":
        if district:
            return f"No units found in {district} district. Please verify the district name."
        return "No units found. Please specify a district name."

    elif tool_name == "list_districts":
        return "No districts found in the database."

    elif tool_name == "count_vacancies_by_unit_rank":
        if district:
            return f"No vacancy data available for {district} district."
        elif unit:
            return f"No vacancy data available for {unit}."
        return "No vacancy data available."

    elif tool_name == "query_recent_transfers":
        days = arguments.get("days", 30)
        if district:
            return f"No transfers recorded in {district} district in the last {days} days."
        return f"No transfers recorded in the last {days} days."

    elif tool_name == "get_unit_command_history":
        if unit:
            return f"No command history available for {unit}."
        return "No command history data available."

    elif tool_name == "get_personnel_distribution":
        if district:
            return f"No personnel data available for {district} district."
        elif unit:
            return f"No personnel data available for {unit}."
        return "No personnel distribution data available."

    elif tool_name == "get_village_coverage":
        if unit:
            return f"No village mappings found for {unit}. This unit may not have any villages assigned."
        return "No village coverage data available."

    elif tool_name == "find_missing_village_mappings":
        if district:
            return f"All units in {district} district have village mappings. No gaps found!"
        return "All units have village mappings. No gaps found!"

    elif tool_name == "check_responsible_user":
        if name:
            return f"I couldn't find anyone named '{name}' in the database."
        return "No responsible user information available."
    elif tool_name == "query_linked_master_data":
        return "No master-data records matched your query."

    # Default message
    return "No results found for your query. Please try with different criteria."


def _looks_like_missing_db_info_error(error: Dict[str, Any]) -> bool:
    """Detect not-found/no-data style tool errors that should get a friendlier fallback."""
    if not isinstance(error, dict):
        return False
    code = str(error.get("code") or "").upper()
    message = str(error.get("message") or "").lower()
    if "NOT_FOUND" in code:
        return True
    if "not found" in message:
        return True
    # Some tools may return validation-like messages when a provided name failed to resolve.
    if "required" in message and any(token in message for token in ("district", "unit", "name")):
        return True
    return False


def _extract_subject_hint(query: str, arguments: Dict[str, Any]) -> str:
    for key in ("name", "unit_name", "district_name", "root_unit_name"):
        value = str(arguments.get(key) or "").strip()
        if value:
            return value
    m = re.search(
        r"\b(?:info|information|details?|about|on|for|of)\b\s+([A-Za-z][A-Za-z0-9\s\.\-]{1,60})",
        query or "",
        re.IGNORECASE,
    )
    if not m:
        return ""
    value = re.sub(r"\s+", " ", m.group(1)).strip(" ?.")
    value = re.sub(r"\b(?:district|dist\.?)\b\s*$", "", value, flags=re.IGNORECASE).strip()
    return value


def _missing_db_fallback_suffix(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    """Add practical fallback suggestions when requested info is not in DB."""
    q = (query or "").lower()
    subject = _extract_subject_hint(query, arguments)
    suggestions: List[str] = []

    if subject:
        # Generic entity fallback: often users mean a unit/sub-division, not a person/district.
        suggestions.append(f"try `Search unit {subject}`")

    if "district" in q or "dist" in q:
        suggestions.append("verify the district name using `List districts`")
        if subject:
            suggestions.append(f"{subject} may be a unit/sub-division (for example, PS/SDPO) rather than a district")

    if tool_name == "search_personnel":
        suggestions.append("try a full name, user ID, mobile, or email")
        if subject:
            suggestions.append(f"if `{subject}` is a place, try `List units in <district>` or `Search unit {subject}`")

    if tool_name in {"list_units_in_district", "get_personnel_distribution"}:
        suggestions.append("try `List districts` to see available district names")

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for s in suggestions:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    if not deduped:
        return ""

    return "\n\nFallback suggestion: " + " ; ".join(deduped[:3]) + "."


def _with_missing_db_fallback_hint(
    base_text: str,
    *,
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> str:
    # Keep the friendlier primary message, but do not append follow-up suggestions
    # in the same chat bubble. The UI should show only the core response text.
    _ = (query, tool_name, arguments)
    return base_text


def generate_natural_language_response(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    """
    Generate a natural language response from query results.

    Args:
        query: Original user query
        tool_name: Tool that was executed
        arguments: Arguments extracted from query
        result: Tool execution result

    Returns:
        Human-readable response string
    """
    query_lower = query.lower()

    # Check if query was successful
    if not result.get("success", False):
        error = result.get("error", {})
        message = error.get("message", "An error occurred.")
        code = error.get("code")
        user_action = error.get("user_action")
        request_id = (error.get("details") or {}).get("request_id")
        if _looks_like_missing_db_info_error(error):
            response = f"I couldn't find that information in the database. {message}"
        else:
            response = f"Sorry, I couldn't process your request. {message}"
        if code:
            response += f" (Error code: {code})"
        if request_id:
            response += f" Request ID: {request_id}."
        if user_action:
            response += f" {user_action}"
        return response

    data = result.get("data", [])
    metadata = result.get("metadata", {})

    # Handle empty results with tool-specific messages
    if not data or (isinstance(data, list) and len(data) == 0):
        return _format_empty_result(tool_name, arguments, metadata)

    handlers = _get_tool_response_handlers(
        query_lower=query_lower,
        data=data,
        arguments=arguments,
        pagination=result.get("pagination", {}),
        metadata=metadata,
    )
    handler = handlers.get(tool_name)
    if handler:
        return handler()

    # Default - return summary
    else:
        if isinstance(data, list):
            return f"Found {len(data)} result(s) for your query."
        return "Query completed successfully."


def _get_tool_response_handlers(
    query_lower: str,
    data: Any,
    arguments: Dict[str, Any],
    pagination: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "check_responsible_user": lambda: _format_responsible_user_response(data, arguments),
        "search_personnel": lambda: _format_personnel_response(
            query_lower,
            data,
            arguments,
            pagination,
            metadata,
        ),
        "search_unit": lambda: _format_unit_response(query_lower, data, arguments),
        "get_unit_hierarchy": lambda: _format_hierarchy_response(data, arguments),
        "list_units_in_district": lambda: _format_district_units_response(
            data,
            arguments,
            metadata,
            pagination,
        ),
        "list_districts": lambda: _format_district_list_response(data),
        "query_personnel_by_unit": lambda: _format_unit_personnel_response(data, arguments),
        "query_personnel_by_rank": lambda: _format_rank_personnel_response(
            query_lower,
            data,
            arguments,
            pagination,
        ),
        "count_vacancies_by_unit_rank": lambda: _format_vacancy_response(data, arguments),
        "query_recent_transfers": lambda: _format_transfer_response(data, arguments),
        "get_unit_command_history": lambda: _format_unit_command_history_response(data, arguments, query_lower, metadata),
        "get_personnel_distribution": lambda: _format_distribution_response(query_lower, data, arguments),
        "get_village_coverage": lambda: _format_village_coverage_response(data, arguments),
        "find_missing_village_mappings": lambda: _format_missing_village_response(data, arguments),
        "query_linked_master_data": lambda: _format_linked_master_data_response(
            data,
            arguments,
            pagination or {},
            metadata,
        ),
    }


def _format_linked_master_data_response(
    data: Any,
    arguments: Dict[str, Any],
    pagination: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    mode = str(arguments.get("mode") or "").strip().lower()

    # Discovery mode response
    if mode == "discover" and isinstance(data, dict):
        requested = data.get("requested_collections", [])
        relations = data.get("relations", [])
        if isinstance(requested, list):
            existing = [c for c in requested if isinstance(c, dict) and c.get("exists")]
            missing = [c for c in requested if isinstance(c, dict) and not c.get("exists")]
            active_relations = [r for r in relations if isinstance(r, dict) and r.get("active")]
            response = (
                f"Master-data discovery completed.\n\n"
                f"- Requested collections: {len(requested)}\n"
                f"- Available in DB: {len(existing)}\n"
                f"- Missing in DB: {len(missing)}\n"
                f"- Active relationships: {len(active_relations)}"
            )
            if missing:
                missing_names = [str(item.get("canonical")) for item in missing if item.get("canonical")]
                if missing_names:
                    response += f"\n\nMissing collections: {', '.join(missing_names)}"
            return response
        return "Master-data discovery completed."

    # Query mode response
    if isinstance(data, list):
        total = pagination.get("total", len(data))
        collection_info = (metadata or {}).get("collection_resolution", {})
        canonical = (
            collection_info.get("canonical")
            or arguments.get("collection")
            or "requested collection"
        )
        response = (
            f"Found {total} record(s) in `{canonical}`."
        )
        if data:
            preview_lines: List[str] = []
            for idx, row in enumerate(data[:5], start=1):
                if not isinstance(row, dict):
                    continue
                label = (
                    row.get("name")
                    or row.get("flowName")
                    or row.get("errorCode")
                    or row.get("key")
                    or row.get("userId")
                    or row.get("_id")
                    or "record"
                )
                preview_lines.append(f"{idx}. {label}")
            if preview_lines:
                response += "\n\nSample results:\n" + "\n".join(preview_lines)
        forward = (metadata or {}).get("forward_links", [])
        reverse = (metadata or {}).get("reverse_links", [])
        if isinstance(forward, list) or isinstance(reverse, list):
            response += (
                f"\n\nLink expansion: "
                f"{len(forward) if isinstance(forward, list) else 0} forward relation(s), "
                f"{len(reverse) if isinstance(reverse, list) else 0} reverse relation(s)."
            )
        return response

    return "Linked master-data query completed."


def _format_hierarchy_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for unit hierarchy"""

    def format_tree(node: Dict, indent: int = 0) -> str:
        """Recursively format hierarchy tree"""
        prefix = "  " * indent
        name = node.get("name", "Unknown")
        unit_type = node.get("unitType", "")
        personnel = node.get("personnelCount", 0)

        line = f"{prefix}{'|- ' if indent > 0 else ''}{name}"
        if unit_type:
            line += f" ({unit_type})"
        if personnel:
            line += f" - {personnel} personnel"

        lines = [line]

        children = node.get("children", [])
        for child in children:
            lines.append(format_tree(child, indent + 1))

        return "\n".join(lines)

    if isinstance(data, dict):
        # Check if it's a district hierarchy response
        if "units" in data:
            district_name = data.get("district_name", "Unknown")
            units = data.get("units", [])
            total = data.get("total_units", len(units))

            response = f"Unit Hierarchy for {district_name} District:\n\n"
            for unit in units:
                response += format_tree(unit) + "\n\n"

            response += f"Total top-level units: {total}"
            return response.strip()

        # Single unit hierarchy
        elif "name" in data:
            return f"Unit Hierarchy:\n\n{format_tree(data)}"

    return "Hierarchy data retrieved."


def _format_responsible_user_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for responsible user check"""

    if isinstance(data, dict):
        person_name = data.get("person_name", arguments.get("name", "The person"))
        is_responsible = data.get("is_responsible_user", False)
        units = data.get("units", [])
        message = data.get("message", "")

        if not data.get("person_found", True):
            return f"I couldn't find anyone named '{arguments.get('name', 'unknown')}' in the database."

        if is_responsible and units:
            response = f"Yes, {person_name} is the responsible user (in-charge) of {len(units)} unit(s):\n\n"
            for i, unit in enumerate(units, 1):
                unit_name = unit.get("name", "Unknown")
                district = unit.get("district", "")
                unit_type = unit.get("unitType", "")

                response += f"  {i}. {unit_name}"
                if district:
                    response += f" ({district})"
                if unit_type:
                    response += f" - {unit_type}"
                response += "\n"

            return response.strip()
        else:
            return f"No, {person_name} is not currently a responsible user (in-charge/SHO) of any unit."

    return "Unable to determine responsible user status."


def _format_personnel_response(
    query: str,
    data: Any,
    arguments: Dict[str, Any],
    pagination: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format natural language response for personnel search"""

    metadata = metadata or {}
    exact_name_match_applied = bool(metadata.get("exact_name_match_applied"))
    exact_match_count = (
        int(metadata.get("exact_name_match_count"))
        if metadata.get("exact_name_match_count") is not None
        else None
    )
    partial_match_count = (
        int(metadata.get("partial_name_match_count"))
        if metadata.get("partial_name_match_count") is not None
        else None
    )
    exact_match_truncated = bool(metadata.get("exact_name_match_truncated"))

    def person_summary_lines(person: Dict[str, Any], index: int) -> List[str]:
        person_name = person.get("name", "Unknown")
        user_id = person.get("userId", "N/A")
        rank_name = (person.get("rank") or {}).get("name", "Unknown")
        unit_name = person.get("primary_unit", "Not assigned")
        mobile_val = person.get("mobile", "N/A")
        email_val = person.get("email", "N/A")
        return [
            f"{index}. {person_name}",
            f"   - User ID: {user_id}",
            f"   - Rank: {rank_name}",
            f"   - Unit: {unit_name}",
            f"   - Mobile: {mobile_val}",
            f"   - Email: {email_val}",
        ]

    # Handle list or single result
    asks_list_all = bool(
        ("all" in query or "list" in query or "show" in query)
        and ("person" in query or "personnel" in query or "details" in query or "info" in query)
    )

    if isinstance(data, list):
        if len(data) == 0:
            return f"I couldn't find anyone matching '{arguments.get('name', 'your search')}'."
        if exact_name_match_applied and len(data) > 1 and not asks_list_all:
            requested_name = arguments.get("name", "your search")
            header_count = exact_match_count if exact_match_count is not None else len(data)
            lines = [f"I found {header_count} exact matches for '{requested_name}'. Showing all exact-name matches:", ""]
            for i, person in enumerate(data, 1):
                lines.extend(person_summary_lines(person, i))
                if i < len(data):
                    lines.append("")
            if partial_match_count and partial_match_count > header_count:
                lines.append("")
                lines.append(
                    f"(Note: Found {partial_match_count} broader matches, but only {header_count} exact name match(es) are shown.)"
                )
            if exact_match_truncated:
                lines.append("(Note: Exact-name matches were truncated to the maximum result limit.)")
            return "\n".join(lines)
        if asks_list_all and len(data) > 1:
            total = (
                int(pagination.get("total"))
                if isinstance(pagination, dict) and pagination.get("total") is not None
                else len(data)
            )
            lines = [f"Found {total} matching personnel:", ""]
            for i, person in enumerate(data, 1):
                name = person.get("name", "Unknown")
                user_id = person.get("userId", "N/A")
                rank = (person.get("rank") or {}).get("name", "Unknown")
                mobile = person.get("mobile", "N/A")
                lines.append(f"{i}. {name} - {rank} - {user_id} - {mobile}")
            if isinstance(pagination, dict):
                page = int(pagination.get("page") or 1)
                page_size = int(pagination.get("page_size") or max(1, len(data)))
                total_pages = int(pagination.get("total_pages") or 1)
                start = ((page - 1) * page_size) + 1 if total > 0 else 0
                end = min(page * page_size, total) if total > 0 else 0
                lines.append("")
                lines.append(f"Showing {start}-{end} (Page {page}/{total_pages})")
                if total > end:
                    lines.append(f"... and {total - end} more personnel.")
            return "\n".join(lines)
        person = data[0]
        total = (
            int(pagination.get("total"))
            if isinstance(pagination, dict) and pagination.get("total") is not None
            else len(data)
        )
    else:
        person = data
        total = 1

    name = person.get("name", "Unknown")

    # Detect what information was asked for
    responses = []

    # Email query
    if any(word in query for word in ["email", "e-mail", "mail id", "email id"]):
        email = person.get("email")
        if email:
            responses.append(f"The email of {name} is {email}")
        else:
            responses.append(f"{name} does not have an email address on record")

    # Date of birth query
    elif any(word in query for word in ["date of birth", "dob", "birthday", "born", "birth date"]):
        dob = person.get("dateOfBirth")
        if dob:
            # Format the date nicely
            if isinstance(dob, str) and "T" in dob:
                dob = dob.split("T")[0]
            responses.append(f"The date of birth of {name} is {dob}")
        else:
            responses.append(f"Date of birth is not available for {name}")

    # Mobile/phone query
    elif any(word in query for word in ["mobile", "phone", "contact number", "cell"]):
        mobile = person.get("mobile")
        if mobile:
            responses.append(f"The mobile number of {name} is {mobile}")
        else:
            responses.append(f"Mobile number is not available for {name}")

    # Unit query
    elif any(word in query for word in ["unit", "station", "belongs", "posted", "work", "assigned"]):
        primary_unit = person.get("primary_unit", "Not assigned")
        responses.append(f"{name} belongs to {primary_unit}")

    # Rank/Designation query
    elif any(word in query for word in ["rank", "designation", "position", "post", "role"]):
        rank = person.get("rank", {})
        rank_name = rank.get("name", "Unknown")
        rank_short = rank.get("shortCode", "")

        # Use "designation" in response if that's what was asked
        label = "designation" if "designation" in query else "rank"

        if rank_short:
            responses.append(f"The {label} of {name} is {rank_name} ({rank_short})")
        else:
            responses.append(f"The {label} of {name} is {rank_name}")

    # Address query
    elif any(word in query for word in ["address", "location", "residence", "stay", "live"]):
        address = person.get("address")
        if address:
            responses.append(f"The address of {name} is {address}")
        else:
            responses.append(f"Address is not available for {name}")

    # Blood group query
    elif any(word in query for word in ["blood", "blood group", "blood type"]):
        blood = person.get("bloodGroup")
        if blood:
            responses.append(f"The blood group of {name} is {blood}")
        else:
            responses.append(f"Blood group is not available for {name}")

    # Father name query
    elif any(word in query for word in ["father", "parent"]):
        father = person.get("fatherName")
        if father:
            responses.append(f"The father's name of {name} is {father}")
        else:
            responses.append(f"Father's name is not available for {name}")

    # User ID / Badge query
    elif any(word in query for word in ["user id", "userid", "badge", "id number"]):
        user_id = person.get("userId")
        badge = person.get("badgeNo")
        if user_id:
            responses.append(f"The User ID of {name} is {user_id}")
        if badge:
            responses.append(f"The Badge Number of {name} is {badge}")
        if not user_id and not badge:
            responses.append(f"ID information is not available for {name}")

    # General info / find person query
    else:
        # Provide a summary
        rank = person.get("rank", {}).get("name", "Unknown rank")
        unit = person.get("primary_unit", "Not assigned")
        mobile = person.get("mobile", "N/A")
        email = person.get("email", "N/A")

        responses.append(f"Here is the information for {name}:")
        responses.append(f"  - Rank: {rank}")
        responses.append(f"  - Unit: {unit}")
        responses.append(f"  - Mobile: {mobile}")
        responses.append(f"  - Email: {email}")

    # Add note if multiple matches
    if total > 1:
        if exact_name_match_applied and exact_match_count is not None and exact_match_count == 1 and partial_match_count and partial_match_count > 1:
            responses.append(
                f"\n(Note: Found {partial_match_count} people matching your search. Showing the single exact name match.)"
            )
        else:
            responses.append(f"\n(Note: Found {total} people matching your search. Showing details for the first match.)")

    return "\n".join(responses)


def _format_unit_response(query: str, data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for unit search"""

    if isinstance(data, list):
        if len(data) == 0:
            return f"I couldn't find any unit matching '{arguments.get('name', 'your search')}'."
        unit = data[0]
        total = len(data)
    else:
        unit = data
        total = 1

    name = unit.get("name", "Unknown")
    district = unit.get("district", {}).get("name", "Unknown")
    unit_type = unit.get("unitType", "Unknown")
    personnel_count = unit.get("personnelCount", 0)
    officer = unit.get("responsibleOfficer", {}).get("name", "Not assigned")

    response = f"{name} is a {unit_type} in {district} district.\n"
    response += f"  - Personnel Count: {personnel_count}\n"
    response += f"  - Responsible Officer: {officer}"

    if total > 1:
        response += f"\n\n(Found {total} units matching your search)"

    return response


def _format_district_units_response(
    data: Any,
    arguments: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
) -> str:
    """Format natural language response for district units list"""

    district = (arguments.get("district_name") or "").strip()
    if district.lower() in {"district", "the district", "db", "database"} and metadata:
        district = (metadata.get("district_name") or "").strip()
    if not district:
        district = "the district"

    if isinstance(data, list):
        if len(data) == 0:
            return f"No units found in {district} district."

        total_units = (
            int(pagination.get("total"))
            if isinstance(pagination, dict) and pagination.get("total") is not None
            else len(data)
        )
        page = int(pagination.get("page") or 1) if isinstance(pagination, dict) else 1
        page_size = (
            int(pagination.get("page_size") or max(1, len(data)))
            if isinstance(pagination, dict)
            else max(1, len(data))
        )
        total_pages = (
            int(pagination.get("total_pages") or 1) if isinstance(pagination, dict) else 1
        )
        page_start = ((page - 1) * page_size) + 1 if total_units > 0 else 0
        page_end = min(page * page_size, total_units) if total_units > 0 else 0
        shown_count = min(len(data), 20)

        response = f"Here are the units in {district} district:\n\n"
        for i, unit in enumerate(data[:20], 1):  # Limit to 20 for readability
            name = unit.get("name", "Unknown")
            unit_type = unit.get("unitType", "")
            response += f"  {i}. {name}"
            if unit_type:
                response += f" ({unit_type})"
            response += "\n"

        if len(data) > shown_count:
            response += f"\n... and {len(data) - shown_count} more units on this page (hidden for readability)."

        if isinstance(pagination, dict):
            response += f"\nShowing {page_start}-{page_end} (Page {page}/{total_pages})"
            if len(data) > shown_count:
                response += f" [displaying first {shown_count} items from this page]"
            response += f"\nTotal: {total_units} units"
        else:
            response += f"\nTotal: {len(data)} units"
        return response

    return f"Found units in {district} district."


def _format_district_list_response(data: Any) -> str:
    """Format natural language response for district list"""
    if isinstance(data, list):
        if len(data) == 0:
            return "No districts found in the database."

        response = "Available districts:\n\n"
        for i, district in enumerate(data[:30], 1):
            name = district.get("district_name") or district.get("name") or "Unknown"
            response += f"  {i}. {name}\n"
        if len(data) > 30:
            response += f"\n... and {len(data) - 30} more districts."
        response += f"\nTotal: {len(data)} districts"
        return response

    return "District list retrieved."


def _format_unit_personnel_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for unit personnel"""

    unit_name = arguments.get("unit_name", "the unit")

    if isinstance(data, list):
        if len(data) == 0:
            return f"No personnel found in {unit_name}."

        response = f"Personnel in {unit_name}:\n\n"
        for i, person in enumerate(data[:15], 1):
            name = person.get("name", "Unknown")
            rank = person.get("rankName") or person.get("rank", {}).get("name", "")
            response += f"  {i}. {name}"
            if rank:
                response += f" - {rank}"
            response += "\n"

        if len(data) > 15:
            response += f"\n... and {len(data) - 15} more personnel."

        response += f"\nTotal: {len(data)} personnel"
        return response

    return f"Found personnel in {unit_name}."


def _format_rank_personnel_response(
    query: str,
    data: Any,
    arguments: Dict[str, Any],
    pagination: Optional[Dict[str, Any]] = None,
) -> str:
    """Format natural language response for personnel by rank"""

    rank_name = arguments.get("rank_name", "the specified rank")
    rank_relation = str(arguments.get("rank_relation") or "exact").strip().lower()
    relation_prefix_map = {
        "above": "Above",
        "below": "Below",
        "at_or_above": "At or Above",
        "at_or_below": "At or Below",
    }
    rank_label = (
        f"{relation_prefix_map.get(rank_relation, 'Exact')} {rank_name} Rank"
        if rank_relation in relation_prefix_map
        else rank_name
    )
    district = arguments.get("district_name", "")
    asks_detailed_view = any(
        token in query for token in [
            "detail", "details", "info", "information", "contact",
            "all details", "full details", "profile", "profiles",
            "their details", "give all",
        ]
    )

    if isinstance(data, list):
        if len(data) == 0:
            if district:
                return f"No {rank_label} personnel found in {district} district."
            return f"No {rank_label} personnel found."

        asks_earliest_dob = bool(
            re.search(r"\b(earliest|oldest)\b", query)
            and re.search(r"\b(date\s+of\s+birth|dob|birthday|birth)\b", query)
        )
        if asks_earliest_dob:
            candidates: List[Dict[str, Any]] = []

            def _dob_key(raw_value: Any) -> Optional[datetime]:
                if isinstance(raw_value, datetime):
                    return raw_value
                if not raw_value:
                    return None
                text = str(raw_value).strip()
                if "T" in text:
                    text = text.split("T", 1)[0]
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
                    try:
                        return datetime.strptime(text, fmt)
                    except Exception:
                        continue
                return None

            for person in data:
                if not isinstance(person, dict):
                    continue
                dob_sort = _dob_key(person.get("dateOfBirth"))
                if dob_sort is None:
                    continue
                candidates.append({"person": person, "dob_sort": dob_sort})

            if not candidates:
                return f"Date of birth is not available for {rank_name} personnel in the current result set."

            candidates.sort(key=lambda item: item["dob_sort"])
            oldest_person = candidates[0]["person"]
            dob_value = oldest_person.get("dateOfBirth")
            dob_text = str(dob_value).split("T", 1)[0] if dob_value else str(candidates[0]["dob_sort"].date())
            person_name = oldest_person.get("name", "Unknown")
            person_district = oldest_person.get("districtName")
            person_mobile = oldest_person.get("mobile")

            parts = [f"{person_name} has the earliest recorded date of birth among {rank_name} personnel: {dob_text}."]
            if person_district:
                parts.append(f"District: {person_district}.")
            if person_mobile:
                parts.append(f"Mobile: {person_mobile}.")
            if len(candidates) < len(data):
                parts.append("(Some records in this result did not include date of birth.)")
            return " ".join(parts)

        location = f" in {district} district" if district else ""
        response = f"{rank_label} Personnel{location}:\n\n"

        # Show the complete current page returned by backend pagination.
        limit = len(data)
        for i, person in enumerate(data[:limit], 1):
            name = person.get("name", "Unknown")
            user_id = person.get("userId", "N/A")
            badge = person.get("badgeNo", "N/A")
            mobile = person.get("mobile", "N/A")
            email = person.get("email", "N/A")
            person_rank = person.get("rankName") or rank_name

            if asks_detailed_view:
                response += f"  {i}. {name}\n"
                response += f"     - User ID: {user_id}\n"
                response += f"     - Badge No: {badge}\n"
                response += f"     - Rank: {person_rank}\n"
                if person.get("districtName"):
                    response += f"     - District: {person.get('districtName')}\n"
                if person.get("unitName"):
                    response += f"     - Unit: {person.get('unitName')}\n"
                response += f"     - Mobile: {mobile}\n"
                response += f"     - Email: {email}\n"
            else:
                unit = person.get("unitName") or person.get("primary_unit", "")
                response += f"  {i}. {name}"
                if unit:
                    response += f" - {unit}"
                response += "\n"

        total_matches = (
            int(pagination.get("total"))
            if isinstance(pagination, dict) and pagination.get("total") is not None
            else len(data)
        )
        response += f"\nTotal: {total_matches} {rank_label} personnel"
        if isinstance(pagination, dict):
            page = int(pagination.get("page") or 1)
            page_size = int(pagination.get("page_size") or max(1, len(data)))
            total_pages = int(pagination.get("total_pages") or 1)
            start = ((page - 1) * page_size) + 1 if total_matches > 0 else 0
            end = min(page * page_size, total_matches) if total_matches > 0 else 0
            response += f"\nShowing {start}-{end} (Page {page}/{total_pages})"
            if total_matches > end:
                response += f"\n... and {total_matches - end} more personnel."
        return response

    # Handle dict response (grouped data)
    if isinstance(data, dict):
        distribution = data.get("distribution", [])
        total = data.get("total", 0)

        response = f"Personnel by Rank (Total: {total:,}):\n\n"
        for item in distribution[:15]:
            rank = item.get("rankName", "Unknown")
            count = item.get("count", 0)
            response += f"  - {rank}: {count:,}\n"

        if len(distribution) > 15:
            response += f"\n... and {len(distribution) - 15} more ranks."

        return response.strip()

    return "Personnel information retrieved."


def _format_vacancy_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for vacancy query"""

    if isinstance(data, dict):
        # Shape A: legacy vacancy payload with explicit vacancy counts.
        total_vacancies = data.get("total_vacancies", 0)
        total_personnel = data.get("total_personnel", 0)
        if total_vacancies or total_personnel:
            response = f"Vacancy Summary:\n"
            response += f"  - Total Personnel: {total_personnel}\n"
            response += f"  - Total Vacancies: {total_vacancies}\n"

            by_rank = data.get("by_rank", [])
            if by_rank and isinstance(by_rank, list):
                response += "\nVacancies by Rank:\n"
                for rank_info in by_rank[:10]:
                    rank_name = rank_info.get("rankName", "Unknown")
                    vacancy = rank_info.get("vacancy", 0)
                    if vacancy > 0:
                        response += f"  - {rank_name}: {vacancy} vacant\n"

            return response

        # Shape B: current tool payload: { units: [...], summary: { totalUnits, totalPersonnel } }
        units = data.get("units", [])
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        if isinstance(units, list) and units:
            rank_totals: Dict[str, Dict[str, Any]] = {}
            for unit in units:
                by_rank = unit.get("byRank", {})
                if not isinstance(by_rank, dict):
                    continue
                for rank_id, rank_info in by_rank.items():
                    if not isinstance(rank_info, dict):
                        continue
                    rank_name = rank_info.get("rankName") or "Unknown"
                    actual_count = rank_info.get("actualCount", 0) or 0
                    if rank_id not in rank_totals:
                        rank_totals[rank_id] = {"rankName": rank_name, "actualCount": 0}
                    rank_totals[rank_id]["actualCount"] += int(actual_count)

            sorted_ranks = sorted(
                rank_totals.values(),
                key=lambda r: r.get("actualCount", 0),
                reverse=True,
            )

            total_units = summary.get("totalUnits", len(units))
            total_personnel = summary.get("totalPersonnel")
            if total_personnel is None:
                total_personnel = sum(u.get("totalPersonnel", 0) or 0 for u in units)

            response = "Personnel Strength by Rank:\n"
            response += f"  - Units Covered: {total_units}\n"
            response += f"  - Total Personnel: {total_personnel}\n"
            response += "  - Note: Exact vacancies require sanctioned strength data.\n"

            if sorted_ranks:
                response += "\nRank-wise personnel:\n"
                for rank_info in sorted_ranks[:12]:
                    response += f"  - {rank_info['rankName']}: {rank_info['actualCount']}\n"

            return response

        # Empty dict-like payload
        return "No vacancy data available."

    return "Vacancy information retrieved."


def _format_transfer_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for transfer query"""

    days = arguments.get("days", 30)

    if isinstance(data, list):
        if len(data) == 0:
            return f"No transfers found in the last {days} days."

        response = f"Recent transfers (last {days} days):\n\n"
        for i, transfer in enumerate(data[:10], 1):
            name = transfer.get("personnelName", "Unknown")
            unit = transfer.get("unitName", "Unknown unit")
            date = transfer.get("changedAt", "")
            if date and "T" in str(date):
                date = str(date).split("T")[0]
            response += f"  {i}. {name} - {unit}"
            if date:
                response += f" ({date})"
            response += "\n"

        if len(data) > 10:
            response += f"\n... and {len(data) - 10} more transfers."

        response += f"\nTotal: {len(data)} transfers"
        return response

    return "Transfer information retrieved."


def _format_unit_command_history_response(
    data: Any,
    arguments: Dict[str, Any],
    query: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format natural language response for unit command history."""
    if not isinstance(data, dict):
        return "Unit command history retrieved."

    unit_name = data.get("unitName") or arguments.get("unit_name") or "the unit"
    current = data.get("currentResponsibleUser") or {}
    history = data.get("history") or []
    q = (query or "").lower()
    command_data_missing = bool((metadata or {}).get("command_data_missing"))

    asks_name_only = bool(
        q and (
            re.search(r"\bwho\s+is\b", q)
            or re.search(r"\bname\s+of\b", q)
            or re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge|responsible\s*user)\b", q)
        )
    )
    asks_history = bool(q and re.search(r"\b(history|historical|previous|past|changes?)\b", q))

    if asks_name_only and not asks_history:
        current_name = current.get("personnelName") if isinstance(current, dict) else None
        if current_name:
            return f"{current_name} is the current in-charge for {unit_name}."
        if history:
            latest = history[0] if isinstance(history[0], dict) else {}
            latest_name = latest.get("personnelName")
            if latest_name:
                return (
                    f"I couldn't find a current responsible-user mapping for {unit_name}, "
                    f"but the latest recorded officer is {latest_name}."
                )
        return (
            f"I couldn't find a current responsible-user entry for {unit_name}. "
            "This may mean the command/in-charge data is missing for that unit."
        )

    if command_data_missing:
        return f"{unit_name} exists, but current responsible-user/command history data is not populated in the database."

    if not history:
        if current and current.get("personnelName"):
            return (
                f"Current responsible user for {unit_name}: "
                f"{current.get('personnelName')} ({current.get('personnelUserId') or 'N/A'}). "
                "No historical command changes are recorded."
            )
        return f"No command history available for {unit_name}."

    response = [f"Command history for {unit_name}:"]
    if current and current.get("personnelName"):
        response.append(
            f"Current responsible user: {current.get('personnelName')} "
            f"({current.get('personnelUserId') or 'N/A'})"
        )
    response.append("")

    for idx, entry in enumerate(history[:10], 1):
        name = entry.get("personnelName") or "Unknown"
        user_id = entry.get("personnelUserId") or "N/A"
        from_date = entry.get("fromDate") or "N/A"
        to_date = entry.get("toDate") or "Present"
        title = entry.get("title") or ""
        line = f"{idx}. {name} ({user_id})"
        if title:
            line += f" - {title}"
        line += f" [{from_date} to {to_date}]"
        response.append(line)

    if len(history) > 10:
        response.append(f"... and {len(history) - 10} more entries.")

    return "\n".join(response)


def _format_distribution_response(query: str, data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for personnel distribution"""

    query_lower = (query or "").lower()
    district_name = arguments.get("district_name", "")
    unit_name = arguments.get("unit_name", "")

    if isinstance(data, dict):
        distribution = data.get("distribution", [])
        total = data.get("total", 0)
        comparison = data.get("comparison", []) if isinstance(data.get("comparison"), list) else []

        def _norm_rank(value: Any) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

        def _collect_requested_ranks() -> List[str]:
            rank_patterns = [
                (r"\bpolice\s+constables?\b", "Police Constable"),
                (r"(?<!police\s)\bconstables?\b", "Constable"),
                (r"\bsub[-\s]?inspectors?\b|\bsi\b|\bsis\b", "Sub Inspector"),
                (r"\bcircle\s+inspectors?\b|\bci\b", "Circle Inspector"),
                (r"\bassistant\s+sub[-\s]?inspectors?\b|\basi\b", "Assistant SubInspector"),
                (r"\bhead\s+constables?\b|\bhc\b", "Head Constable"),
                (r"\bdeputy\s+superintendent\s+of\s+police\b|\bdsp\b|\bdysp\b", "Deputy Superintendent of Police"),
                (r"\bsuperintendent\s+of\s+police\b|\bsp\b", "Superintendent of Police"),
            ]
            hits: List[tuple[int, str]] = []
            for pattern, label in rank_patterns:
                for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                    hits.append((match.start(), label))
            hits.sort(key=lambda item: item[0])
            ordered: List[str] = []
            for _, label in hits:
                if label not in ordered:
                    ordered.append(label)
            return ordered

        if comparison:
            lines = ["District Comparison:", ""]
            if "senior officer" in query_lower:
                def _is_senior(rank_label: str) -> bool:
                    name = rank_label.lower()
                    if any(term in name for term in ["constable", "assistant sub", "junior"]):
                        return False
                    if re.search(r"\bsub\s+inspector\b", name):
                        return False
                    return bool(
                        re.search(
                            r"\b(inspector|superintendent|commissioner|dysp|dsp|sp|igp|ips|director general)\b",
                            name,
                        )
                    )

                scored: List[Dict[str, Any]] = []
                for district_row in comparison:
                    district_label = district_row.get("districtName", "Unknown")
                    district_distribution = district_row.get("distribution", []) if isinstance(district_row, dict) else []
                    district_total = int(district_row.get("total") or 0) if isinstance(district_row, dict) else 0
                    senior_count = 0
                    for row in district_distribution:
                        if not isinstance(row, dict):
                            continue
                        rank_label = row.get("rankName") or "Unknown"
                        if _is_senior(rank_label):
                            senior_count += int(row.get("count") or 0)
                    scored.append(
                        {
                            "district": district_label,
                            "senior_count": senior_count,
                            "total": district_total,
                        }
                    )
                scored.sort(key=lambda row: row["senior_count"], reverse=True)
                for row in scored:
                    lines.append(
                        f"  - {row['district']}: {row['senior_count']:,} senior officers (total personnel: {row['total']:,})"
                    )
                if len(scored) >= 2:
                    if scored[0]["senior_count"] > scored[1]["senior_count"]:
                        lines.append("")
                        lines.append(f"{scored[0]['district']} has more senior officers.")
                    else:
                        lines.append("")
                        lines.append("The compared districts have the same number of senior officers.")
                return "\n".join(lines).strip()

            for district_row in comparison:
                district_label = district_row.get("districtName", "Unknown")
                district_total = int(district_row.get("total") or 0)
                district_distribution = district_row.get("distribution", []) if isinstance(district_row, dict) else []
                top_ranks = ", ".join(
                    f"{item.get('rankName', 'Unknown')}: {int(item.get('count') or 0):,}"
                    for item in district_distribution[:3]
                    if isinstance(item, dict)
                )
                lines.append(f"  - {district_label}: {district_total:,} personnel")
                if top_ranks:
                    lines.append(f"    Top ranks: {top_ranks}")
            return "\n".join(lines).strip()

        if "ratio" in query_lower and isinstance(distribution, list):
            requested_ranks = _collect_requested_ranks()
            if requested_ranks:
                rank_counts: Dict[str, int] = {}
                for row in distribution:
                    if not isinstance(row, dict):
                        continue
                    rank_label = row.get("rankName") or ""
                    rank_counts[_norm_rank(rank_label)] = int(row.get("count") or 0)

                ordered_counts: List[int] = []
                missing: List[str] = []
                for requested in requested_ranks:
                    requested_key = _norm_rank(requested)
                    exact = rank_counts.get(requested_key)
                    if exact is None:
                        token_match = 0
                        for k, v in rank_counts.items():
                            if requested_key and requested_key in k:
                                token_match = v
                                break
                        exact = token_match if token_match else None
                    if exact is None:
                        missing.append(requested)
                        ordered_counts.append(0)
                    else:
                        ordered_counts.append(exact)

                non_zero = [count for count in ordered_counts if count > 0]
                if non_zero:
                    divisor = non_zero[0]
                    for value in non_zero[1:]:
                        divisor = math.gcd(divisor, value)
                    simplified = [int(value / divisor) if divisor > 0 else value for value in ordered_counts]

                    ratio_label = " : ".join(str(v) for v in simplified)
                    details = ", ".join(
                        f"{requested_ranks[idx]}={ordered_counts[idx]:,}"
                        for idx in range(len(requested_ranks))
                    )
                    response = (
                        f"Ratio ({' : '.join(requested_ranks)}): {ratio_label}\n"
                        f"Counts used: {details}"
                    )
                    if missing:
                        response += f"\nMissing rank categories: {', '.join(missing)}"
                    return response

        # Build header based on filters
        if district_name:
            header = f"Personnel in {district_name} District"
        elif unit_name:
            header = f"Personnel in {unit_name}"
        else:
            header = "Personnel Distribution"

        response = f"{header} (Total: {total:,}):\n\n"

        if not distribution:
            return f"No personnel found in {district_name or unit_name or 'the specified area'}."

        for rank_info in distribution[:15]:
            rank_name = rank_info.get("rankName") or rank_info.get("districtName") or rank_info.get("unitTypeName") or "Unknown"
            count = rank_info.get("count", 0)
            percentage = (count / total * 100) if total > 0 else 0
            response += f"  - {rank_name}: {count:,} ({percentage:.1f}%)\n"

        if len(distribution) > 15:
            response += f"\n... and {len(distribution) - 15} more categories."

        return response.strip()

    return "Distribution information retrieved."


def _format_village_coverage_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for village coverage"""

    unit_name = arguments.get("unit_name", "")

    if isinstance(data, list):
        if len(data) == 0:
            if unit_name:
                return f"No villages are mapped to {unit_name}."
            return "No village coverage data found."

        # Single unit result
        if len(data) == 1 or unit_name:
            unit = data[0]
            name = unit.get("unitName", unit_name or "Unknown")
            village_count = unit.get("villageCount", 0)
            mandal_count = unit.get("mandalCount", 0)
            villages = unit.get("villages", [])

            response = f"Village coverage for {name}:\n\n"
            response += f"  - Total Villages: {village_count}\n"
            response += f"  - Mandals Covered: {mandal_count}\n"

            if villages:
                response += f"\n  Villages (showing up to 10):\n"
                for i, village in enumerate(villages[:10], 1):
                    response += f"    {i}. {village}\n"

            return response.strip()

        # Multiple units
        response = "Village Coverage Summary:\n\n"
        for i, unit in enumerate(data[:15], 1):
            name = unit.get("unitName", "Unknown")
            village_count = unit.get("villageCount", 0)
            response += f"  {i}. {name}: {village_count} villages\n"

        if len(data) > 15:
            response += f"\n... and {len(data) - 15} more units."

        return response.strip()

    return "Village coverage information retrieved."


def _format_missing_village_response(data: Any, arguments: Dict[str, Any]) -> str:
    """Format natural language response for missing village mappings"""

    if isinstance(data, list):
        if len(data) == 0:
            return "All units have village mappings. No gaps found."

        response = "Units without village mappings:\n\n"
        for i, unit in enumerate(data[:15], 1):
            name = unit.get("name", "Unknown")
            district = unit.get("districtName", "")
            unit_type = unit.get("unitType", "")
            personnel = unit.get("personnelCount", 0)

            response += f"  {i}. {name}"
            if district:
                response += f" ({district})"
            if unit_type:
                response += f" - {unit_type}"
            response += f" [{personnel} personnel]\n"

        if len(data) > 15:
            response += f"\n... and {len(data) - 15} more units without mappings."

        response += f"\n\nTotal units without village mappings: {len(data)}"
        return response

    return "Missing village mapping information retrieved."
