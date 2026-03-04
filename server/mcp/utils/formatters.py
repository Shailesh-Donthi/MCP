"""Minimal natural-language response formatters for V2 tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from datetime import datetime

try:
    from bson import ObjectId as _BsonObjectId
    _ObjectIdType: Type[Any] = _BsonObjectId
except ImportError:
    _BsonObjectId = None  # type: ignore[assignment]
    _ObjectIdType = type(None)  # fallback: never matches real documents


def _is_object_id(val: Any) -> bool:
    """Runtime-safe ObjectId check that works with or without bson installed."""
    if _BsonObjectId is None:
        return isinstance(val, dict) and "$oid" in val
    return isinstance(val, _BsonObjectId)

def stringify_object_ids(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all ObjectId fields in a document to strings"""
    result: Dict[str, Any] = {}
    for key, value in doc.items():
        if _is_object_id(value):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = stringify_object_ids(value)
        elif isinstance(value, list):
            result[key] = [
                stringify_object_ids(item) if isinstance(item, dict)
                else str(item) if _is_object_id(item)
                else item
                for item in value
            ]
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result

def _stringify_id(val: Any) -> Any:
    """Helper to safely convert ObjectId to string"""
    if _is_object_id(val):
        return str(val)
    if isinstance(val, dict) and "$oid" in val:
        return val["$oid"]
    return val

def _first_truthy(*args: Any) -> Any:
    """Helper to return the first truthy value or empty string"""
    for arg in args:
        if arg:
            return arg
    return ""

def format_personnel(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Format a personnel document for response."""
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
        rows = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        total = int(pagination.get("total", len(rows)) or 0)
        if total == 0:
            return "No personnel records matched the query."
        if not rows:
            return f"Found {total} personnel records."

        def _format_date(value: Any) -> str:
            text = str(value or "").strip()
            if not text:
                return "N/A"
            if "T" in text:
                return text.split("T", 1)[0]
            return text

        def _display_text(
            value: Any,
            *,
            default: str = "N/A",
            preferred_keys: List[str] | None = None,
        ) -> str:
            if value is None:
                return default
            if isinstance(value, str):
                text = value.strip()
                return text or default
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                keys = preferred_keys or ["name", "label", "value", "shortCode", "code"]
                for key in keys:
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                    if isinstance(candidate, (int, float, bool)):
                        return str(candidate)
                # Last-resort: first non-empty scalar string from dict.
                for candidate in value.values():
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                return default
            if isinstance(value, list):
                scalar_values = [str(item).strip() for item in value if not isinstance(item, (dict, list))]
                scalar_values = [item for item in scalar_values if item]
                return ", ".join(scalar_values) if scalar_values else default
            text = str(value).strip()
            return text or default

        def _collect_assignments(person: Dict[str, Any]) -> List[Dict[str, str]]:
            source = person.get("assignments")
            if not isinstance(source, list) or not source:
                source = person.get("units", [])
            if not isinstance(source, list):
                return []
            assignments: List[Dict[str, str]] = []
            for item in source:
                if not isinstance(item, dict):
                    continue
                assignments.append(
                    {
                        "unitName": str(item.get("unitName") or "").strip(),
                        "districtName": str(item.get("districtName") or "").strip(),
                        "designationName": str(item.get("designationName") or "").strip(),
                    }
                )
            return assignments

        def _assignment_text(assignment: Dict[str, str]) -> str:
            text = assignment.get("unitName") or "Not assigned"
            district = assignment.get("districtName") or ""
            designation = assignment.get("designationName") or ""
            if district:
                text += f" ({district})"
            if designation:
                text += f" - {designation}"
            return text

        def _rank_text(person: Dict[str, Any]) -> str:
            rank = person.get("rank") if isinstance(person.get("rank"), dict) else {}
            rank_name = str(rank.get("name") or person.get("rankName") or "Unknown")
            rank_code = str(rank.get("shortCode") or person.get("rankShortCode") or "").strip()
            if rank_code:
                return f"{rank_name} ({rank_code})"
            return rank_name

        def _person_lines(person: Dict[str, Any], index: int) -> List[str]:
            assignments = _collect_assignments(person)
            primary_unit = str(person.get("primary_unit") or "").strip()
            if not primary_unit or primary_unit.lower() == "not assigned":
                primary_unit = _assignment_text(assignments[0]) if assignments else "Not assigned"
            district_name = (
                (assignments[0].get("districtName") if assignments else "")
                or _display_text(person.get("districtName"))
                or "N/A"
            )
            status = "Active" if bool(person.get("isActive", True)) else "Inactive"
            lines = [
                f"{index}. {_display_text(person.get('name'), default='Unknown')}",
                f"   - User ID: {_display_text(person.get('userId'))}",
                f"   - Badge No: {_display_text(person.get('badgeNo'))}",
                f"   - Rank: {_rank_text(person)}",
                f"   - Department: {_display_text(person.get('department'), preferred_keys=['name', 'departmentName', 'shortCode'])}",
                f"   - Status: {status}",
                f"   - Unit/Station: {primary_unit}",
                f"   - District: {district_name}",
                f"   - Gender: {_display_text(person.get('gender'))}",
                f"   - Date of birth: {_format_date(person.get('dateOfBirth'))}",
                f"   - Mobile: {_display_text(person.get('mobile'))}",
                f"   - Email: {_display_text(person.get('email'))}",
                f"   - Address: {_display_text(person.get('address'))}",
                f"   - Blood group: {_display_text(person.get('bloodGroup'))}",
                f"   - Father name: {_display_text(person.get('fatherName'))}",
                f"   - Date of joining: {_format_date(person.get('dateOfJoining'))}",
                f"   - Date of retirement: {_format_date(person.get('dateOfRetirement'))}",
            ]
            if assignments:
                lines.append("   - Active assignments:")
                for idx, assignment in enumerate(assignments, start=1):
                    lines.append(f"     {idx}. {_assignment_text(assignment)}")
            return lines

        lines = [f"Found {total} matching personnel record(s) with full details:", ""]
        for idx, person in enumerate(rows, start=1):
            lines.extend(_person_lines(person, idx))
            if idx < len(rows):
                lines.append("")

        if total > len(rows):
            page = int(pagination.get("page") or 1)
            total_pages = int(pagination.get("total_pages") or 1)
            lines.append("")
            lines.append(
                f"Showing {len(rows)} record(s) on page {page}/{total_pages}. "
                f"{total - len(rows)} more record(s) are available."
            )

        return "\n".join(lines)

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

    # Fallback response for unsupported tool natural language generation
    return str(result.get("response", "Action completed successfully."))
