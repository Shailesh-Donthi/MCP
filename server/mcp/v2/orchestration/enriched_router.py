"""V2 feature-flagged enriched routing helpers for MCP natural-language queries."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple


_DESIGNATION_ALIASES = [
    (r"\bsub[\s-]?inspectors?\b|\bsis?\b", "Sub Inspector"),
    (r"\bassistant\s+sub[\s-]?inspectors?\b|\basi\b|\basis\b", "Assistant Sub Inspector"),
    (r"\bhead\s+constables?\b|\bhc\b", "Head Constable"),
    (r"\bpolice\s+constables?\b|\bconstables?\b|\bpc\b", "Police Constable"),
    (r"\binspectors?\b", "Inspector"),
    (r"\bcircle\s+inspectors?\b", "Circle Inspector"),
    (r"\bdeputy\s+superintendent\s+of\s+police\b|\bdsp\b|\bdysp\b", "Deputy Superintendent of Police"),
    (r"\bsuperintendent\s+of\s+police\b|\bsp\b", "Superintendent of Police"),
]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_name(value: str) -> str:
    cleaned = re.sub(r"[\?\.,!]+$", "", _normalize_text(value))
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_temporal_tail(value: str) -> str:
    text = _normalize_text(value)
    if not text:
        return text
    text = re.sub(
        r"\s+(?:in|for|during|within)\s+(?:the\s+)?(?:last|past|previous)\s+\d+\s+"
        r"(?:day|days|week|weeks|month|months|year|years)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\s+(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _extract_tail_name(query: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, query, re.IGNORECASE)
    if not match:
        return None
    value = _strip_temporal_tail(_clean_name(match.group(1)))
    return value or None


def _extract_designation(query_lower: str) -> Optional[str]:
    for pattern, designation in _DESIGNATION_ALIASES:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return designation
    return None


def _extract_place_hint(query: str) -> Optional[str]:
    clean_query = _clean_name(query)
    filler_tokens = {
        "in",
        "of",
        "for",
        "the",
        "all",
        "officers",
        "officer",
        "personnel",
        "staff",
        "show",
        "list",
        "find",
        "get",
        "who",
        "is",
    }

    patterns = [
        r"\b(?:in|for|of)\s+([A-Za-z][A-Za-z\s]{1,60}?)(?:\s+district)?(?:\?|$)",
        r"\b([A-Za-z][A-Za-z\s]{1,60}?)\s+district\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_query, re.IGNORECASE)
        if not match:
            continue
        raw = _normalize_text(match.group(1))
        words = [w for w in re.findall(r"[A-Za-z]+", raw) if w]
        words = [w for w in words if w.lower() not in filler_tokens]
        if not words:
            continue
        candidate = " ".join(words[-4:]).strip()
        if candidate:
            return candidate.title()
    return None


def _is_primary_assignment_query(query_lower: str) -> bool:
    text = _normalize_text(query_lower).lower()
    return bool(
        re.search(r"^\s*(?:show|list|get|find)?\s*(?:all\s+)?(?:assignments?|posting|postings)\b", text)
        or re.search(r"\b(?:assignments?|posting|posted)\s+(?:for|of|under|in)\b", text)
    )


def _is_person_lookup_dominant(query_lower: str) -> bool:
    text = _normalize_text(query_lower).lower()
    return bool(
        re.search(r"\bwho\s+has\s+user\s*id\b", text)
        or re.search(r"\b(?:who\s+is|find|search|lookup)\s+(?:officer|person|personnel)\b", text)
        or re.search(r"\btell\s+me\s+about\b", text)
        or re.search(r"\b(?:mobile|phone|email|contact|badge)\b", text)
    )


def route_query_to_tool_enriched(
    query: str,
    *,
    hint: Optional[str] = None,
    available_tools: Optional[Iterable[str]] = None,
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Route query using richer TFS-inspired intent rules.

    Returns:
        (tool_name, arguments). tool_name is None when no enriched route matched.
    """
    raw_query = _normalize_text(query)
    if not raw_query:
        return "__help__", {"reason": "Please enter a query."}

    query_lower = raw_query.lower()
    enabled_tools = set(available_tools) if available_tools is not None else None

    def can_use(tool_name: str) -> bool:
        return enabled_tools is None or tool_name in enabled_tools

    if hint and can_use(hint):
        return hint, {}

    if re.search(r"\b(help|capabilit(?:y|ies)|what can you help)\b", query_lower):
        return "__help__", {}

    if re.search(r"\b(transfers?|posting|postings|movement)\b", query_lower):
        return "__help__", {"reason": "Transfer queries are not available in V2 yet."}

    if re.search(r"\b(village|villages?|mapping|coverage|unmapped)\b", query_lower):
        return "__help__", {"reason": "Village-mapping queries are not available in V2 yet."}

    if can_use("search_unit"):
        role_unit_name = _extract_tail_name(
            raw_query,
            r"\b(?:who\s+is|name\s+of)\s+(?:the\s+)?(?:sho|head\s+of)\s+(?:of\s+)?([A-Za-z0-9][A-Za-z0-9\s\.'/-]{1,120}?)(?:\?|$)",
        )
        if role_unit_name:
            parsed_unit = re.sub(
                r"\b(units?|station|police\s+station)\b",
                "",
                role_unit_name,
                flags=re.IGNORECASE,
            ).strip()
            return "search_unit", {"name": parsed_unit or role_unit_name}

    if can_use("get_unit_command_history"):
        role_unit_name = _extract_tail_name(
            raw_query,
            r"\b(?:who\s+(?:is|was)|name\s+of)\s+(?:the\s+)?(?:sho|sdpo|spdo|in[\s-]?charge)\s+(?:of\s+)?"
            r"([A-Za-z0-9][A-Za-z0-9\s\.'/-]{1,120}?)(?:\?|$)",
        )
        if role_unit_name:
            normalized = re.sub(r"\bSPDO\b", "SDPO", role_unit_name, flags=re.IGNORECASE).strip()
            return "get_unit_command_history", {"unit_name": normalized}

    if can_use("check_responsible_user") and re.search(r"\b(responsible(?:\s+user)?|heads?)\b", query_lower):
        uid_match = re.search(r"\b(?:user\s*id|userid)\s*[:#-]?\s*(\d{6,12})\b", raw_query, re.IGNORECASE)
        if uid_match:
            return "check_responsible_user", {"user_id": uid_match.group(1)}

        person_name = _extract_tail_name(
            raw_query,
            r"\b(?:is|check|find)\s+([A-Za-z][A-Za-z\s\.'-]{1,80}?)(?:\s+(?:a\s+)?(?:responsible|sho|in[\s-]?charge)|\?|$)",
        )
        if person_name:
            return "check_responsible_user", {"name": person_name}
        return "check_responsible_user", {}

    if can_use("search_assignment") and re.search(r"\b(assignments?|assigned|posting|posted)\b", query_lower):
        # Keep assignment route for assignment-first requests, but avoid
        # overriding person-identification queries that only add assignment as
        # a secondary detail.
        if _is_person_lookup_dominant(query_lower) and not _is_primary_assignment_query(query_lower):
            pass
        else:
            args: Dict[str, str] = {}
            uid_match = re.search(r"\b(?:user\s*id|userid)\s*[:#-]?\s*(\d{6,12})\b", raw_query, re.IGNORECASE)
            if uid_match:
                args["user_id"] = uid_match.group(1)
            unit_name = _extract_tail_name(
                raw_query,
                r"\b(?:in|to|under)\s+([A-Za-z0-9][A-Za-z0-9\s\.'/-]{1,120}?)(?:\s+(?:unit|station|ps|assignments?)|\?|$)",
            )
            if unit_name:
                args["unit_name"] = unit_name
            return "search_assignment", args

    if can_use("search_personnel"):
        mobile_match = re.search(r"\b(?:mobile|phone)\s*[:#-]?\s*(\d{8,15})\b", raw_query, re.IGNORECASE)
        if mobile_match:
            return "search_personnel", {"mobile": mobile_match.group(1)}

        email_match = re.search(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", raw_query, re.IGNORECASE)
        if email_match:
            return "search_personnel", {"email": email_match.group(1)}

    uid_match = re.search(r"\b(?:user\s*id|userid)\s*[:#-]?\s*(\d{6,12})\b", raw_query, re.IGNORECASE)
    if uid_match and can_use("search_personnel"):
        return "search_personnel", {"user_id": uid_match.group(1)}

    designation_name = _extract_designation(query_lower) if can_use("search_personnel") else None
    if designation_name:
        args: Dict[str, str] = {"designation_name": designation_name}
        place_hint = _extract_place_hint(raw_query)
        if place_hint:
            args["district_name"] = place_hint
        if place_hint or re.search(r"\b(show|list|find|get|who is)\b", query_lower):
            return "search_personnel", args

    if can_use("search_personnel") and re.search(r"\b(email|mobile|phone|dob|date of birth|contact|badge)\b", query_lower):
        person_name = _extract_tail_name(
            raw_query,
            r"\b(?:of|for)\s+([A-Za-z][A-Za-z\s\.'-]{1,80}?)(?:\?|$)",
        )
        if person_name:
            return "search_personnel", {"name": person_name}

    if can_use("search_personnel"):
        person_name = _extract_tail_name(
            _clean_name(raw_query),
            r"\b(?:who\s+is|details?\s+(?:of|for|about)|info(?:rmation)?\s+(?:of|for|about)|tell\s+me\s+about)\s+([A-Za-z][A-Za-z\s\.'-]{1,80})\s*$",
        )
        if person_name:
            return "search_personnel", {"name": person_name}

    if can_use("search_unit") and re.search(r"\b(where is|locate|search|find)\b", query_lower):
        unit_name = _extract_tail_name(
            raw_query,
            r"\b(?:where\s+is|find|search|locate)\s+(?:the\s+)?(?:units?|station)?\s*([A-Za-z0-9][A-Za-z0-9\s\.'/-]{1,120})",
        )
        if unit_name:
            parsed = re.sub(r"\b(units?|station|police\s+station)\b", "", unit_name, flags=re.IGNORECASE).strip()
            return "search_unit", {"name": parsed or raw_query}
        return "search_unit", {}

    if can_use("search_unit") and re.search(r"\b(units?|station|district|police\s+station|ps)\b", query_lower):
        return "search_unit", {"name": raw_query}

    if can_use("search_personnel") and re.fullmatch(r"[A-Za-z][A-Za-z\s\.'-]{1,80}", _clean_name(raw_query)):
        return "search_personnel", {"name": _clean_name(raw_query)}

    return "__help__", {"reason": "I could not map that query to an available V2 tool."}
