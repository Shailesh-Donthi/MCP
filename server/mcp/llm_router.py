"""LLM-powered query routing and response orchestration for MCP V2."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from mcp.router.llm_client import call_openai_api, has_llm_api_key
from mcp.router.extractors import format_followup_district_response, is_followup_district_query, extract_ordinal_index, extract_list_reference_index
from mcp.schemas.context_schema import UserContext
from mcp.utils.output_layer import build_output_payload
from mcp.handlers.tool_handler import get_tool_handler
from mcp.orchestration import route_query_to_tool_enriched
from mcp.utils import generate_natural_language_response

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT_V2 = """You are a strict JSON router for a Police Personnel MCP system.

Available tools:
1) search_personnel - Find individual officers by name, ID, badge, phone, email, or designation.
   Use for: "find officer X", "who has badge 1234", "personnel with designation SDPO", "phone of X"
   Args: name, user_id, badge_no, mobile, email, designation_name, district_name, include_inactive, page, page_size
2) search_unit - Find police units/stations by name, reference ID, city, or district.
   Use for: "where is X PS", "find unit named Y", "units in Z city"
   Args: name, police_reference_id, city, district_name, page, page_size
3) check_responsible_user - Find the SHO/in-charge/responsible officer of a unit.
   Use for: "who is the SHO of X PS", "in-charge of Y station"
   Args: name, user_id
4) search_assignment - Find posting/assignment records by officer or unit.
   Use for: "where is officer X posted", "current posting of Y", "assignments at unit Z"
   Args: name, user_id, unit_id, unit_name, post_code, include_inactive, page, page_size
5) query_personnel_by_unit - List all officers in a specific unit/PS/station.
   Use for: "officers at X PS", "staff at Y station", "who works at Z DPO"
   Args: unit_name, unit_id, group_by_rank
6) query_personnel_by_rank - List officers of a given rank, optionally filtered to a district.
   Use for: "all SIs in Guntur", "list DSPs", "HCs in Chittoor", "CIs in Guntur district"
   Args: rank_name, rank_id, rank_relation, district_name
7) get_unit_hierarchy - Show parent-child unit tree for a unit or district.
   Use for: "hierarchy of X PS", "sub-units under Y", "org structure of Z district"
   Args: unit_name, unit_id, district_name
8) list_units_in_district - List all police units in a district, optionally filtered by unit type. Requires a district name.
   Use for: "all PS in Guntur", "police stations in Chittoor", "how many units in X district"
   For system-wide unit type queries like "Ranges in AP", use dynamic_query instead.
   Args: district_name (required), unit_type_name (e.g. "Range", "Police Station", "DPO")
9) list_districts - List all districts or find a specific one.
   Use for: "list all districts", "how many districts", "is X a district"
   Args: name
10) count_vacancies_by_unit_rank - Vacancy counts by rank for a unit or district.
    Use for: "vacancies in X", "empty posts in Y district"
    Args: unit_name, district_name
11) get_personnel_distribution - System-wide or district-level personnel statistics.
    Use for: "rank-wise distribution", "gender stats", "how many officers total", "officers per district"
    NEVER use for specific-unit queries -- use query_personnel_by_unit instead.
    Args: group_by (rank|gender|district|unit_type), district_name
12) query_recent_transfers - Recent transfer/posting orders.
    Use for: "recent transfers", "who transferred in last 30 days"
    Args: days, district_name
13) get_unit_command_history - Past commanders/SHOs of a unit.
    Use for: "past SHOs of X PS", "command history of Y"
    Args: unit_name, unit_id
14) find_missing_village_mappings - Units with no village mappings.
    Use for: "unmapped villages", "PS without village coverage"
    Args: district_name
15) get_village_coverage - Village coverage info for a unit.
    Use for: "villages under X PS", "village mapping for Y"
    Args: unit_name, district_name
16) query_linked_master_data - Browse/search reference tables (ranks, designations, unit types, etc.).
    Use for: "list all ranks", "what designations exist", "unit types in system"
    Args: mode, collection, filters, search_text, include_related, include_reverse, include_integrity
17) dynamic_query - LAST RESORT. Only when no tool above fits AND confidence < 0.4.
    Args: intent (plain-English description of what to find)

## AP Police Terminology
Ranks (high to low): DGP > IGP/AIGP > DIG > SP > Addl.SP/ASP > DSP/DySP > Inspector > CI > SI > ASI > HC > PC
Abbreviations: SP=Superintendent of Police, DSP=Deputy SP, CI=Circle Inspector, SI=Sub-Inspector, ASI=Assistant SI, HC=Head Constable, PC=Police Constable
SDPO = Sub-Divisional Police Officer (a DESIGNATION, not a rank -- use search_personnel with designation_name="SDPO")
SHO = Station House Officer (in-charge of a PS -- use check_responsible_user)
Unit types: PS=Police Station, DPO=District Police Office, UPS=Urban Police Station, SDPO=Sub-Division office
Unit hierarchy: State > Range > District > Sub-Division > Circle > PS
"boss/head of X district" = SP of that district -> query_personnel_by_rank with rank_name="SP"

Return exactly one JSON object:
{"tool":"tool_name","arguments":{},"understood_query":"short summary","confidence":0.0}

Rules:
- Output JSON only. One tool only.
- If unsure, choose closest tool with partial arguments.
- Confidence between 0-1. Only use dynamic_query if confidence < 0.4.
- When query mentions a specific unit/PS/station name, always use query_personnel_by_unit -- NEVER get_personnel_distribution.

Examples:
Input: "how many officers at Guntur Traffic PS"
Output: {"tool":"query_personnel_by_unit","arguments":{"unit_name":"Guntur Traffic PS"},"understood_query":"Count officers at Guntur Traffic PS","confidence":0.95}

Input: "who are the SDPOs in Chittoor?"
Output: {"tool":"search_personnel","arguments":{"designation_name":"SDPO","district_name":"Chittoor"},"understood_query":"SDPOs in Chittoor district","confidence":0.92}

Input: "list all DSPs with their posting units"
Output: {"tool":"query_personnel_by_rank","arguments":{"rank_name":"DSP"},"understood_query":"All DSPs with posting details","confidence":0.90}

Input: "who is the SP of Guntur?"
Output: {"tool":"query_personnel_by_rank","arguments":{"rank_name":"SP","district_name":"Guntur"},"understood_query":"SP of Guntur district","confidence":0.95}

Input: "how many Ranges are there in AP?"
Output: {"tool":"dynamic_query","arguments":{"intent":"Count all units of type Range in the system"},"understood_query":"Count Range-type units in AP","confidence":0.35}

Input: "list all Circle Inspectors in Guntur district"
Output: {"tool":"query_personnel_by_rank","arguments":{"rank_name":"CI","district_name":"Guntur"},"understood_query":"CIs in Guntur district","confidence":0.92}

Input: "how many Head Constables in each district?"
Output: {"tool":"get_personnel_distribution","arguments":{"group_by":"district"},"understood_query":"HC count per district","confidence":0.85}

Input: "who is the boss of Chittoor police?"
Output: {"tool":"query_personnel_by_rank","arguments":{"rank_name":"SP","district_name":"Chittoor"},"understood_query":"SP of Chittoor district","confidence":0.93}
"""

RESPONSE_FORMATTER_PROMPT_V2 = """You are a concise assistant. Use only the provided tool result.
If the result is empty, say no records matched.
Keep answer factual and short.

Important rules:
- Always echo back key entities from the user's question (district names, rank names, unit names, person names) in your response.
- Use proper rank titles (e.g. "Sub-Inspector", "Constable", "Inspector", "ASI") when listing personnel.
- When answering about a specific district, always mention the district name in your response.
- When answering about personnel by rank, mention both the rank and any district/unit context.
- If no results were found, still mention the entity names from the query (e.g. "No ASI personnel found in Annamayya district").
- For gender-related queries, include the word "female" or "male" as appropriate in your response.
- When listing officers, include their names and posting details, not just counts."""

_ROUTE_TOOLS_REQUIRING_NONEMPTY_ARGS = {
    "search_personnel",
    "search_unit",
    "search_assignment",
}

_TOOL_ALIASES = {
    "count_vacancies_by_unit": "count_vacancies_by_unit_rank",
    "count_vacancies": "count_vacancies_by_unit_rank",
    "vacancies_by_unit": "count_vacancies_by_unit_rank",
}


def _has_nonempty_args(arguments: Dict[str, Any]) -> bool:
    for value in arguments.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value <= 0:
            continue
        return True
    return False


def _has_meaningful_route_args(tool: str, arguments: Dict[str, Any]) -> bool:
    key_fields = {
        "search_personnel": {"name", "user_id", "badge_no", "mobile", "email", "designation_name"},
        "search_unit": {"name", "police_reference_id", "city", "district_name", "unit_type_name"},
        "search_assignment": {"name", "user_id", "unit_id", "unit_name", "post_code"},
    }
    fields = key_fields.get(tool)
    if not fields:
        return _has_nonempty_args(arguments)
    for key in fields:
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        if isinstance(value, (list, dict)):
            if len(value) > 0:
                return True
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if value > 0:
                return True
            continue
        return True
    return False


def _extract_district_from_context(conversation_context: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not conversation_context:
        return None
    for row in reversed(conversation_context):
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "")
        match = re.search(r"\bin\s+([A-Za-z][A-Za-z\s]{1,60}?)\s+district\b", content, re.IGNORECASE)
        if match:
            district = re.sub(r"\s+", " ", match.group(1)).strip()
            if district:
                return district.title()
    return None


def _parse_route_response(
    response: Optional[str],
    query: str,
    *,
    available_tools: Optional[Set[str]] = None,
) -> Optional[Tuple[str, Dict[str, Any], str, float]]:
    if available_tools is None:
        available_tools = _default_available_tools()
    if not response or not str(response).strip():
        return None

    # Strip markdown fences (```json ... ```) that LLMs sometimes wrap around JSON
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Try to extract the outermost JSON object
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        return None

    raw_json = json_match.group()
    result = None
    try:
        result = json.loads(raw_json)
    except json.JSONDecodeError:
        # Aggressive recovery: try fixing common issues
        # 1. Trailing commas before closing brace
        fixed = re.sub(r",\s*}", "}", raw_json)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            result = json.loads(fixed)
        except json.JSONDecodeError:
            # 2. Try extracting just the first complete JSON object
            depth = 0
            start = None
            for i, ch in enumerate(cleaned):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            result = json.loads(cleaned[start : i + 1])
                            break
                        except json.JSONDecodeError:
                            start = None
            if result is None:
                return None

    if not isinstance(result, dict):
        return None

    tool = result.get("tool")
    if not isinstance(tool, str):
        return None
    tool = tool.strip()
    if not tool:
        return None
    tool = _TOOL_ALIASES.get(tool, tool)
    if tool != "__help__" and tool not in available_tools:
        return None

    arguments = result.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None

    if tool in _ROUTE_TOOLS_REQUIRING_NONEMPTY_ARGS and not _has_meaningful_route_args(tool, arguments):
        return None
    if tool == "query_personnel_by_rank" and not any(
        isinstance(arguments.get(key), str) and arguments.get(key).strip()
        for key in ("rank_name", "rank_id")
    ):
        return None

    understood_query = result.get("understood_query")
    if not isinstance(understood_query, str) or not understood_query.strip():
        understood_query = query

    confidence_raw = result.get("confidence", 0.65)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.65
    confidence = max(0.0, min(1.0, confidence))

    return tool, arguments, understood_query, confidence


def _default_available_tools() -> Set[str]:
    try:
        return set(get_tool_handler().get_tool_names())
    except Exception:
        return {
            "search_personnel",
            "search_unit",
            "check_responsible_user",
            "search_assignment",
            "query_personnel_by_unit",
            "query_personnel_by_rank",
            "get_unit_hierarchy",
            "list_units_in_district",
            "list_districts",
            "count_vacancies_by_unit_rank",
            "get_personnel_distribution",
            "query_recent_transfers",
            "get_unit_command_history",
            "find_missing_village_mappings",
            "get_village_coverage",
            "query_linked_master_data",
            "dynamic_query",
        }


async def llm_route_query(
    query: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
    *,
    available_tools: Optional[Iterable[str]] = None,
) -> Tuple[str, Dict[str, Any], str, float, str]:
    available = set(available_tools or _default_available_tools())

    query_lower = str(query or "").lower()
    # Follow-up intent like "what unit are these SIs attached to" should stay on
    # rank listing flow instead of drifting to empty assignment search.
    if (
        "query_personnel_by_rank" in available
        and re.search(r"\b(these|those|their|them)\b", query_lower)
        and re.search(r"\b(si|sis|sub[\s-]?inspectors?)\b", query_lower)
        and re.search(r"\b(unit|attached|attachment|assigned|posted)\b", query_lower)
    ):
        args: Dict[str, Any] = {"rank_name": "Sub-Inspector"}
        district = _extract_district_from_context(conversation_context)
        if district:
            args["district_name"] = district
        return "query_personnel_by_rank", args, query, 0.7, "heuristic_followup"

    def _assignment_is_primary(text: str) -> bool:
        lowered = str(text or "").lower()
        return bool(
            re.search(r"^\s*(?:show|list|get|find)?\s*(?:all\s+)?(?:assignments?|posting|postings)\b", lowered)
            or re.search(r"\b(?:assignments?|posting|posted)\s+(?:for|of|under|in)\b", lowered)
        )

    def _person_lookup_dominant(text: str) -> bool:
        lowered = str(text or "").lower()
        return bool(
            re.search(r"\bwho\s+has\s+user\s*id\b", lowered)
            or re.search(r"\b(?:who\s+is|find|search|lookup)\s+(?:officer|person|personnel)\b", lowered)
            or re.search(r"\btell\s+me\s+about\b", lowered)
            or re.search(r"\b(?:mobile|phone|email|contact|badge)\b", lowered)
        )

    def _extract_person_lookup_args(text: str, seed: Dict[str, Any]) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for key in ("user_id", "mobile", "email", "name"):
            value = (seed or {}).get(key)
            if isinstance(value, str):
                value = value.strip()
            if value:
                args[key] = value

        if "user_id" not in args:
            uid_match = re.search(r"\b(?:user\s*id|userid)\s*[:#-]?\s*(\d{6,12})\b", text, re.IGNORECASE)
            if uid_match:
                args["user_id"] = uid_match.group(1)
        if "mobile" not in args:
            mobile_match = re.search(r"\b(?:mobile|phone)\s*[:#-]?\s*(\d{8,15})\b", text, re.IGNORECASE)
            if mobile_match:
                args["mobile"] = mobile_match.group(1)
        if "email" not in args:
            email_match = re.search(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", text, re.IGNORECASE)
            if email_match:
                args["email"] = email_match.group(1)
        return args

    def _repair_route_choice(
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        normalized_tool = _TOOL_ALIASES.get(str(tool_name or "").strip(), str(tool_name or "").strip())
        normalized_args = dict(arguments or {})
        if (
            normalized_tool == "search_assignment"
            and "search_personnel" in available
            and _person_lookup_dominant(query)
            and not _assignment_is_primary(query)
        ):
            person_args = _extract_person_lookup_args(query, normalized_args)
            if person_args:
                normalized_tool = "search_personnel"
                normalized_args = person_args
        if normalized_tool and normalized_tool not in {"__help__", "search_assignment"}:
            try:
                from mcp.router import repair_route

                repaired_tool, repaired_args = repair_route(query, normalized_tool, normalized_args)
                if isinstance(repaired_tool, str) and repaired_tool.strip():
                    normalized_tool = repaired_tool.strip()
                if isinstance(repaired_args, dict):
                    normalized_args = repaired_args
            except Exception:
                pass
        normalized_tool = _TOOL_ALIASES.get(normalized_tool, normalized_tool)
        if normalized_tool and normalized_tool != "__help__" and normalized_tool not in available:
            return tool_name, arguments
        return normalized_tool or tool_name, normalized_args

    messages: List[Dict[str, str]] = []
    if conversation_context:
        messages.extend(conversation_context[-8:])
    messages.append({"role": "user", "content": query})
    llm_response = await call_openai_api(messages, ROUTER_SYSTEM_PROMPT_V2)

    parsed = _parse_route_response(
        llm_response,
        query,
        available_tools=available,
    )
    if parsed:
        tool, args, understood, confidence = parsed
        repaired_tool, repaired_args = _repair_route_choice(tool, args)
        return repaired_tool, repaired_args, understood, confidence, "llm"

    logger.warning("LLM route response could not be parsed, falling back to heuristic (query=%.80s)", query)

    fallback = fallback_route_query(query, available_tools=available)
    if isinstance(fallback, tuple) and len(fallback) >= 4:
        tool = fallback[0]
        args = fallback[1] if isinstance(fallback[1], dict) else {}
        understood = fallback[2] if isinstance(fallback[2], str) and fallback[2].strip() else query
        try:
            confidence = float(fallback[3])
        except Exception:
            confidence = 0.55
        repaired_tool, repaired_args = _repair_route_choice(tool, args)
        return repaired_tool, repaired_args, understood, confidence, "heuristic_fallback"
    tool, args = fallback if isinstance(fallback, tuple) and len(fallback) >= 2 else ("__help__", {})
    repaired_tool, repaired_args = _repair_route_choice(tool, args)
    return repaired_tool, repaired_args, query, 0.55, "heuristic_fallback"


async def llm_format_response(
    query: str,
    tool_name: str,
    result: Dict[str, Any],
    conversation_context: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    if not has_llm_api_key():
        return None

    payload = json.dumps(
        {"query": query, "tool_name": tool_name, "result": result},
        default=str,
    )
    if len(payload) > 6000:
        payload = payload[:6000] + "...[truncated]"

    messages: List[Dict[str, str]] = []
    if conversation_context:
        messages.extend(conversation_context[-6:])
    messages.append({"role": "user", "content": payload})
    return await call_openai_api(
        messages,
        RESPONSE_FORMATTER_PROMPT_V2,
        max_tokens=512,
    )


def fallback_route_query(
    query: str,
    *,
    available_tools: Optional[Iterable[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    try:
        from mcp.server_http import route_query_to_tool as route_query_to_tool_v2

        tool, args = route_query_to_tool_v2(query)
        if available_tools is None:
            return tool, args
        if tool == "__help__" or tool in set(available_tools):
            return tool, args
    except Exception:
        pass

    tool, args = route_query_to_tool_enriched(query, available_tools=available_tools)
    if not tool:
        return "__help__", {}
    return tool, args


def fallback_format_response(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    return generate_natural_language_response(query, tool_name, arguments, result)


def _capability_help_response_text() -> str:
    return (
        "I can help with personnel, unit, assignment, and responsible-officer lookups.\n\n"
        "- 'SP in Guntur'\n"
        "- 'Mobile number of Ravi Kumar'\n"
        "- 'Where is Kuppam SDPO unit?'\n"
        "- 'Who is in-charge of Guntur PS?'\n"
        "- 'Assignments for user id 14402876'"
    )


class IntelligentQueryHandler:
    """V2 intelligent handler with bounded session memory and TTL."""

    _SESSION_TTL_S = 1800  # 30 minutes idle timeout
    _MAX_LAST_RESULT_ITEMS = 50  # truncate large result lists stored in state

    def __init__(self):
        self.tool_handler = get_tool_handler()
        self.use_llm = has_llm_api_key()
        self.max_history_messages = 12
        self.max_session_contexts = 500
        self._history: Dict[str, Deque[Dict[str, str]]] = OrderedDict()
        self._state: Dict[str, Dict[str, Any]] = OrderedDict()
        self._last_access: Dict[str, float] = {}  # session_id -> monotonic timestamp
        self._routing_modes: Dict[str, str] = {}  # session_id -> "smart_ai" | "mcp_mode"

    def _ensure_session_slot(self, session_id: str) -> None:
        self._evict_expired_sessions()
        if session_id in self._history:
            self._history.move_to_end(session_id)
        else:
            self._history[session_id] = deque(maxlen=self.max_history_messages)
        self._last_access[session_id] = time.monotonic()
        while len(self._history) > self.max_session_contexts:
            oldest = next(iter(self._history))
            self._history.pop(oldest, None)
            self._state.pop(oldest, None)
            self._last_access.pop(oldest, None)
            self._routing_modes.pop(oldest, None)

    def _evict_expired_sessions(self) -> None:
        """Remove sessions that have been idle longer than _SESSION_TTL_S."""
        now = time.monotonic()
        expired = [
            sid for sid, ts in self._last_access.items()
            if now - ts > self._SESSION_TTL_S
        ]
        for sid in expired:
            self._history.pop(sid, None)
            self._state.pop(sid, None)
            self._last_access.pop(sid, None)
            self._routing_modes.pop(sid, None)
        if expired:
            logger.info("Evicted %d expired sessions (TTL=%ds)", len(expired), self._SESSION_TTL_S)

    def _get_history(self, session_id: str) -> Deque[Dict[str, str]]:
        self._ensure_session_slot(session_id)
        return self._history[session_id]

    def _get_state(self, session_id: str) -> Dict[str, Any]:
        self._ensure_session_slot(session_id)
        state = self._state.get(session_id)
        if state is None:
            state = {}
            self._state[session_id] = state
        self._state.move_to_end(session_id)
        while len(self._state) > self.max_session_contexts:
            oldest = next(iter(self._state))
            self._state.pop(oldest, None)
            self._last_access.pop(oldest, None)
        return state

    def get_routing_mode(self, session_id: str) -> str:
        return self._routing_modes.get(session_id or "default", "smart_ai")

    def set_routing_mode(self, session_id: str, mode: str) -> None:
        if mode not in ("smart_ai", "mcp_mode"):
            raise ValueError(f"Unknown routing mode: {mode!r}")
        self._routing_modes[session_id or "default"] = mode

    @classmethod
    def _truncate_last_result(cls, result: Any) -> Any:
        """Truncate large result data to prevent unbounded memory growth."""
        if not isinstance(result, dict):
            return result
        data = result.get("data")
        if isinstance(data, list) and len(data) > cls._MAX_LAST_RESULT_ITEMS:
            truncated = dict(result)
            truncated["data"] = data[: cls._MAX_LAST_RESULT_ITEMS]
            truncated["_truncated_from"] = len(data)
            return truncated
        return result

    def _available_tools(self) -> Set[str]:
        if hasattr(self.tool_handler, "get_tool_names"):
            try:
                names = self.tool_handler.get_tool_names()
                if isinstance(names, list):
                    return set(names) | {"dynamic_query"}
                if isinstance(names, set):
                    return names | {"dynamic_query"}
            except Exception:
                pass
        return _default_available_tools()

    @staticmethod
    def _is_detail_followup(query: str) -> bool:
        """Detect follow-up queries asking for more info about previous results.

        Matches: "give me their info", "show details", "more about them",
        "tell me about them", "their details", "show their details",
        "full info", "details please", etc.
        """
        q = str(query or "").lower().strip()
        return bool(re.search(
            r"\b(?:their|them|his|her|those|these)\b.*\b(?:info|details?|profiles?|data|records?|contacts?|phone|email|mobile)\b"
            r"|\b(?:info|details?|profiles?|full|more|all)\b.*\b(?:about|of|for)\b.*\b(?:them|their|those|these|him|her)\b"
            r"|\b(?:give|show|get|tell|list)\b.*\b(?:all|full|more|complete)\b.*\b(?:info|details?)\b"
            r"|\b(?:more|full|complete|all)\s+(?:info|details?|profiles?|records?)s?\b"
            r"|\b(?:show|give|get|tell)\b.*\b(?:their|them)\b"
            r"|\b(?:show|give|get|list)\b\s+(?:me\s+)?(?:the\s+)?(?:details?|info|profiles?|records?)\b"
            r"|\bdetails?\s+(?:please|pls)\b",
            q,
        ))

    @staticmethod
    def _extract_user_ids_from_result(result: Any) -> List[str]:
        """Extract userId values from a tool result's data field."""
        if not isinstance(result, dict):
            return []
        data = result.get("data", [])
        if not isinstance(data, list):
            return []
        user_ids = []
        for row in data:
            if isinstance(row, dict):
                uid = str(row.get("userId") or row.get("user_id") or row.get("_id") or "").strip()
                if uid:
                    user_ids.append(uid)
        return user_ids[:20]  # cap to avoid huge batch lookups

    @staticmethod
    def _is_attribute_only_followup(query: str) -> bool:
        q = str(query or "").lower()
        return bool(
            re.fullmatch(
                r"\s*(?:mobile|phone|email|contact|dob|date\s+of\s+birth|address|blood(?:\s+group)?|badge|rank|designation)"
                r"(?:\s+number)?\s*\??\s*",
                q,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _normalize_verbose_unit_names(query: str) -> str:
        """Convert verbose unit names to their database short forms.

        Examples:
        - "District Police Office (DPO) - Nandyal city" → "Nandyal DPO"
        - "Police Station (PS) - Kuppam" → "Kuppam PS"
        - "Sub-Divisional Police Office (SDPO), Tirupati" → "Tirupati SDPO"
        """
        # Pattern: "Full Name (ABBR) - Place" or "Full Name (ABBR), Place"
        _VERBOSE_UNIT = re.compile(
            r'(?:District\s+Police\s+Office|Police\s+Station|Sub[- ]?Divisional\s+Police\s+Office|'
            r'Circle\s+Inspector\s+Office|Armed\s+Reserve)\s*'
            r'\(([A-Z]{2,5})\)\s*[-–,]\s*'
            r'([A-Za-z][A-Za-z\s]+?)(?:\s+(?:city|town|village|district))?\s*$',
            re.IGNORECASE,
        )

        def _replace(m: re.Match) -> str:
            abbr = m.group(1).upper()
            place = m.group(2).strip()
            return f"{place} {abbr}"

        result = _VERBOSE_UNIT.sub(_replace, query)

        # Also handle: "DPO Nandyal" → "Nandyal DPO" (abbreviation before place)
        result = re.sub(
            r'\b(DPO|SDPO|PS|UPS)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            lambda m: f"{m.group(2)} {m.group(1)}",
            result,
        )

        return result

    @staticmethod
    def _enrich_followup_query(query: str, state: Dict[str, Any]) -> str:
        """Enrich short contextual follow-ups using previous query state.

        Handles patterns like:
        - "what about Chittoor?" after "SP of Guntur" → "SP of Chittoor"
        - "and in Vizag?" after a district query → reuses last tool with new district
        - "how about DSPs?" after rank query → reuses district with new rank
        """
        q = query.strip()
        last_tool = state.get("last_tool") or ""
        last_args = state.get("last_arguments") or {}

        if not last_tool or not last_args:
            return q

        # Pattern: "what about X?" / "and in X?" / "how about X?" / "and X?"
        swap_match = re.match(
            r"^(?:what|how|and|or)\s+(?:about|in|for)\s+(.+?)\??$",
            q, re.IGNORECASE,
        )
        if not swap_match:
            # Also match bare "and X?" or "X?"
            swap_match = re.match(
                r"^(?:and\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*\??$",
                q,
            )

        if swap_match:
            new_entity = swap_match.group(1).strip()
            # Common rank short-codes for cleaner enriched queries
            _RANK_SHORT = {
                "superintendent of police": "SP",
                "deputy superintendent of police": "DSP",
                "sub-inspector": "SI", "sub inspector": "SI",
                "assistant sub-inspector": "ASI",
                "police constable": "Constable",
                "head constable": "HC",
                "inspector general of police": "IGP",
                "deputy inspector general": "DIG",
            }

            # If last query was district-based, swap district
            if last_args.get("district_name") and last_tool in (
                "query_personnel_by_rank", "list_units_in_district",
                "get_district_info", "query_personnel_by_unit",
                "count_vacancies_by_unit_rank",
            ):
                rank = last_args.get("rank_name", "")
                if rank:
                    short = _RANK_SHORT.get(rank.lower(), rank)
                    return f"{short} of {new_entity}"
                return f"officers in {new_entity}"

            # If last query was rank-based, swap rank
            if last_args.get("rank_name") and last_tool == "query_personnel_by_rank":
                district = last_args.get("district_name", "")
                if district:
                    return f"{new_entity} in {district}"
                return f"list all {new_entity}"

            # If last query was unit-based, try new unit
            if last_args.get("unit_name") and last_tool in (
                "query_personnel_by_unit", "search_unit",
            ):
                return f"officers at {new_entity}"

        return q

    @staticmethod
    def _suggest_alternate_tool(query: str, failed_tool: str, failed_args: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """When a pre-built tool returns empty, suggest an alternate tool to try.

        Returns (alt_tool_name, alt_arguments) or (None, {}).
        """
        # If search_personnel failed, try query_personnel_by_rank with extracted rank/district
        if failed_tool == "search_personnel":
            rank_match = re.search(
                r'\b(SP|DSP|Inspector|Sub[- ]?Inspector|SI|ASI|Constable|PC|HC|IGP|DIG|AIG)\b',
                query, re.IGNORECASE,
            )
            district_match = re.search(
                r'\b(?:in|of|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                query,
            )
            if rank_match:
                args: Dict[str, Any] = {"rank_name": rank_match.group(1).strip()}
                if district_match:
                    args["district_name"] = district_match.group(1).strip()
                return "query_personnel_by_rank", args

        # If query_personnel_by_rank failed, try search_personnel with name-based search
        if failed_tool == "query_personnel_by_rank":
            district = failed_args.get("district_name") or ""
            rank = failed_args.get("rank_name") or ""
            if district:
                return "search_personnel", {"designation_name": rank, "name": district}

        # If query_personnel_by_unit returned empty, try search_unit first to verify unit exists
        if failed_tool == "query_personnel_by_unit":
            unit = failed_args.get("unit_name") or ""
            if unit:
                return "search_unit", {"name": unit}

        # If get_personnel_distribution returned empty, try with district filter
        if failed_tool == "get_personnel_distribution":
            district_match = re.search(
                r'\b(?:in|of|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
                query,
            )
            if district_match:
                return "query_personnel_by_rank", {"district_name": district_match.group(1).strip(), "rank_name": "all"}

        # If get_district_info returned empty, try search_unit with district name
        if failed_tool == "get_district_info":
            district = failed_args.get("district_name") or ""
            if district:
                return "search_unit", {"district_name": district}

        return None, {}

    @staticmethod
    def _split_complex_query_chain(query: str) -> List[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        parts = re.split(r"\s+\band\s+then\b\s+", raw, flags=re.IGNORECASE)
        parts = [part.strip() for part in parts if part and part.strip()]
        return parts if len(parts) >= 2 else []

    @staticmethod
    def _extract_role_and_unit(query: str) -> Tuple[Optional[str], Optional[str]]:
        q = str(query or "").strip()
        if not q:
            return None, None
        patterns = [
            r"\bwho\s+is\s+(?:the\s+)?([A-Za-z][A-Za-z\s\-]{1,30})\s+of\s+([A-Za-z0-9][A-Za-z0-9\s\.\-']{1,120})\??$",
            r"^\s*([A-Za-z][A-Za-z\s\-]{1,30})\s+of\s+([A-Za-z0-9][A-Za-z0-9\s\.\-']{1,120})\??$",
        ]
        for pattern in patterns:
            m = re.search(pattern, q, re.IGNORECASE)
            if not m:
                continue
            role = re.sub(r"\s+", " ", m.group(1)).strip().lower()
            unit = re.sub(r"\s+", " ", m.group(2)).strip(" ?.").strip()
            if role and unit:
                return role, unit
        return None, None

    @staticmethod
    def _normalized_role_key(role: str) -> str:
        role_raw = (role or "").strip().lower()
        aliases = {
            "sp": "sp",
            "superintendent of police": "sp",
            "igp": "igp",
            "inspector general of police": "igp",
            "ao": "ao",
            "administrative officer": "ao",
            "pc": "pc",
            "police constable": "pc",
        }
        return aliases.get(role_raw, role_raw)

    @staticmethod
    def _normalize_unit_name(unit_name: str) -> str:
        acronyms = {"PS", "DPO", "SDPO", "SPDO", "IGP", "AO", "PC", "APSP"}
        tokens = re.split(r"\s+", str(unit_name or "").strip())
        out: List[str] = []
        for token in tokens:
            raw = token.strip()
            if not raw:
                continue
            alpha = re.sub(r"[^A-Za-z]", "", raw).upper()
            if alpha in acronyms:
                out.append(alpha)
            elif re.fullmatch(r"[A-Z]{2,5}", raw):
                out.append(raw)
            else:
                out.append(raw.capitalize())
        return " ".join(out).strip()

    @staticmethod
    def _filter_role_candidates(rows: List[Dict[str, Any]], role_key: str, unit_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []

        def _rank_name(row: Dict[str, Any]) -> str:
            return str(row.get("rankName") or ((row.get("rank") or {}).get("name") if isinstance(row.get("rank"), dict) else "") or "").strip().lower()

        def _rank_code(row: Dict[str, Any]) -> str:
            return str(row.get("rankShortCode") or ((row.get("rank") or {}).get("shortCode") if isinstance(row.get("rank"), dict) else "") or "").strip().upper()

        def _unit_ok(row: Dict[str, Any]) -> bool:
            if not unit_name:
                return True
            row_unit = str(row.get("unitName") or "").strip()
            if not row_unit:
                return True
            return row_unit.lower() == unit_name.lower()

        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not _unit_ok(row):
                continue
            rn = _rank_name(row)
            rc = _rank_code(row)
            if role_key == "sp" and (rc == "SP" or rn == "superintendent of police"):
                filtered.append(row)
            elif role_key == "igp" and ("inspector general of police" in rn or rc == "IGP"):
                filtered.append(row)
            elif role_key == "ao" and ("administrative officer" in rn or rc == "AO"):
                filtered.append(row)
            elif role_key == "pc" and (("police constable" in rn or "constable" in rn) or rc == "PC"):
                filtered.append(row)
            elif role_key and role_key in rn:
                filtered.append(row)
        return filtered

    async def _route(
        self,
        query_text: str,
        llm_context: List[Dict[str, str]],
        available_tools: Set[str],
    ) -> Tuple[str, Dict[str, Any], str, float, str]:
        if self.use_llm:
            try:
                return await llm_route_query(
                    query_text,
                    llm_context,
                    available_tools=available_tools,
                )
            except TypeError:
                # Backward-compatible path for patched/mocked signatures in tests.
                return await llm_route_query(query_text, llm_context)
        fallback = fallback_route_query(query_text, available_tools=available_tools)
        if isinstance(fallback, tuple) and len(fallback) >= 4:
            tool = str(fallback[0] or "__help__")
            args = fallback[1] if isinstance(fallback[1], dict) else {}
            understood = fallback[2] if isinstance(fallback[2], str) and fallback[2].strip() else query_text
            try:
                confidence = float(fallback[3])
            except Exception:
                confidence = 0.55
            return tool, args, understood, confidence, "heuristic_fallback"
        tool, args = fallback if isinstance(fallback, tuple) and len(fallback) >= 2 else ("__help__", {})
        return tool, args, query_text, 0.55, "heuristic_only"

    async def process_query(
        self,
        query: str,
        context: Optional[UserContext],
        *,
        session_id: Optional[str] = None,
        output_format: Optional[str] = None,
        allow_download: Optional[bool] = None,
    ) -> Dict[str, Any]:
        session_key = session_id or "default"
        history = self._get_history(session_key)
        state = self._get_state(session_key)
        context = context or UserContext(scope_level="state")
        clean_query = self._normalize_verbose_unit_names(str(query or "").strip())

        if not clean_query:
            response_text = "Please enter a query."
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=output_format,
                allow_download=allow_download,
            )
            return {
                "success": True,
                "query": clean_query,
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "extracted_arguments": {},
                "data": {},
                "route_source": "empty_query",
                "route_confidence": 1.0,
                "understood_as": "empty_query",
                "output": output_payload,
                "history_size": len(history),
            }

        available_tools = self._available_tools()
        llm_context = list(history)

        # ── dynamic_only mode: bypass all tool routing ──────────────────────
        from mcp.config import mcp_settings
        if mcp_settings.ROUTING_MODE == "dynamic_only":
            from mcp.core.database import get_database
            from mcp.orchestration.dynamic_query_orchestrator import DynamicQueryOrchestrator
            from mcp.tools.dynamic_query_tool import DynamicQueryExecutor

            db = get_database()
            orchestrator = DynamicQueryOrchestrator(DynamicQueryExecutor(db))
            dq_result = await orchestrator.run(
                intent=clean_query,
                context=context,
                conversation_context=None,  # each query is independent — no history bleed
            )
            response_text = dq_result.get("response") or "No answer returned."
            history.append({"role": "user", "content": clean_query})
            history.append({"role": "assistant", "content": response_text})
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to="dynamic_query",
                arguments={},
                result=dq_result,
                requested_format=output_format,
                allow_download=allow_download,
            )
            return {
                "success": dq_result.get("success", True),
                "query": clean_query,
                "response": response_text,
                "routed_to": "dynamic_query",
                "arguments": {},
                "extracted_arguments": {},
                "data": dq_result.get("data", {}),
                "route_source": "dynamic_only",
                "route_confidence": 1.0,
                "understood_as": clean_query,
                "output": output_payload,
                "history_size": len(history),
            }
        # ────────────────────────────────────────────────────────────────────

        # ── mcp_mode: per-session MongoDB MCP simulation ─────────────────
        effective_mode = self.get_routing_mode(session_key)
        if effective_mode == "mcp_mode":
            from mcp.core.database import get_database
            from mcp.core.schema_scanner import generate_mcp_system_prompt, get_schema_info
            from mcp.orchestration.dynamic_query_orchestrator import DynamicQueryOrchestrator
            from mcp.tools.dynamic_query_tool import DynamicQueryExecutor

            db = get_database()
            orchestrator = DynamicQueryOrchestrator(DynamicQueryExecutor(db))
            mcp_prompt = generate_mcp_system_prompt(get_schema_info()).replace(
                "{max_turns}", str(12)
            )
            dq_result = await orchestrator.run_with_prompt(
                intent=clean_query,
                context=context,
                system_prompt=mcp_prompt,
            )
            response_text = dq_result.get("response") or "No answer returned."
            history.append({"role": "user", "content": clean_query})
            history.append({"role": "assistant", "content": response_text})
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to="dynamic_query",
                arguments={},
                result=dq_result,
                requested_format=output_format,
                allow_download=allow_download,
            )
            return {
                "success": dq_result.get("success", True),
                "query": clean_query,
                "response": response_text,
                "routed_to": "dynamic_query",
                "arguments": {},
                "extracted_arguments": {},
                "data": dq_result.get("data", {}),
                "route_source": "mcp_mode",
                "route_confidence": 1.0,
                "understood_as": clean_query,
                "output": output_payload,
                "history_size": len(history),
            }
        # ────────────────────────────────────────────────────────────────────

        # ── hybrid mode: pre-built tools first, dynamic query fallback ────
        if mcp_settings.ROUTING_MODE == "hybrid":
            # ── Follow-up detection: resolve pronouns from previous results ──
            if self._is_detail_followup(clean_query) and state.get("last_result"):
                user_ids = self._extract_user_ids_from_result(state["last_result"])
                if user_ids:
                    logger.info("Hybrid follow-up: resolving %d user IDs from previous result", len(user_ids))
                    all_results: List[Dict[str, Any]] = []
                    for uid in user_ids:
                        person = await self.tool_handler.execute("search_personnel", {"user_id": uid}, context)
                        if isinstance(person, dict) and person.get("data"):
                            pdata = person["data"]
                            if isinstance(pdata, list):
                                all_results.extend(pdata)
                            elif isinstance(pdata, dict):
                                all_results.append(pdata)
                    if all_results:
                        combined_result = {"success": True, "data": all_results}
                        response_text = fallback_format_response(clean_query, "search_personnel", {}, combined_result)
                        if self.use_llm:
                            llm_text = await llm_format_response(clean_query, "search_personnel", combined_result, llm_context)
                            if isinstance(llm_text, str) and llm_text.strip():
                                response_text = llm_text.strip()
                        history.append({"role": "user", "content": clean_query})
                        history.append({"role": "assistant", "content": response_text})
                        state["last_tool"] = "search_personnel"
                        state["last_result"] = self._truncate_last_result(combined_result)
                        output_payload = build_output_payload(
                            query=clean_query,
                            response_text=response_text,
                            routed_to="search_personnel",
                            arguments={},
                            result=combined_result,
                            requested_format=output_format,
                            allow_download=allow_download,
                        )
                        return {
                            "success": True,
                            "query": clean_query,
                            "response": response_text,
                            "routed_to": "search_personnel",
                            "arguments": {},
                            "extracted_arguments": {},
                            "data": all_results,
                            "route_source": "hybrid_followup",
                            "route_confidence": 0.95,
                            "understood_as": clean_query,
                            "output": output_payload,
                            "history_size": len(history),
                        }

            # ── Contextual follow-up: enrich short queries using previous context ──
            enriched_query = self._enrich_followup_query(clean_query, state)

            tool_name, arguments, understood_query, confidence, route_source = await self._route(
                enriched_query, llm_context, available_tools,
            )
            if enriched_query != clean_query:
                logger.info("Hybrid: enriched query %r -> %r", clean_query, enriched_query)
            logger.info("Hybrid router: tool=%s confidence=%.2f source=%s", tool_name, confidence, route_source)

            # Post-routing correction: if the query mentions a specific unit/PS/station
            # but was misrouted, fix it.
            _unit_pattern = re.search(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(DPO|PS|UPS|SDPO|Circle|APSP)\b',
                enriched_query,
            )
            if not _unit_pattern:
                _unit_pattern = re.search(
                    r'(?:at|in|of|from|on)\s+(?:the\s+)?(.+?\b(?:PS|UPS|DPO|Police\s+Station|Station|Circle|SDPO)\b)',
                    enriched_query, re.IGNORECASE,
                )
            if _unit_pattern and tool_name in ("get_personnel_distribution", "dynamic_query", "__help__", "search_personnel"):
                unit_name = _unit_pattern.group(0).strip()
                # Clean up prepositions
                unit_name = re.sub(r'^(?:at|in|of|from|on)\s+(?:the\s+)?', '', unit_name, flags=re.IGNORECASE).strip()
                # Detect intent: "details/info" → search_unit, "personnel/officers/posted" → query_personnel_by_unit
                if re.search(r'\b(personnel|officer|posted|staff|who|list)\b', enriched_query, re.IGNORECASE):
                    logger.info("Hybrid: correcting misroute %s -> query_personnel_by_unit (unit=%s)", tool_name, unit_name)
                    tool_name = "query_personnel_by_unit"
                    arguments = {"unit_name": unit_name}
                    confidence = 0.92
                else:
                    logger.info("Hybrid: correcting misroute %s -> search_unit (name=%s)", tool_name, unit_name)
                    tool_name = "search_unit"
                    arguments = {"name": unit_name}
                    confidence = 0.92

            use_prebuilt = False
            # Confident match to a real pre-built tool → try fast path
            if confidence >= 0.5 and tool_name not in ("dynamic_query", "__help__"):
                from mcp.core.cache import tool_cache

                async def _try_prebuilt(t_name, t_args):
                    """Execute a pre-built tool, using cache. Returns (result, result_data, has_data, source)."""
                    _cached = await tool_cache.get(t_name, t_args)
                    if _cached is not None:
                        _result = _cached
                        _src = f"{route_source}_cached"
                    else:
                        _result = await self.tool_handler.execute(t_name, t_args, context)
                        _src = route_source
                        # Only cache results that have actual data (prevent cache poisoning)
                        _rd = _result.get("data", []) if isinstance(_result, dict) else _result
                        _has = bool(_rd) if isinstance(_rd, (list, dict)) else bool(_result)
                        if _has and isinstance(_result, dict) and _result.get("success", True):
                            await tool_cache.set(t_name, t_args, _result)
                    _rd = _result.get("data", []) if isinstance(_result, dict) else _result
                    _has = bool(_rd) if isinstance(_rd, (list, dict)) else bool(_result)
                    _ok = _result.get("success", True) if isinstance(_result, dict) else True
                    if isinstance(_rd, list) and len(_rd) > 1000:
                        _has = False
                    return _result, _rd, (_has and _ok), _src

                result, result_data, has_data, route_source = await _try_prebuilt(tool_name, arguments)

                # ── Retry-on-empty: try alternate tool before falling to dynamic ──
                if not has_data:
                    alt_tool, alt_args = self._suggest_alternate_tool(clean_query, tool_name, arguments)
                    if alt_tool:
                        logger.info("Hybrid: %s returned empty, retrying with %s(%s)", tool_name, alt_tool, alt_args)
                        result, result_data, has_data, route_source = await _try_prebuilt(alt_tool, alt_args)
                        if has_data:
                            tool_name = alt_tool
                            arguments = alt_args
                            route_source = f"{route_source}_retry"

                if has_data:
                    use_prebuilt = True
                    response_text = fallback_format_response(clean_query, tool_name, arguments, result)
                    # If the formatter returned a generic "No records matched" but data exists,
                    # fall through to dynamic query for better formatting
                    if response_text and "no records matched" in response_text.lower() and result_data:
                        use_prebuilt = False
                        logger.info("Hybrid: pre-built tool %s data exists but formatter returned 'no records matched', falling through", tool_name)
                    elif self.use_llm and tool_name not in {"search_personnel", "search_unit", "check_responsible_user", "search_assignment"}:
                        llm_text = await llm_format_response(clean_query, tool_name, result, llm_context)
                        if isinstance(llm_text, str) and llm_text.strip():
                            response_text = llm_text.strip()
                if use_prebuilt:
                    history.append({"role": "user", "content": clean_query})
                    history.append({"role": "assistant", "content": response_text})
                    # Save state for follow-up queries
                    state["last_tool"] = tool_name
                    state["last_arguments"] = dict(arguments or {})
                    state["last_result"] = self._truncate_last_result(result if isinstance(result, dict) else {"data": result})
                    output_payload = build_output_payload(
                        query=clean_query,
                        response_text=response_text,
                        routed_to=tool_name,
                        arguments=arguments,
                        result=result if isinstance(result, dict) else {"data": result},
                        requested_format=output_format,
                        allow_download=allow_download,
                    )
                    return {
                        "success": True,
                        "query": clean_query,
                        "response": response_text,
                        "routed_to": tool_name,
                        "arguments": arguments,
                        "extracted_arguments": arguments,
                        "data": result_data,
                        "route_source": f"hybrid_tool:{route_source}",
                        "route_confidence": confidence,
                        "understood_as": understood_query,
                        "output": output_payload,
                        "history_size": len(history),
                    }
                else:
                    logger.info("Hybrid: pre-built tool %s returned no data, falling through to dynamic query", tool_name)

            # Low confidence, no matching tool, or empty result → dynamic query fallback
            if not use_prebuilt:
                from mcp.core.database import get_database
                from mcp.orchestration.dynamic_query_orchestrator import DynamicQueryOrchestrator
                from mcp.tools.dynamic_query_tool import DynamicQueryExecutor

                db = get_database()
                orchestrator = DynamicQueryOrchestrator(DynamicQueryExecutor(db))
                dq_result = await orchestrator.run(
                    intent=clean_query,
                    context=context,
                    conversation_context=None,
                )
                response_text = dq_result.get("response") or "No answer returned."
                history.append({"role": "user", "content": clean_query})
                history.append({"role": "assistant", "content": response_text})
                state["last_tool"] = "dynamic_query"
                state["last_arguments"] = {}
                state["last_result"] = self._truncate_last_result(dq_result)
                output_payload = build_output_payload(
                    query=clean_query,
                    response_text=response_text,
                    routed_to="dynamic_query",
                    arguments={},
                    result=dq_result,
                    requested_format=output_format,
                    allow_download=allow_download,
                )
                return {
                    "success": dq_result.get("success", True),
                    "query": clean_query,
                    "response": response_text,
                    "routed_to": "dynamic_query",
                    "arguments": {},
                    "extracted_arguments": {},
                    "data": dq_result.get("data", {}),
                    "route_source": "hybrid_dynamic",
                    "route_confidence": confidence,
                    "understood_as": clean_query,
                    "output": output_payload,
                    "history_size": len(history),
                }
        # ────────────────────────────────────────────────────────────────────

        # Backward-compatible multi-step chain support used by unit tests.
        chain_parts = self._split_complex_query_chain(clean_query)
        if chain_parts:
            step_results: List[Dict[str, Any]] = []
            prev_tool: Optional[str] = None
            prev_args: Dict[str, Any] = {}
            prev_result: Dict[str, Any] = {}
            combined_lines: List[str] = []

            for idx, part in enumerate(chain_parts):
                step_query = part
                step_tool: str
                step_args: Dict[str, Any]
                understood_query: str
                confidence: float
                route_source: str

                if idx > 0 and re.search(r"\bnext\s+page\b", part, re.IGNORECASE) and prev_tool:
                    step_tool = prev_tool
                    step_args = dict(prev_args or {})
                    step_args["page"] = int(step_args.get("page") or 1) + 1
                    understood_query = part
                    confidence = 0.8
                    route_source = "chain_pagination"
                elif idx > 0 and re.search(r"\b(details?|info|profile)\b", part, re.IGNORECASE):
                    ordinal = extract_ordinal_index(part) or extract_list_reference_index(part) or 1
                    rows = prev_result.get("data", []) if isinstance(prev_result, dict) else []
                    chosen = rows[ordinal - 1] if isinstance(rows, list) and 0 < ordinal <= len(rows) else {}
                    user_id = str((chosen or {}).get("userId") or "").strip() if isinstance(chosen, dict) else ""
                    if user_id:
                        step_tool = "search_personnel"
                        step_args = {"user_id": user_id}
                        understood_query = part
                        confidence = 0.85
                        route_source = "chain_ordinal_followup"
                    else:
                        step_tool, step_args, understood_query, confidence, route_source = await self._route(
                            step_query,
                            llm_context,
                            available_tools,
                        )
                else:
                    step_tool, step_args, understood_query, confidence, route_source = await self._route(
                        step_query,
                        llm_context,
                        available_tools,
                    )

                step_result = await self.tool_handler.execute(step_tool, step_args, context)
                step_response = fallback_format_response(step_query, step_tool, step_args, step_result)
                if step_response:
                    combined_lines.append(step_response)
                step_results.append(
                    {
                        "query": step_query,
                        "routed_to": step_tool,
                        "arguments": step_args,
                        "response": step_response,
                        "success": bool(step_result.get("success", True)) if isinstance(step_result, dict) else True,
                        "data": step_result.get("data") if isinstance(step_result, dict) else {},
                        "route_source": route_source,
                        "route_confidence": confidence,
                        "understood_query": understood_query,
                    }
                )
                prev_tool, prev_args, prev_result = step_tool, step_args, step_result

            response_text = "\n\n".join(combined_lines) if combined_lines else "No steps were executed."
            all_success = all(bool(item.get("success", False)) for item in step_results)
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to=step_results[-1].get("routed_to") if step_results else None,
                arguments=step_results[-1].get("arguments", {}) if step_results else {},
                result={"success": all_success, "data": {"steps": step_results}},
                requested_format=output_format,
                allow_download=allow_download,
            )
            history.append({"role": "user", "content": clean_query})
            history.append({"role": "assistant", "content": response_text})
            return {
                "success": all_success,
                "query": clean_query,
                "response": response_text,
                "routed_to": step_results[-1].get("routed_to") if step_results else None,
                "arguments": step_results[-1].get("arguments", {}) if step_results else {},
                "extracted_arguments": step_results[-1].get("arguments", {}) if step_results else {},
                "data": {"steps": step_results},
                "route_source": "complex_query_chain",
                "route_confidence": 0.9,
                "understood_query": "complex_query_chain",
                "understood_as": "complex_query_chain",
                "output": output_payload,
                "history_size": len(history),
            }

        if (
            is_followup_district_query(clean_query)
            and str(state.get("last_tool") or "") == "query_personnel_by_rank"
            and isinstance(state.get("last_arguments"), dict)
            and state["last_arguments"].get("rank_name")
        ):
            tool_name = "query_personnel_by_rank"
            arguments = {"rank_name": state["last_arguments"].get("rank_name")}
            if state["last_arguments"].get("district_name"):
                arguments["district_name"] = state["last_arguments"].get("district_name")
            if state["last_arguments"].get("district_names"):
                arguments["district_names"] = state["last_arguments"].get("district_names")
            understood_query = clean_query
            confidence = 0.85
            route_source = "district_followup_memory"
        elif self._is_attribute_only_followup(clean_query) and state.get("selected_user_id"):
            tool_name = "search_personnel"
            arguments = {"user_id": state["selected_user_id"]}
            understood_query = clean_query
            confidence = 0.85
            route_source = "attribute_followup_memory"
        else:
            tool_name, arguments, understood_query, confidence, route_source = await self._route(
                clean_query,
                llm_context,
                available_tools,
            )

        role_raw, role_unit = self._extract_role_and_unit(clean_query)
        role_key = self._normalized_role_key(role_raw or "")
        # For IGP/AO/PC-like role-holder asks, bypass command-history tool and
        # query the unit roster directly.
        if tool_name == "get_unit_command_history" and role_unit and role_key not in {"", "sp", "sdpo", "sho"}:
            tool_name = "query_personnel_by_unit"
            arguments = {"unit_name": self._normalize_unit_name(role_unit)}

        if tool_name == "__help__":
            reason = str(arguments.get("reason") or "").strip()
            response_text = f"{reason}\n\n{_capability_help_response_text()}".strip() if reason else _capability_help_response_text()
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=output_format,
                allow_download=allow_download,
            )
            history.append({"role": "user", "content": clean_query})
            history.append({"role": "assistant", "content": response_text})
            return {
                "success": True,
                "query": clean_query,
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "extracted_arguments": {},
                "data": {},
                "route_source": route_source,
                "route_confidence": confidence,
                "understood_query": understood_query,
                "understood_as": "capability_help",
                "output": output_payload,
                "history_size": len(history),
            }

        if tool_name == "dynamic_query":
            if confidence >= 0.4 and route_source not in ("heuristic_only", "heuristic_fallback"):
                # Safety guard: LLM is overusing dynamic_query with too much confidence.
                # Skip guard for rule-based routes that intentionally target dynamic_query.
                reason = "I couldn't find a specific tool for your query."
                response_text = f"{reason}\n\n{_capability_help_response_text()}".strip()
            else:
                from mcp.core.database import get_database
                from mcp.orchestration.dynamic_query_orchestrator import DynamicQueryOrchestrator
                from mcp.tools.dynamic_query_tool import DynamicQueryExecutor

                db = get_database()
                orchestrator = DynamicQueryOrchestrator(DynamicQueryExecutor(db))
                dq_result = await orchestrator.run(
                    intent=str(arguments.get("intent") or clean_query),
                    context=context,
                    conversation_context=llm_context,
                )
                response_text = str(
                    dq_result.get("response")
                    or (dq_result.get("data") or {}).get("answer")
                    or "I couldn't find that information in the database."
                )
                output_payload = build_output_payload(
                    query=clean_query,
                    response_text=response_text,
                    routed_to="dynamic_query",
                    arguments=arguments,
                    result=dq_result,
                    requested_format=output_format,
                    allow_download=allow_download,
                )
                history.append({"role": "user", "content": clean_query})
                history.append({"role": "assistant", "content": response_text})
                state["last_tool"] = "dynamic_query"
                state["last_arguments"] = dict(arguments)
                state["last_result"] = self._truncate_last_result(dq_result)
                return {
                    "success": bool(dq_result.get("success", True)),
                    "query": clean_query,
                    "response": response_text,
                    "routed_to": "dynamic_query",
                    "arguments": arguments,
                    "extracted_arguments": arguments,
                    "data": dq_result,
                    "route_source": route_source,
                    "route_confidence": confidence,
                    "understood_query": understood_query,
                    "understood_as": understood_query,
                    "output": output_payload,
                    "history_size": len(history),
                }
            output_payload = build_output_payload(
                query=clean_query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=output_format,
                allow_download=allow_download,
            )
            history.append({"role": "user", "content": clean_query})
            history.append({"role": "assistant", "content": response_text})
            return {
                "success": True,
                "query": clean_query,
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "extracted_arguments": {},
                "data": {},
                "route_source": route_source,
                "route_confidence": confidence,
                "understood_query": understood_query,
                "understood_as": "capability_help",
                "output": output_payload,
                "history_size": len(history),
            }

        result = await self.tool_handler.execute(tool_name, arguments, context)
        force_deterministic_response = False

        # Recover designation prompts routed to rank lookup when rank resolution failed.
        if (
            tool_name == "query_personnel_by_rank"
            and isinstance(result, dict)
            and not result.get("success", True)
        ):
            error = result.get("error", {}) if isinstance(result.get("error"), dict) else {}
            message = str(error.get("message") or "")
            if "rank not found" in message.lower() and re.search(r"\b(?:sdpo|spdo|designation)\b", clean_query, re.IGNORECASE):
                tool_name = "search_personnel"
                arguments = {"designation_name": "SDPO"}
                result = await self.tool_handler.execute(tool_name, arguments, context)

        # Recover role-of-unit asks via unit roster filtering.
        if role_unit and role_key in {"sp", "igp", "ao", "pc"}:
            normalized_unit = self._normalize_unit_name(role_unit)

            async def _resolve_via_unit_filter() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
                unit_tool = "query_personnel_by_unit"
                unit_args = {"unit_name": normalized_unit}
                unit_result = await self.tool_handler.execute(unit_tool, unit_args, context)
                rows = unit_result.get("data", []) if isinstance(unit_result, dict) else []
                candidates = self._filter_role_candidates(rows if isinstance(rows, list) else [], role_key, normalized_unit)
                if len(candidates) == 1:
                    user_id = str(candidates[0].get("userId") or "").strip()
                    if user_id:
                        person_tool = "search_personnel"
                        person_args = {"user_id": user_id}
                        person_result = await self.tool_handler.execute(person_tool, person_args, context)
                        return person_tool, person_args, person_result
                if len(candidates) > 1:
                    total = len(candidates)
                    unit_result = {
                        "success": True,
                        "data": candidates,
                        "pagination": {"page": 1, "page_size": total, "total": total, "total_pages": 1},
                    }
                return unit_tool, unit_args, unit_result

            def _resolve_from_current_unit_result(
                current_result: Dict[str, Any],
            ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
                rows = current_result.get("data", []) if isinstance(current_result, dict) else []
                candidates = self._filter_role_candidates(rows if isinstance(rows, list) else [], role_key, normalized_unit)
                if len(candidates) == 1:
                    user_id = str(candidates[0].get("userId") or "").strip()
                    if user_id:
                        return "search_personnel", {"user_id": user_id}, {}
                if len(candidates) > 1:
                    total = len(candidates)
                    return (
                        "query_personnel_by_unit",
                        {"unit_name": normalized_unit},
                        {
                            "success": True,
                            "data": candidates,
                            "pagination": {"page": 1, "page_size": total, "total": total, "total_pages": 1},
                        },
                    )
                return None

            if tool_name == "get_unit_command_history":
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                command_missing = bool(metadata.get("command_data_missing"))
                if role_key != "sp" or command_missing:
                    tool_name, arguments, result = await _resolve_via_unit_filter()
                    force_deterministic_response = True
            elif tool_name == "query_personnel_by_rank":
                # Try unit-name lookup first (gives the person in the specific unit).
                new_tool, new_args, new_result = await _resolve_via_unit_filter()
                new_rows = new_result.get("data") if isinstance(new_result, dict) else None
                if new_rows:
                    # Unit filter found results — use them.
                    tool_name, arguments, result = new_tool, new_args, new_result
                else:
                    # Unit filter returned nothing (e.g. "Guntur" is a district name, not a unit).
                    # Fall back to the original (already district-scoped) rank result.
                    current_rows = result.get("data", []) if isinstance(result, dict) else []
                    if isinstance(current_rows, list) and current_rows:
                        candidates = self._filter_role_candidates(current_rows, role_key, unit_name=None)
                        if len(candidates) == 1:
                            user_id = str(candidates[0].get("userId") or "").strip()
                            if user_id:
                                pr = await self.tool_handler.execute(
                                    "search_personnel", {"user_id": user_id}, context
                                )
                                tool_name, arguments, result = "search_personnel", {"user_id": user_id}, pr
                        elif len(candidates) > 1:
                            total = len(candidates)
                            result = {
                                "success": True,
                                "data": candidates,
                                "pagination": {"page": 1, "page_size": total, "total": total, "total_pages": 1},
                            }
                        # else: no exact-role match; keep original rank result
                force_deterministic_response = True
            elif tool_name == "query_personnel_by_unit":
                resolved = _resolve_from_current_unit_result(result if isinstance(result, dict) else {})
                if resolved:
                    resolved_tool, resolved_args, resolved_result = resolved
                    if resolved_tool == "search_personnel":
                        person_result = await self.tool_handler.execute(resolved_tool, resolved_args, context)
                        tool_name, arguments, result = resolved_tool, resolved_args, person_result
                    else:
                        tool_name, arguments, result = resolved_tool, resolved_args, resolved_result
                    force_deterministic_response = True
            else:
                # LLM picked a different tool (e.g. search_personnel) for a role-of-unit
                # query — override with the authoritative unit-filter path.
                tool_name, arguments, result = await _resolve_via_unit_filter()
                force_deterministic_response = True

        use_followup_district = bool(is_followup_district_query(clean_query))
        district_response: Optional[str] = None
        if use_followup_district and isinstance(result, dict):
            data_payload = result.get("data")
            if isinstance(data_payload, list):
                district_response = format_followup_district_response(data_payload)

        response_text = district_response or fallback_format_response(clean_query, tool_name, arguments, result)
        if (
            self.use_llm
            and not district_response
            and not force_deterministic_response
            and tool_name not in {"search_personnel", "search_unit", "check_responsible_user", "search_assignment"}
        ):
            llm_text = await llm_format_response(clean_query, tool_name, result, llm_context)
            if isinstance(llm_text, str) and llm_text.strip():
                response_text = llm_text.strip()

        output_payload = build_output_payload(
            query=clean_query,
            response_text=response_text,
            routed_to=tool_name,
            arguments=arguments,
            result=result,
            requested_format=output_format,
            allow_download=allow_download,
        )

        history.append({"role": "user", "content": clean_query})
        history.append({"role": "assistant", "content": response_text})
        state["last_tool"] = tool_name
        state["last_arguments"] = dict(arguments or {})
        state["last_result"] = self._truncate_last_result(result if isinstance(result, dict) else {})
        if isinstance(result, dict) and isinstance(result.get("data"), list):
            rows = result.get("data") or []
            if rows and isinstance(rows[0], dict):
                user_id = str(rows[0].get("userId") or "").strip()
                if user_id:
                    state["selected_user_id"] = user_id

        return {
            "success": bool(result.get("success", True)) if isinstance(result, dict) else True,
            "query": clean_query,
            "response": response_text,
            "routed_to": tool_name,
            "arguments": arguments,
            "extracted_arguments": arguments,
            "data": result,
            "route_source": route_source,
            "route_confidence": confidence,
            "understood_query": understood_query,
            "understood_as": understood_query,
            "output": output_payload,
            "history_size": len(history),
        }


_handler: Optional[IntelligentQueryHandler] = None


def get_intelligent_handler() -> IntelligentQueryHandler:
    global _handler
    if _handler is None:
        _handler = IntelligentQueryHandler()
    return _handler
