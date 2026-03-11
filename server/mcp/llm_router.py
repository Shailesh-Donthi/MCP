"""LLM-powered query routing and response orchestration for MCP V2."""

from __future__ import annotations

import json
import logging
import re
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
1) search_personnel
   Args: name, user_id, badge_no, mobile, email, designation_name, district_name, include_inactive, page, page_size
2) search_unit
   Args: name, police_reference_id, city, district_name, page, page_size
3) check_responsible_user
   Args: name, user_id
4) search_assignment
   Args: name, user_id, unit_id, unit_name, post_code, include_inactive, page, page_size
5) query_personnel_by_unit
   Args: unit_name, unit_id, group_by_rank
6) query_personnel_by_rank
   Args: rank_name, rank_id, rank_relation, district_name
7) get_unit_hierarchy
   Args: unit_name, unit_id, district_name
8) list_units_in_district
   Args: district_name
9) list_districts
   Args: name
10) count_vacancies_by_unit_rank
    Args: unit_name, district_name
11) get_personnel_distribution
    Args: group_by, district_name
12) query_recent_transfers
    Args: days, district_name
13) get_unit_command_history
    Args: unit_name, unit_id
14) find_missing_village_mappings
    Args: district_name
15) get_village_coverage
    Args: unit_name, district_name
16) query_linked_master_data
    Args: mode, collection, filters, search_text, include_related, include_reverse, include_integrity
17) dynamic_query
    Use ONLY when no tool above can answer the question. Requires a plain-English description.
    Args: intent (required — describe what data to find)
    IMPORTANT: Only pick this tool if confidence would be below 0.4. Never use for queries tools 1-16 can handle.

Return exactly one JSON object:
{
  "tool": "tool_name",
  "arguments": {},
  "understood_query": "short summary",
  "confidence": 0.0
}

Rules:
- Output JSON only.
- Use one tool only.
- If unsure, choose the closest tool with partial arguments.
- Never invent unsupported tools.
- Confidence between 0 and 1.
- Only use dynamic_query if no tool 1-16 fits AND confidence is below 0.4.
"""

RESPONSE_FORMATTER_PROMPT_V2 = """You are a concise assistant. Use only the provided tool result.
If the result is empty, say no records matched.
Keep answer factual and short."""

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

    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        return None

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
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

    confidence_raw = result.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.5
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
    # Backward-compatible strict retry behavior without context.
    retry_messages = [{"role": "user", "content": query}]
    retry_response = await call_openai_api(retry_messages, ROUTER_SYSTEM_PROMPT_V2)
    retry_parsed = _parse_route_response(
        retry_response,
        query,
        available_tools=available,
    )
    if retry_parsed:
        tool, args, understood, confidence = retry_parsed
        repaired_tool, repaired_args = _repair_route_choice(tool, args)
        return repaired_tool, repaired_args, understood, confidence, "llm_retry_no_context"

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
    """V2 intelligent handler with bounded session memory."""

    def __init__(self):
        self.tool_handler = get_tool_handler()
        self.use_llm = has_llm_api_key()
        self.max_history_messages = 12
        self.max_session_contexts = 500
        self._history: Dict[str, Deque[Dict[str, str]]] = OrderedDict()
        self._state: Dict[str, Dict[str, Any]] = OrderedDict()

    def _ensure_session_slot(self, session_id: str) -> None:
        if session_id in self._history:
            self._history.move_to_end(session_id)
        else:
            self._history[session_id] = deque(maxlen=self.max_history_messages)
        while len(self._history) > self.max_session_contexts:
            oldest = next(iter(self._history))
            self._history.pop(oldest, None)

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
        return state

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
        "tell me about them", "their details", "full info", etc.
        """
        q = str(query or "").lower().strip()
        return bool(re.search(
            r"\b(?:their|them|his|her|those|these)\b.*\b(?:info|detail|profile|data|record|contact|phone|email|mobile)\b"
            r"|\b(?:info|detail|profile|full|more|all)\b.*\b(?:about|of|for)\b.*\b(?:them|their|those|these|him|her)\b"
            r"|\b(?:give|show|get|tell|list)\b.*\b(?:all|full|more|complete)\b.*\b(?:info|detail)\b"
            r"|\b(?:more|full|complete|all)\s+(?:info|detail|profile|record)s?\b",
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
        clean_query = str(query or "").strip()

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
                        state["last_result"] = combined_result
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

            tool_name, arguments, understood_query, confidence, route_source = await self._route(
                clean_query, llm_context, available_tools,
            )
            logger.info("Hybrid router: tool=%s confidence=%.2f source=%s", tool_name, confidence, route_source)

            use_prebuilt = False
            # High-confidence match to a real pre-built tool → try fast path
            if confidence >= 0.7 and tool_name not in ("dynamic_query", "__help__"):
                result = await self.tool_handler.execute(tool_name, arguments, context)
                # Check if the tool returned useful, focused data
                result_data = result.get("data", []) if isinstance(result, dict) else result
                has_data = bool(result_data) if isinstance(result_data, (list, dict)) else bool(result)
                result_success = result.get("success", True) if isinstance(result, dict) else True
                # If the tool returned too many results, it likely didn't filter properly
                if isinstance(result_data, list) and len(result_data) > 1000:
                    has_data = False

                if has_data and result_success:
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
                    state["last_result"] = result if isinstance(result, dict) else {"data": result}
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
                state["last_result"] = dq_result
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
            if confidence >= 0.4:
                # Safety guard: router is overusing dynamic_query with too much confidence.
                # Fall back to capability help rather than running an unconstrained DB query.
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
                state["last_result"] = dq_result
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
        state["last_result"] = result if isinstance(result, dict) else {}
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
