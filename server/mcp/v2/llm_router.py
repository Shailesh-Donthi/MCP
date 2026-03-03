"""LLM-powered query routing and response orchestration for MCP V2."""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

from mcp.router.llm_client import call_openai_api, has_llm_api_key
from mcp.schemas.context_schema import UserContext
from mcp.utils.output_layer import build_output_payload
from mcp.v2.handlers.tool_handler import get_tool_handler
from mcp.v2.orchestration import route_query_to_tool_enriched
from mcp.v2.utils import generate_natural_language_response

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
    available_tools: Set[str],
) -> Optional[Tuple[str, Dict[str, Any], str, float]]:
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

    def _repair_route_choice(
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        normalized_tool = _TOOL_ALIASES.get(str(tool_name or "").strip(), str(tool_name or "").strip())
        normalized_args = dict(arguments or {})
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

    if has_llm_api_key():
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

    tool, args = fallback_route_query(query, available_tools=available)
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
        from mcp.v2.server_http import route_query_to_tool as route_query_to_tool_v2

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
                "output": output_payload,
                "history_size": len(history),
            }

        available_tools = set(self.tool_handler.get_tool_names())
        llm_context = list(history)

        if self.use_llm:
            tool_name, arguments, understood_query, confidence, route_source = await llm_route_query(
                clean_query,
                llm_context,
                available_tools=available_tools,
            )
        else:
            tool_name, arguments = fallback_route_query(clean_query, available_tools=available_tools)
            understood_query = clean_query
            confidence = 0.55
            route_source = "heuristic_only"

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
                "output": output_payload,
                "history_size": len(history),
            }

        result = await self.tool_handler.execute(tool_name, arguments, context)

        response_text = fallback_format_response(clean_query, tool_name, arguments, result)
        if self.use_llm and tool_name not in {"search_personnel", "search_unit", "check_responsible_user", "search_assignment"}:
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
            "output": output_payload,
            "history_size": len(history),
        }


_handler: Optional[IntelligentQueryHandler] = None


def get_intelligent_handler() -> IntelligentQueryHandler:
    global _handler
    if _handler is None:
        _handler = IntelligentQueryHandler()
    return _handler
