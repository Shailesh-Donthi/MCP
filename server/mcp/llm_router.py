"""LLM-powered query routing and response orchestration."""

import json
import logging
import re
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

from mcp.handlers.tool_handler import get_tool_handler
from mcp.utils.output_layer import build_output_payload
from mcp.router import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_STRICT_SYSTEM_PROMPT,
    RESPONSE_FORMATTER_PROMPT,
    call_claude_api,
    call_openai_api,
    repair_route,
    needs_clarification,
    is_followup_district_query,
    format_followup_district_response,
)
from mcp.router.llm_client import has_llm_api_key
from mcp.router.extractors import (
    extract_place_hint,
    extract_rank_hint,
    extract_unit_hint,
    extract_person_hint,
    extract_user_id_hint,
    extract_mobile_hint,
    extract_ordinal_index,
    extract_list_reference_index,
    normalize_common_query_typos,
)

logger = logging.getLogger(__name__)

_ROUTE_TOOLS_REQUIRING_NONEMPTY_ARGS = {
    "search_personnel",
    "search_unit",
    "query_personnel_by_unit",
    "query_personnel_by_rank",
    "list_units_in_district",
    "get_unit_hierarchy",
    "count_vacancies_by_unit_rank",
    "query_recent_transfers",
    "get_unit_command_history",
    "get_village_coverage",
    "query_linked_master_data",
}


def _has_nonempty_args(arguments: Dict[str, Any]) -> bool:
    for value in arguments.values():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        return True
    return False


def _parse_route_response(
    response: Optional[str],
    query: str,
) -> Optional[Tuple[str, Dict[str, Any], str, float]]:
    if not response or not str(response).strip():
        return None

    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        return None

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM router response JSON: %s", exc)
        return None

    if not isinstance(result, dict):
        return None

    tool = result.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    tool = tool.strip()

    arguments = result.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None

    if tool in _ROUTE_TOOLS_REQUIRING_NONEMPTY_ARGS and not _has_nonempty_args(arguments):
        return None

    understood_query = result.get("understood_query")
    if not isinstance(understood_query, str) or not understood_query.strip():
        understood_query = query

    confidence_raw = result.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return tool, arguments, understood_query, confidence


async def llm_route_query(
    query: str,
    conversation_context: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any], str, float, str]:
    messages: List[Dict[str, str]] = []
    if conversation_context:
        messages.extend(conversation_context[-8:])
    messages.append({"role": "user", "content": query})

    response = await call_claude_api(messages, ROUTER_SYSTEM_PROMPT)
    if not response:
        response = await call_openai_api(messages, ROUTER_SYSTEM_PROMPT)

    parsed = _parse_route_response(response, query)
    if parsed:
        tool, args, understood_query, confidence = parsed
        return tool, args, understood_query, confidence, "llm"

    # Retry once with stricter output instructions and no conversation context.
    strict_messages = [{"role": "user", "content": query}]
    retry_response = await call_claude_api(strict_messages, ROUTER_STRICT_SYSTEM_PROMPT)
    if not retry_response:
        retry_response = await call_openai_api(strict_messages, ROUTER_STRICT_SYSTEM_PROMPT)
    retry_parsed = _parse_route_response(retry_response, query)
    if retry_parsed:
        tool, args, understood_query, confidence = retry_parsed
        return tool, args, understood_query, confidence, "llm_retry_no_context"

    logger.warning("LLM routing invalid/empty after retry, using fallback")

    tool, args, understood_query, confidence = fallback_route_query(query)
    return tool, args, understood_query, confidence, "heuristic_fallback"


async def llm_format_response(
    query: str,
    tool_name: str,
    result: Dict[str, Any],
    conversation_context: Optional[List[Dict[str, str]]] = None,
) -> str:
    data_summary = json.dumps(result, indent=2, default=str)
    if len(data_summary) > 4000:
        data_summary = data_summary[:4000] + "\n... (truncated)"

    prompt = (
        f"User's Question: {query}\n\n"
        f"Tool Used: {tool_name}\n\n"
        f"Data Retrieved:\n{data_summary}\n\n"
        "Please provide a natural language response to answer the user's question based on this data."
    )

    messages: List[Dict[str, str]] = []
    if conversation_context:
        messages.extend(conversation_context[-8:])
    messages.append({"role": "user", "content": prompt})

    response = await call_claude_api(messages, RESPONSE_FORMATTER_PROMPT, max_tokens=2048)
    if not response:
        response = await call_openai_api(messages, RESPONSE_FORMATTER_PROMPT, max_tokens=2048)
    if response:
        return response
    return fallback_format_response(query, tool_name, {}, result)


def fallback_route_query(query: str) -> Tuple[str, Dict[str, Any], str, float]:
    from mcp.server_http import route_query_to_tool

    tool, args = route_query_to_tool(query)
    return tool, args, query, 0.5


def fallback_format_response(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    from mcp.utils.formatters import generate_natural_language_response

    return generate_natural_language_response(query, tool_name, arguments or {}, result)


def _capability_help_response_text() -> str:
    return (
        "I can help with police personnel, unit reporting, and linked master-data queries. Try asking:\n\n"
        "- 'What is the mobile number of A Ashok Kumar?'\n"
        "- 'List all SIs in Chittoor district'\n"
        "- 'How many personnel are in Guntur district?'\n"
        "- 'What is the unit hierarchy for Chittoor district?'\n"
        "- 'List units in Guntur district'\n"
        "- 'Show notification master entries linked to modules'\n"
        "- 'Which villages are mapped to K V Palli PS?'\n"
        "- 'Show recent transfers in the last 30 days'\n"
        "- 'Who is the SDPO of Kuppam?'\n\n"
        "After list results, you can ask follow-ups like 'next page', 'previous page', or 'show details of the 1st one'."
    )


class IntelligentQueryHandler:
    """Handles natural language queries using LLM + deterministic repair logic."""

    def __init__(self):
        self.tool_handler = get_tool_handler()
        self.use_llm = has_llm_api_key()
        self.max_history_messages = 12
        try:
            self.max_session_contexts = max(
                50,
                int(__import__("os").getenv("MCP_MAX_SESSION_CONTEXTS", "500")),
            )
        except Exception:
            self.max_session_contexts = 500
        # Bounded per-session caches (LRU by session access) to avoid unbounded memory growth.
        self._history: Dict[str, deque[Dict[str, str]]] = OrderedDict()
        # Structured session state for follow-up robustness.
        self._session_state: Dict[str, Dict[str, Any]] = OrderedDict()
        self._deterministic_format_tools = {
            "search_personnel",
            "list_units_in_district",
            "list_districts",
            "get_personnel_distribution",
            "query_personnel_by_rank",
            "get_unit_command_history",
            "get_unit_hierarchy",
            "query_linked_master_data",
        }

    def _ensure_session_slot(self, session_id: str) -> None:
        if session_id in self._history:
            cast_history = self._history
            if hasattr(cast_history, "move_to_end"):
                cast_history.move_to_end(session_id)  # type: ignore[attr-defined]
        else:
            self._history[session_id] = deque(maxlen=self.max_history_messages)

        if session_id in self._session_state:
            cast_state = self._session_state
            if hasattr(cast_state, "move_to_end"):
                cast_state.move_to_end(session_id)  # type: ignore[attr-defined]
        else:
            self._session_state[session_id] = {}

        while len(self._history) > self.max_session_contexts:
            oldest = next(iter(self._history))
            self._history.pop(oldest, None)
            self._session_state.pop(oldest, None)

        while len(self._session_state) > self.max_session_contexts:
            oldest = next(iter(self._session_state))
            self._session_state.pop(oldest, None)
            self._history.pop(oldest, None)

    def _get_history(self, session_id: str) -> deque[Dict[str, str]]:
        self._ensure_session_slot(session_id)
        return self._history[session_id]

    def _get_state(self, session_id: str) -> Dict[str, Any]:
        self._ensure_session_slot(session_id)
        return self._session_state[session_id]

    def _update_state(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        state = self._get_state(session_id)
        state["last_tool"] = tool_name
        state["last_arguments"] = dict(arguments or {})
        state["last_result"] = result
        state["last_rank"] = arguments.get("rank_name") or extract_rank_hint((arguments.get("rank_name") or ""))
        state["last_place"] = arguments.get("district_name") or extract_place_hint((arguments.get("district_name") or ""))

        data_list = result.get("data", []) if isinstance(result, dict) else []
        list_user_ids: List[str] = []
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict):
                    uid = item.get("userId")
                    if isinstance(uid, str) and uid.strip():
                        list_user_ids.append(uid.strip())
        state["last_list_user_ids"] = list_user_ids[:100]
        self._set_last_list_context(state, tool_name, result)

        # Track last resolved person so pronoun follow-ups can remain grounded.
        if tool_name == "search_personnel" and isinstance(data_list, list) and data_list:
            first = data_list[0] if isinstance(data_list[0], dict) else {}
            if isinstance(first, dict):
                last_user_id = first.get("userId")
                last_name = first.get("name")
                if isinstance(last_user_id, str) and last_user_id.strip():
                    state["last_person_user_id"] = last_user_id.strip()
                if isinstance(last_name, str) and last_name.strip():
                    state["last_person_name"] = last_name.strip()

        # Track units from hierarchy/list/detail tools for follow-up "details on X".
        unit_names = self._extract_unit_names_from_result(tool_name, result)
        if unit_names:
            state["last_unit_names"] = unit_names[:300]

        self._update_selected_entity_from_result(state, tool_name, result)
        if isinstance(result, dict) and not result.get("success", False):
            state["last_error"] = result.get("error")
        elif isinstance(result, dict):
            state.pop("last_error", None)

    def _extract_unit_names_from_result(self, tool_name: str, result: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        data = result.get("data") if isinstance(result, dict) else None
        if not data:
            return out

        def add_name(value: Any) -> None:
            if isinstance(value, str):
                v = value.strip()
                if v and v not in out:
                    out.append(v)

        def walk_hierarchy(node: Any) -> None:
            if isinstance(node, dict):
                add_name(node.get("name"))
                children = node.get("children")
                if isinstance(children, list):
                    for child in children:
                        walk_hierarchy(child)
            elif isinstance(node, list):
                for item in node:
                    walk_hierarchy(item)

        if tool_name == "get_unit_hierarchy":
            if isinstance(data, dict):
                if isinstance(data.get("units"), list):
                    walk_hierarchy(data["units"])
                else:
                    walk_hierarchy(data)
        elif tool_name in {"list_units_in_district", "search_unit"}:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        add_name(item.get("name"))
            elif isinstance(data, dict):
                add_name(data.get("name"))
        elif tool_name == "get_unit_command_history" and isinstance(data, dict):
            add_name(data.get("unitName"))
        return out

    def _infer_entity_type_from_item(self, item: Dict[str, Any], source_tool: str) -> str:
        if not isinstance(item, dict):
            return "unknown"
        if item.get("userId") or source_tool in {"search_personnel", "query_personnel_by_rank", "query_personnel_by_unit"}:
            if item.get("userId") or item.get("badgeNo") or item.get("rank"):
                return "personnel"
        if item.get("district_id") or item.get("district_name"):
            if source_tool == "list_districts" or ("district_id" in item and "district_name" in item and len(item.keys()) <= 4):
                return "district"
        unitish_keys = {
            "unitId", "unitName", "policeReferenceId", "unitType", "responsibleUserName", "responsibleOfficer",
            "personnelCount", "villageCount",
        }
        if source_tool in {
            "list_units_in_district", "search_unit", "find_missing_village_mappings", "get_village_coverage",
            "get_unit_hierarchy", "get_unit_command_history",
        } or any(k in item for k in unitish_keys):
            return "unit"
        if "districtName" in item and ("count" in item or "totalPersonnel" in item):
            return "district"
        return "unknown"

    def _list_item_label(self, item: Dict[str, Any], entity_type: str) -> str:
        if entity_type == "personnel":
            return str(item.get("name") or item.get("userId") or "Unknown")
        if entity_type == "district":
            return str(item.get("district_name") or item.get("districtName") or item.get("name") or "Unknown")
        return str(item.get("name") or item.get("unitName") or item.get("districtName") or "Unknown")

    def _extract_list_context_items(self, tool_name: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        data = result.get("data")
        items: List[Dict[str, Any]] = []
        raw_items: List[Dict[str, Any]] = []

        if isinstance(data, list):
            raw_items = [item for item in data if isinstance(item, dict)]
        elif tool_name == "get_unit_hierarchy" and isinstance(data, dict):
            hierarchy_roots = data.get("units") if isinstance(data.get("units"), list) else [data]
            for root in hierarchy_roots:
                if isinstance(root, dict):
                    raw_items.append(root)
        elif isinstance(data, dict):
            # Some tools return a single structured unit/person object.
            raw_items = [data]

        pagination = result.get("pagination") if isinstance(result.get("pagination"), dict) else {}
        page = int((pagination or {}).get("page") or 1)
        page_size = int((pagination or {}).get("page_size") or max(1, len(raw_items) or 1))
        base_index = (max(page, 1) - 1) * max(page_size, 1)

        for idx, item in enumerate(raw_items, start=1):
            entity_type = self._infer_entity_type_from_item(item, tool_name)
            entry: Dict[str, Any] = {
                "entity_type": entity_type,
                "page_index": idx,
                "absolute_index": base_index + idx,
                "label": self._list_item_label(item, entity_type),
            }
            if entity_type == "personnel":
                if isinstance(item.get("userId"), str):
                    entry["user_id"] = item.get("userId")
                if isinstance(item.get("name"), str):
                    entry["name"] = item.get("name")
                if isinstance(item.get("_id"), str):
                    entry["id"] = item.get("_id")
            elif entity_type == "district":
                district_name = item.get("district_name") or item.get("districtName") or item.get("name")
                district_id = item.get("district_id") or item.get("districtId") or item.get("_id")
                if isinstance(district_name, str):
                    entry["district_name"] = district_name
                    entry["name"] = district_name
                if isinstance(district_id, str):
                    entry["district_id"] = district_id
                    entry["id"] = district_id
            else:
                unit_name = item.get("name") or item.get("unitName")
                unit_id = item.get("_id") or item.get("unitId")
                district_name = item.get("districtName") or (
                    item.get("district", {}).get("name") if isinstance(item.get("district"), dict) else None
                )
                if isinstance(unit_name, str):
                    entry["unit_name"] = unit_name
                    entry["name"] = unit_name
                if isinstance(unit_id, str):
                    entry["unit_id"] = unit_id
                    entry["id"] = unit_id
                if isinstance(district_name, str):
                    entry["district_name"] = district_name
            items.append(entry)
        return items

    def _set_last_list_context(self, state: Dict[str, Any], tool_name: str, result: Dict[str, Any]) -> None:
        items = self._extract_list_context_items(tool_name, result)
        if not items:
            return
        pagination = result.get("pagination") if isinstance(result, dict) else None
        entity_type = items[0].get("entity_type") if items else "unknown"
        list_context = {
            "source_tool": tool_name,
            "entity_type": entity_type,
            "page": int((pagination or {}).get("page") or 1) if isinstance(pagination, dict) else 1,
            "page_size": int((pagination or {}).get("page_size") or len(items) or 1) if isinstance(pagination, dict) else (len(items) or 1),
            "total": int((pagination or {}).get("total") or len(items)) if isinstance(pagination, dict) else len(items),
            "items": items[:200],
        }
        state["last_list_context"] = list_context
        # Add list position indices to response metadata so clients/debug logs can inspect stable references.
        if isinstance(result, dict):
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                result["metadata"] = metadata
            metadata["list_context"] = {
                "entity_type": list_context["entity_type"],
                "source_tool": list_context["source_tool"],
                "page": list_context["page"],
                "page_size": list_context["page_size"],
                "total": list_context["total"],
                "items": [
                    {
                        "page_index": item.get("page_index"),
                        "absolute_index": item.get("absolute_index"),
                        "label": item.get("label"),
                        "entity_type": item.get("entity_type"),
                    }
                    for item in items[:50]
                ],
            }

    def _update_selected_entity_from_result(self, state: Dict[str, Any], tool_name: str, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict) or not result.get("success", False):
            return
        items = self._extract_list_context_items(tool_name, result)
        if len(items) == 1:
            state["selected_entity"] = dict(items[0])
            if items[0].get("entity_type") == "unit" and isinstance(items[0].get("unit_name"), str):
                state["last_unit_name"] = items[0]["unit_name"]
            if items[0].get("entity_type") == "district" and isinstance(items[0].get("district_name"), str):
                state["last_place"] = items[0]["district_name"]
            if items[0].get("entity_type") == "personnel":
                if isinstance(items[0].get("user_id"), str):
                    state["last_person_user_id"] = items[0]["user_id"]
                if isinstance(items[0].get("name"), str):
                    state["last_person_name"] = items[0]["name"]
            return

        data = result.get("data")
        if tool_name == "get_unit_command_history" and isinstance(data, dict):
            unit_name = data.get("unitName")
            unit_id = data.get("unitId")
            if isinstance(unit_name, str) and unit_name.strip():
                state["selected_entity"] = {
                    "entity_type": "unit",
                    "unit_name": unit_name.strip(),
                    "unit_id": unit_id if isinstance(unit_id, str) else None,
                    "label": unit_name.strip(),
                }
                state["last_unit_name"] = unit_name.strip()

    def _get_last_list_item_by_ordinal(self, state: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        idx = extract_ordinal_index(query) or extract_list_reference_index(query)
        if not isinstance(idx, int) or idx < 1:
            return None
        list_context = state.get("last_list_context")
        if not isinstance(list_context, dict):
            return None
        items = list_context.get("items")
        if not isinstance(items, list):
            return None
        if idx <= len(items):
            item = items[idx - 1]
            return item if isinstance(item, dict) else None
        return None

    def _query_mentions_person_detail(self, q: str) -> bool:
        return bool(__import__("re").search(
            r"\b(person|personnel|officer|their|them|they|his|her|email|e-?mail|mobile|phone|dob|date of birth|address|blood group|badge|rank|designation|details?)\b",
            q,
        ))

    def _query_mentions_unit_or_officer(self, q: str) -> bool:
        return bool(__import__("re").search(
            r"\b(unit|station|ps|sho|sdpo|spdo|dpo|in charge|command history|villages?|mapping|coverage|where is|details?)\b",
            q,
        ))

    def _query_mentions_district(self, q: str) -> bool:
        return bool(__import__("re").search(r"\b(district|districts|units in|hierarchy)\b", q))

    def _inject_state_hints(self, session_id: str, query: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        state = self._get_state(session_id)
        q = (query or "").lower()
        new_args = dict(arguments or {})
        explicit_rank_in_query = bool(extract_rank_hint(query))
        explicit_place_in_query = bool(extract_place_hint(query))
        explicit_unit_in_query = bool(extract_unit_hint(query))
        explicit_person_in_query = bool(extract_person_hint(query))
        explicit_user_in_query = bool(extract_user_id_hint(query))
        ordinal_item = self._get_last_list_item_by_ordinal(state, q)
        if ordinal_item:
            state["selected_entity"] = dict(ordinal_item)
            entity_type = ordinal_item.get("entity_type")
            if entity_type == "personnel":
                if not new_args.get("user_id") and ordinal_item.get("user_id"):
                    new_args["user_id"] = ordinal_item["user_id"]
                elif not new_args.get("name") and ordinal_item.get("name"):
                    new_args["name"] = ordinal_item["name"]
            elif entity_type == "unit":
                if self._query_mentions_unit_or_officer(q):
                    if not new_args.get("unit_name") and ordinal_item.get("unit_name"):
                        new_args["unit_name"] = ordinal_item["unit_name"]
                    if not new_args.get("name") and ordinal_item.get("unit_name"):
                        new_args["name"] = ordinal_item["unit_name"]
                if not new_args.get("district_name") and ordinal_item.get("district_name"):
                    new_args["district_name"] = ordinal_item["district_name"]
            elif entity_type == "district":
                if not new_args.get("district_name") and ordinal_item.get("district_name"):
                    new_args["district_name"] = ordinal_item["district_name"]

        if not new_args.get("user_id"):
            ids = state.get("last_list_user_ids") or []
            idx = extract_ordinal_index(q) or extract_list_reference_index(q)
            if isinstance(idx, int) and 1 <= idx <= len(ids):
                new_args["user_id"] = ids[idx - 1]

        selected = state.get("selected_entity")
        if isinstance(selected, dict) and __import__("re").search(r"\b(their|them|they|that one|this one|that unit|this unit|there)\b", q):
            selected_type = selected.get("entity_type")
            if selected_type == "personnel":
                # Avoid leaking prior selected-person hints into standalone
                # queries that already carry explicit rank/place/unit targets.
                if not (explicit_rank_in_query or explicit_place_in_query or explicit_unit_in_query or explicit_person_in_query or explicit_user_in_query):
                    if not new_args.get("user_id") and selected.get("user_id"):
                        new_args["user_id"] = selected["user_id"]
                    if not new_args.get("name") and selected.get("name"):
                        new_args["name"] = selected["name"]
            elif selected_type == "unit":
                selected_unit_name = selected.get("unit_name")
                # For deictic follow-ups ("there", "that unit"), prefer the
                # previously selected concrete unit over any fuzzy LLM value.
                if isinstance(selected_unit_name, str) and selected_unit_name.strip():
                    if not explicit_unit_in_query:
                        new_args["unit_name"] = selected_unit_name
                        new_args["name"] = selected_unit_name
                    else:
                        if not new_args.get("unit_name"):
                            new_args["unit_name"] = selected_unit_name
                        if not new_args.get("name"):
                            new_args["name"] = selected_unit_name
                if not new_args.get("district_name") and selected.get("district_name"):
                    new_args["district_name"] = selected["district_name"]
            elif selected_type == "district":
                if not new_args.get("district_name") and selected.get("district_name"):
                    new_args["district_name"] = selected["district_name"]

        if not new_args.get("rank_name") and __import__("re").search(r"\b(their|them|they|those|these)\b", q):
            if state.get("last_rank"):
                new_args["rank_name"] = state["last_rank"]
        if (
            not new_args.get("district_name")
            and not explicit_place_in_query
            and __import__("re").search(r"\b(their|them|they|those|these)\b", q)
        ):
            if state.get("last_place"):
                new_args["district_name"] = state["last_place"]
        return new_args

    def _is_followup_person_detail_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        # If the query already carries explicit rank/place/unit targets, treat
        # it as a standalone request instead of a pronoun follow-up.
        if extract_rank_hint(query) or extract_place_hint(query) or extract_unit_hint(query):
            return False
        has_pronoun = bool(
            __import__("re").search(r"\b(their|them|they|his|her|that person|this person)\b", q)
        )
        asks_details = bool(
            __import__("re").search(
                r"\b(details?|info|information|profile|contact|email|mobile|phone|dob|date of birth|address|badge(?:\s+number)?|blood\s+group|rank|designation)\b",
                q,
            )
            or "all their details" in q
            or "all details" in q
        )
        # Let existing district follow-up logic handle district/belonging intents.
        district_intent = bool(__import__("re").search(r"\b(districts?|belong|belongs|belonging)\b", q))
        return has_pronoun and asks_details and not district_intent

    def _is_attribute_only_person_followup_query(self, query: str, state: Dict[str, Any]) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        if not state.get("last_person_user_id") and not state.get("last_person_name"):
            return False
        # If user already specified a target, let normal routing handle it.
        if extract_person_hint(query) or extract_user_id_hint(query) or extract_mobile_hint(query):
            return False
        if __import__("re").search(r"\b(?:sho|sdpo|spdo|unit|station|ps|district|villages?|mapping|coverage)\b", q):
            return False
        attr_only = bool(
            __import__("re").search(
                r"^\s*(?:what\s+is\s+(?:the\s+)?)?"
                r"(?:mobile(?:\s+number)?|phone(?:\s+number)?|contact(?:\s+number)?|email|e-?mail|"
                r"dob|date\s+of\s+birth|birthday|address|blood\s+group|badge(?:\s+number)?|rank|designation)"
                r"\s*\??\s*$",
                q,
            )
        )
        return attr_only

    def _is_unit_detail_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        return bool(
            __import__("re").search(r"\b(details?|info|information|about|on)\b", q)
            and __import__("re").search(r"\b(unit|station|ps|dpo|circle|range|wing|office|sub division|sub-division)\b", q)
        )

    def _is_followup_unit_personnel_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        asks_personnel = bool(__import__("re").search(r"\b(personnel|staff|officers?|people)\b", q))
        # Handle typo-heavy prompts like "ist the details of the personnel there"
        # by not requiring a strict action verb.
        asks_details = bool(__import__("re").search(r"\b(details?|info|information)\b", q))
        has_action = bool(__import__("re").search(r"\b(list|show|get|give|who|which|tell|provide|find)\b", q))
        refers_place = bool(
            __import__("re").search(r"\b(there|here|that unit|this unit|that station|this station|in that|in this)\b", q)
        )
        return asks_personnel and refers_place and (has_action or asks_details)

    def _is_capability_help_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
            return False
        return bool(
            __import__("re").search(
                r"\b(what\s+can\s+you\s+help\s+me\s+with|what\s+reports?\s+can\s+i\s+ask\s+for|what\s+can\s+i\s+ask|help(?:\s+me)?)\b",
                q,
            )
        )

    def _normalize_unit_key(self, text: str) -> str:
        t = (text or "").lower()
        t = __import__("re").sub(r"\([^)]*\)", " ", t)  # remove parenthetical suffix
        t = __import__("re").sub(r"\b(the|a|an|details?|info|information|about|on|of|for)\b", " ", t)
        t = __import__("re").sub(r"[^a-z0-9\s]", " ", t)
        t = __import__("re").sub(r"\s+", " ", t).strip()
        return t

    def _resolve_followup_unit_name(self, query: str, state: Dict[str, Any]) -> Optional[str]:
        # First, try explicit unit hint extraction.
        direct = extract_unit_hint(query)
        if direct:
            return direct

        # Then try a generic "details/info on/about <unit-like phrase>" capture.
        m = __import__("re").search(
            r"\b(?:details?|info|information)\s+(?:on|about|of|for)\s+([A-Za-z0-9\.\-\s]+?)(?:\s*\([^)]*\))?(?:\?|$)",
            query,
            __import__("re").IGNORECASE,
        )
        if m:
            candidate = __import__("re").sub(r"\s+", " ", m.group(1)).strip()
            candidate = __import__("re").sub(r"^(?:the|a|an)\s+", "", candidate, flags=__import__("re").IGNORECASE).strip()
            # Reject generic placeholders that are not unit names.
            if __import__("re").search(
                r"\b(personnel|staff|officers?|people|there|here|their|them|those|these)\b",
                candidate,
                __import__("re").IGNORECASE,
            ):
                candidate = ""
            if candidate:
                return candidate

        candidates = state.get("last_unit_names") or []
        if not isinstance(candidates, list) or not candidates:
            return None

        q_key = self._normalize_unit_key(query)
        if not q_key:
            return None

        def score(name: str) -> int:
            n_key = self._normalize_unit_key(name)
            if not n_key:
                return -999
            s = 0
            if n_key in q_key:
                s += 10
            if q_key in n_key:
                s += 8
            q_tokens = set(q_key.split())
            n_tokens = set(n_key.split())
            overlap = len(q_tokens & n_tokens)
            s += overlap * 2
            # prefer tighter names on tie
            s -= max(0, len(n_tokens) - overlap)
            return s

        ranked = sorted(
            [c for c in candidates if isinstance(c, str) and c.strip()],
            key=score,
            reverse=True,
        )
        if not ranked:
            return None
        best = ranked[0]
        if score(best) <= 0:
            return None
        return best

    def _pagination_delta(self, query: str) -> int:
        q = (query or "").strip().lower()
        if not q:
            return 0
        if __import__("re").search(r"\bnext\s*page\b|\bpage\s*next\b|\bgo\s*to\s*next\s*page\b", q):
            return 1
        if __import__("re").search(r"\bprevious\s*page\b|\bprev\s*page\b|\bpage\s*previous\b|\bgo\s*to\s*previous\s*page\b", q):
            return -1
        return 0

    def _extract_output_preference(self, query: str) -> Optional[str]:
        q = (query or "").lower().strip()
        if not q:
            return None
        if not __import__("re").search(r"\b(prefer|default|always|from now on)\b", q):
            return None
        if "json" in q:
            return "json"
        if __import__("re").search(r"\btable|tabular\b", q):
            return "table"
        if __import__("re").search(r"\btree\b", q):
            return "tree"
        if __import__("re").search(r"\btext|plain text|txt\b", q):
            return "text"
        if __import__("re").search(r"\bpie\b", q):
            return "pie"
        if __import__("re").search(r"\bline\b", q):
            return "line"
        if __import__("re").search(r"\bbar\b", q):
            return "bar"
        if __import__("re").search(r"\bchart|graph|plot|visual\b", q):
            return "bar"
        return None

    def _is_output_preference_query(self, query: str) -> bool:
        q = (query or "").lower()
        return bool(
            self._extract_output_preference(query)
            and __import__("re").search(r"\b(prefer|default|always|from now on)\b", q)
        )

    def _split_complex_query_chain(self, query: str) -> List[str]:
        q = (query or "").strip()
        if not q:
            return []
        # Support explicit chain syntax only; avoid splitting natural phrases like
        # "SHO and SDPO of Kuppam" which should still clarify.
        normalized = q.replace("\n", " ").strip()
        if not __import__("re").search(r"\b(and then|then)\b|->|[+]|;", normalized, __import__("re").IGNORECASE):
            return []
        parts = __import__("re").split(
            r"\s*(?:->|;+\s*|\band then\b|\bthen\b|\s+\+\s+)\s*",
            normalized,
            flags=__import__("re").IGNORECASE,
        )
        cleaned = [p.strip(" .") for p in parts if p and p.strip(" .")]
        if len(cleaned) < 2:
            return []
        return cleaned[:3]

    def _apply_ordinal_route_overrides(
        self,
        query: str,
        tool_name: str,
        arguments: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        item = self._get_last_list_item_by_ordinal(state, query)
        if not item:
            return tool_name, arguments
        args = dict(arguments or {})
        entity_type = item.get("entity_type")
        q = (query or "").lower()
        if entity_type == "personnel" and self._query_mentions_person_detail(q):
            if item.get("user_id"):
                args["user_id"] = item["user_id"]
            elif item.get("name"):
                args["name"] = item["name"]
            return "search_personnel", args
        if entity_type == "unit":
            if __import__("re").search(r"\b(command history|history)\b", q):
                if item.get("unit_name"):
                    args["unit_name"] = item["unit_name"]
                return "get_unit_command_history", args
            if __import__("re").search(r"\b(villages?|mapping|coverage)\b", q):
                if item.get("unit_name"):
                    args["unit_name"] = item["unit_name"]
                return "get_village_coverage", args
            if __import__("re").search(r"\b(sho|sdpo|spdo|dpo|in charge|heads?)\b", q):
                if item.get("unit_name"):
                    args["unit_name"] = item["unit_name"]
                return "get_unit_command_history", args
            if item.get("unit_name"):
                args["name"] = item["unit_name"]
            return "search_unit", args
        if entity_type == "district":
            if item.get("district_name"):
                args["district_name"] = item["district_name"]
            if __import__("re").search(r"\b(units?|stations?)\b", q):
                return "list_units_in_district", args
            if __import__("re").search(r"\b(hierarchy|tree)\b", q):
                return "get_unit_hierarchy", args
            if __import__("re").search(r"\b(personnel|rank|distribution)\b", q):
                return "get_personnel_distribution", args
            if __import__("re").search(r"\b(details?|info|information|about|on)\b", q):
                return "list_units_in_district", args
        return tool_name, args

    def _extract_error_signature(self, result: Dict[str, Any]) -> Tuple[Optional[str], str]:
        error = result.get("error") if isinstance(result, dict) else None
        if not isinstance(error, dict):
            return None, ""
        code = error.get("code")
        message = str(error.get("message") or "")
        if not isinstance(code, str):
            code = None
        return code, message

    async def _recover_failed_result(
        self,
        *,
        query: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        state: Dict[str, Any],
        context: Optional[Any],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        if not isinstance(result, dict) or result.get("success", False):
            return tool_name, arguments, result
        code, message = self._extract_error_signature(result)
        args = dict(arguments or {})

        # Reuse selected entity when the routed tool failed due to missing target args.
        if (code and "MCP-INPUT-001" in code) or (code == "MISSING_PARAMETER"):
            selected = state.get("selected_entity")
            if isinstance(selected, dict):
                if selected.get("entity_type") == "personnel" and self._query_mentions_person_detail(query):
                    retry_args = dict(args)
                    if selected.get("user_id"):
                        retry_args["user_id"] = selected["user_id"]
                    elif selected.get("name"):
                        retry_args["name"] = selected["name"]
                    retry_result = await self.tool_handler.execute("search_personnel", retry_args, context)
                    if isinstance(retry_result, dict) and retry_result.get("success", False):
                        return "search_personnel", retry_args, retry_result
                if selected.get("entity_type") == "unit":
                    retry_args = dict(args)
                    unit_name = selected.get("unit_name")
                    if isinstance(unit_name, str) and unit_name.strip():
                        retry_args.setdefault("unit_name", unit_name.strip())
                        retry_args.setdefault("name", unit_name.strip())
                    if __import__("re").search(r"\b(villages?|mapping|coverage)\b", query, __import__("re").IGNORECASE):
                        retry_result = await self.tool_handler.execute("get_village_coverage", retry_args, context)
                        if isinstance(retry_result, dict) and retry_result.get("success", False):
                            return "get_village_coverage", retry_args, retry_result

        # Alias-normalize district/unit args on NOT_FOUND and retry once.
        if code == "NOT_FOUND" or "not found" in message.lower():
            retry_args = dict(args)
            changed = False
            for key in ("district_name", "unit_name", "name"):
                value = retry_args.get(key)
                if isinstance(value, str):
                    normalized = normalize_common_query_typos(value).strip()
                    if normalized and normalized != value:
                        retry_args[key] = normalized
                        changed = True
            if changed:
                retry_result = await self.tool_handler.execute(tool_name, retry_args, context)
                if isinstance(retry_result, dict) and retry_result.get("success", False):
                    return tool_name, retry_args, retry_result

        # Recover designation-style SDPO/SPDO prompts that were incorrectly
        # routed as rank lookups by LLM/heuristics.
        if (
            tool_name == "query_personnel_by_rank"
            and ("rank not found" in message.lower())
        ):
            failed_rank = str(args.get("rank_name") or "").strip().lower()
            if failed_rank in {"sdpo", "spdo"}:
                q = (query or "").lower()
                if (
                    "designation" in q
                    or "post" in q
                    or "role" in q
                    or __import__("re").search(r"\blist\s+(?:all\s+)?(?:sdpo|spdo)s?\b", q)
                ):
                    retry_args = {"designation_name": "SDPO"}
                    retry_result = await self.tool_handler.execute("search_personnel", retry_args, context)
                    if isinstance(retry_result, dict) and retry_result.get("success", False):
                        return "search_personnel", retry_args, retry_result

        # If a generic person search failed/empty for a place-like term, try district->unit fallback.
        if tool_name == "search_personnel":
            candidate_name = (args or {}).get("name")
            if isinstance(candidate_name, str) and candidate_name.strip():
                district_probe = await self.tool_handler.execute("list_districts", {"name": candidate_name}, context)
                districts = district_probe.get("data", []) if isinstance(district_probe, dict) else []
                if district_probe.get("success", False) and isinstance(districts, list) and districts:
                    district_name = districts[0].get("district_name") or candidate_name
                    retry_args = {"district_name": district_name}
                    retry_result = await self.tool_handler.execute("list_units_in_district", retry_args, context)
                    if isinstance(retry_result, dict) and retry_result.get("success", False):
                        return "list_units_in_district", retry_args, retry_result

        return tool_name, args, result

    def _extract_leader_rank_hint(self, query: str) -> Optional[str]:
        q = (query or "").lower()
        if not q:
            return None
        if re.search(r"\b(superintendent\s+of\s+police|sp)\b", q):
            return "Superintendent of Police"
        if re.search(r"\b(deputy\s+superintendent\s+of\s+police|dysp|dsp|sdpo|spdo)\b", q):
            return "Deputy Superintendent of Police"
        return None

    def _extract_district_from_unit_name(self, unit_name: str) -> Optional[str]:
        raw = re.sub(r"\s+", " ", str(unit_name or "")).strip()
        if not raw:
            return None
        m = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9\s\.\-']{1,80}?)\s+"
            r"(?:dpo|district\s+police\s+office|sdpo|spdo|ps|police\s+station|station|range|circle|wing|office)\b",
            raw,
            re.IGNORECASE,
        )
        if not m:
            return None
        district = re.sub(r"\s+", " ", m.group(1)).strip(" .-")
        return district.title() if district else None

    @staticmethod
    def _norm_token_text(value: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _looks_like_unit_target(value: str) -> bool:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return False
        if re.search(r"\bdistrict\b", text, re.IGNORECASE) and not re.search(
            r"\b(dpo|gpo|sdpo|spdo|ps|station|sub\s+division|division|circle|range|wing|office|btn|battalion|apsp)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        return bool(
            re.search(
                r"\b(dpo|gpo|sdpo|spdo|ps|police\s+station|station|sub\s+division|division|circle|range|wing|office|btn|battalion|apsp)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _extract_role_of_unit_query(self, query: str) -> Optional[Tuple[str, str]]:
        text = re.sub(r"\s+", " ", str(query or "")).strip()
        if not text:
            return None
        patterns = [
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)\s+(?:the\s+)?([A-Za-z0-9\[\]\(\),/&\-\s\.]{1,60})\s+of\s+([A-Za-z0-9\[\]\(\),/&\-\s\.]{2,120})(?:\?|$)",
            r"^\s*([A-Za-z0-9\[\]\(\),/&\-\s\.]{1,60})\s+of\s+([A-Za-z0-9\[\]\(\),/&\-\s\.]{2,120})(?:\?|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            role_text = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" ?.")
            unit_text = re.sub(r"\s+", " ", (m.group(2) or "")).strip(" ?.")
            role_text = re.sub(r"^(?:the|a|an)\s+", "", role_text, flags=re.IGNORECASE).strip()
            unit_text = re.sub(r"^(?:the|a|an)\s+", "", unit_text, flags=re.IGNORECASE).strip()
            unit_text = re.sub(r"\bSPDO\b", "SDPO", unit_text, flags=re.IGNORECASE)
            unit_text = re.sub(r"\bGPO\b", "DPO", unit_text, flags=re.IGNORECASE)
            if not role_text or not unit_text:
                continue
            if not self._looks_like_unit_target(unit_text):
                continue
            return role_text, unit_text
        return None

    @staticmethod
    def _extract_role_keys(text: str) -> set[str]:
        value = re.sub(r"\s+", " ", str(text or "").lower()).strip()
        if not value:
            return set()
        keys: set[str] = set()
        patterns: List[Tuple[str, str]] = [
            (r"\b(deputy\s+inspector\s+general\s+of\s+police|dig)\b", "dig"),
            (r"\b(inspector\s+general\s+of\s+police|igp)\b", "igp"),
            (r"\b(addl\.?\s*sp|additional\s+superintendent\s+of\s+police|asp)\b", "addl_sp"),
            (r"\b(superintendent\s+of\s+police|sp)\b", "sp"),
            (r"\b(deputy\s+superintendent\s+of\s+police|dysp|dsp|sdpo|spdo)\b", "dsp"),
            (r"\b(circle\s+inspector|ci)\b", "ci"),
            (r"\b(sub[\s-]?inspector|si)\b", "si"),
            (r"\b(assistant\s+sub[\s-]?inspector|asi)\b", "asi"),
            (r"\b(head\s+constable|hc)\b", "hc"),
            (r"\b(police\s+constable|pc)\b", "pc"),
            (r"\b(administrative\s+officer|ao)\b", "ao"),
            (r"\b(office\s+superintendent|os)\b", "os"),
            (r"\b(jr\s*/\s*sr\s*asst|jr\s*sr\s*asst|junior\s+office\s+assistant|joa)\b", "joa"),
            (r"\b(public\s+relations\s+officer|pro)\b", "pro"),
            (r"\b(inspector\s+of\s+police|inspector)\b", "inspector"),
            (r"\b(sho)\b", "sho"),
        ]
        for pattern, key in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                keys.add(key)
        if "addl_sp" in keys:
            keys.discard("sp")
        if "asi" in keys:
            keys.discard("si")
        return keys

    def _row_role_keys(self, row: Dict[str, Any]) -> set[str]:
        if not isinstance(row, dict):
            return set()
        rank_name = str(row.get("rankName") or "")
        rank_code = str(row.get("rankShortCode") or "")
        return self._extract_role_keys(f"{rank_name} {rank_code}")

    def _row_matches_requested_role(self, row: Dict[str, Any], requested_keys: set[str]) -> bool:
        if not requested_keys:
            return False
        row_keys = self._row_role_keys(row)
        if not row_keys:
            return False
        if requested_keys & row_keys:
            return True
        if "sho" in requested_keys and row_keys & {"si", "ci", "inspector"}:
            return True
        return False

    def _extract_forced_senior_role_unit(self, query: str) -> Optional[Tuple[str, str]]:
        parsed = self._extract_role_of_unit_query(query or "")
        if not parsed:
            return None
        role_text, unit_name = parsed
        keys = self._extract_role_keys(role_text)
        if not ({"igp", "dig"} & keys):
            return None
        if not re.search(r"\b(range|office)\b", unit_name, re.IGNORECASE):
            return None
        return role_text, unit_name

    async def _recover_role_of_unit_with_personnel_lookup(
        self,
        *,
        query: str,
        context: Optional[Any],
    ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        parsed = self._extract_role_of_unit_query(query or "")
        if not parsed:
            return None
        role_text, unit_name = parsed
        requested_keys = self._extract_role_keys(role_text)
        if not requested_keys:
            return None

        unit_args: Dict[str, Any] = {"unit_name": unit_name, "page_size": 500}
        unit_result = await self.tool_handler.execute("query_personnel_by_unit", unit_args, context)
        if not isinstance(unit_result, dict) or not unit_result.get("success", False):
            return None
        rows = unit_result.get("data")
        if not isinstance(rows, list) or not rows:
            return None

        candidates = [row for row in rows if isinstance(row, dict) and self._row_matches_requested_role(row, requested_keys)]
        if not candidates:
            return None

        # Ambiguous role-in-unit mapping: return candidate list rather than
        # silently choosing an arbitrary first record.
        if len(candidates) > 1:
            ordered = sorted(
                candidates,
                key=lambda row: (
                    str((row or {}).get("name") or "").lower(),
                    str((row or {}).get("userId") or ""),
                ),
            )
            list_result: Dict[str, Any] = {
                "success": True,
                "query_type": "personnel_by_unit",
                "data": ordered,
                "pagination": {
                    "page": 1,
                    "page_size": len(ordered),
                    "total": len(ordered),
                    "total_pages": 1,
                },
                "metadata": {
                    "disambiguation": True,
                    "role_text": role_text,
                    "unit_name": unit_name,
                    "candidate_count": len(ordered),
                },
            }
            return "query_personnel_by_unit", {"unit_name": unit_name}, list_result

        chosen = candidates[0]
        chosen_user_id = chosen.get("userId")
        if not isinstance(chosen_user_id, str) or not chosen_user_id.strip():
            return None

        person_args = {"user_id": chosen_user_id.strip()}
        person_result = await self.tool_handler.execute("search_personnel", person_args, context)
        if isinstance(person_result, dict) and person_result.get("success", False):
            return "search_personnel", person_args, person_result
        return None

    async def _recover_missing_command_with_personnel_lookup(
        self,
        *,
        query: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        context: Optional[Any],
    ) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        if not isinstance(result, dict) or not result.get("success", False):
            return None
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if not metadata.get("command_data_missing"):
            return None

        rank_name = self._extract_leader_rank_hint(query or "")
        if not rank_name:
            return None

        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        unit_name = (
            (data.get("unitName") if isinstance(data, dict) else None)
            or metadata.get("resolved_unit_name")
            or arguments.get("unit_name")
        )
        if not isinstance(unit_name, str) or not unit_name.strip():
            return None

        rank_args: Dict[str, Any] = {
            "rank_name": rank_name,
            "rank_relation": "exact",
            "page_size": 200,
        }
        district_name = self._extract_district_from_unit_name(unit_name)
        if district_name:
            rank_args["district_name"] = district_name

        rank_result = await self.tool_handler.execute("query_personnel_by_rank", rank_args, context)
        if not isinstance(rank_result, dict) or not rank_result.get("success", False):
            return None
        rows = rank_result.get("data")
        if not isinstance(rows, list) or not rows:
            return None

        target_unit = self._norm_token_text(unit_name)
        exact_unit_rows: List[Dict[str, Any]] = []
        loose_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_unit = self._norm_token_text(row.get("unitName") or row.get("primary_unit"))
            if not row_unit:
                continue
            if row_unit == target_unit:
                exact_unit_rows.append(row)
            elif target_unit and (target_unit in row_unit or row_unit in target_unit):
                loose_rows.append(row)
        candidates = exact_unit_rows or loose_rows or rows
        chosen = candidates[0] if candidates else None
        if not isinstance(chosen, dict):
            return None

        chosen_user_id = chosen.get("userId")
        if isinstance(chosen_user_id, str) and chosen_user_id.strip():
            person_args = {"user_id": chosen_user_id.strip()}
            person_result = await self.tool_handler.execute("search_personnel", person_args, context)
            if isinstance(person_result, dict) and person_result.get("success", False):
                return "search_personnel", person_args, person_result

        return None

    async def process_query(
        self,
        query: str,
        context: Optional[Any] = None,
        session_id: Optional[str] = None,
        output_format: Optional[str] = None,
        allow_download: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # Refresh each request so .env-backed key changes are picked up without
        # requiring a process restart of the singleton handler.
        self.use_llm = has_llm_api_key()
        session_key = session_id or "default"
        original_query = query or ""
        normalized_query = normalize_common_query_typos(original_query)
        history = self._get_history(session_key)
        context_messages = list(history)
        state = self._get_state(session_key)
        effective_output_format = (
            state.get("preferred_output_format")
            if (output_format is None or str(output_format).lower() == "auto")
            else output_format
        )

        last_user_query = next((m.get("content") for m in reversed(context_messages) if m.get("role") == "user"), None)
        last_assistant_response = next((m.get("content") for m in reversed(context_messages) if m.get("role") == "assistant"), None)

        if self._is_output_preference_query(normalized_query):
            preferred = self._extract_output_preference(normalized_query)
            if preferred:
                state["preferred_output_format"] = preferred
                response_text = f"Okay. I'll prefer `{preferred}` format for future results in this chat unless you ask otherwise."
                output_payload = build_output_payload(
                    query=original_query,
                    response_text=response_text,
                    routed_to=None,
                    arguments={},
                    result={"preference": {"output_format": preferred}},
                    requested_format="text",
                    allow_download=False,
                )
                history.append({"role": "user", "content": original_query})
                history.append({"role": "assistant", "content": response_text})
                return {
                    "success": True,
                    "query": original_query,
                    "understood_as": "set_output_preference",
                    "response": response_text,
                    "routed_to": None,
                    "arguments": {"preferred_output_format": preferred},
                    "confidence": 1.0,
                    "data": {"preferred_output_format": preferred},
                    "llm_enabled": self.use_llm,
                    "route_source": "state_preference",
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

        chain_parts = self._split_complex_query_chain(normalized_query)
        if chain_parts:
            step_results: List[Dict[str, Any]] = []
            all_success = True
            for idx, part in enumerate(chain_parts):
                is_last = idx == len(chain_parts) - 1
                step = await self.process_query(
                    part,
                    context=context,
                    session_id=session_key,
                    output_format=effective_output_format if is_last else None,
                    allow_download=allow_download if is_last else False,
                )
                step_results.append({
                    "step": idx + 1,
                    "query": part,
                    "success": bool(step.get("success")),
                    "routed_to": step.get("routed_to"),
                    "response": step.get("response"),
                    "arguments": step.get("arguments"),
                    "data": step.get("data"),
                })
                if not step.get("success"):
                    all_success = False
                    break

            combined_lines = []
            for step in step_results:
                combined_lines.append(
                    f"Step {step['step']} ({step['query']}):\n{step.get('response') or 'No response'}"
                )
            response_text = "\n\n".join(combined_lines) if combined_lines else "No steps were executed."
            last_step = step_results[-1] if step_results else {}
            output_payload = build_output_payload(
                query=original_query,
                response_text=response_text,
                routed_to=last_step.get("routed_to"),
                arguments=last_step.get("arguments") or {},
                result={"success": all_success, "data": {"steps": step_results}},
                requested_format=effective_output_format,
                allow_download=allow_download,
            )
            return {
                "success": all_success,
                "query": original_query,
                "understood_as": "complex_query_chain",
                "response": response_text,
                "routed_to": last_step.get("routed_to"),
                "arguments": last_step.get("arguments") or {},
                "confidence": 0.95,
                "data": {"steps": step_results},
                "llm_enabled": self.use_llm,
                "route_source": "chain_executor",
                "session_id": session_key,
                "history_size": len(history),
                "output": output_payload,
            }

        page_delta = self._pagination_delta(normalized_query)
        if page_delta != 0:
            last_tool = state.get("last_tool")
            last_args = dict(state.get("last_arguments") or {})
            last_result = state.get("last_result") or {}
            pagination = last_result.get("pagination") if isinstance(last_result, dict) else None

            if not last_tool or not isinstance(pagination, dict):
                response_text = "No paginated result is active yet. Run a list query first."
                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=None,
                    arguments={},
                    result={},
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                return {
                    "success": True,
                    "query": query,
                    "understood_as": "pagination_unavailable",
                    "response": response_text,
                    "routed_to": None,
                    "arguments": {},
                    "confidence": 1.0,
                    "data": {},
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }
            current_page = int(pagination.get("page") or 1)
            total_pages = max(1, int(pagination.get("total_pages") or 1))
            target_page = min(max(1, current_page + page_delta), total_pages)

            if target_page == current_page:
                boundary = "last" if page_delta > 0 else "first"
                response_text = f"You are already on the {boundary} page ({current_page}/{total_pages})."
                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=last_tool,
                    arguments=last_args,
                    result=last_result,
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                return {
                    "success": True,
                    "query": query,
                    "understood_as": "pagination_boundary",
                    "response": response_text,
                    "routed_to": last_tool,
                    "arguments": last_args,
                    "confidence": 1.0,
                    "data": last_result,
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

            page_args = dict(last_args)
            page_args["page"] = target_page
            result = await self.tool_handler.execute(last_tool, page_args, context)

            if is_followup_district_query(normalized_query):
                data_payload = result.get("data", []) if isinstance(result, dict) else []
                district_response = format_followup_district_response(data_payload)
                response_text = district_response or fallback_format_response(query, last_tool, page_args, result)
            elif last_tool in self._deterministic_format_tools:
                response_text = fallback_format_response(query, last_tool, page_args, result)
            elif self.use_llm:
                response_text = await llm_format_response(query, last_tool, result, context_messages)
            else:
                response_text = fallback_format_response(query, last_tool, page_args, result)

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            if isinstance(result, dict) and result.get("success", False):
                self._update_state(session_key, last_tool, page_args, result)

            output_payload = build_output_payload(
                query=query,
                response_text=response_text,
                routed_to=last_tool,
                arguments=page_args,
                result=result,
                requested_format=effective_output_format,
                allow_download=allow_download,
            )
            return {
                "success": bool(result.get("success")) if isinstance(result, dict) else True,
                "query": query,
                "understood_as": f"pagination_page_{target_page}",
                "response": response_text,
                "routed_to": last_tool,
                "arguments": page_args,
                "confidence": 1.0,
                "data": result,
                "llm_enabled": self.use_llm,
                "session_id": session_key,
                "history_size": len(history),
                "output": output_payload,
            }

        if self._is_capability_help_query(normalized_query):
            response_text = _capability_help_response_text()
            output_payload = build_output_payload(
                query=query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=effective_output_format,
                allow_download=allow_download,
            )
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            return {
                "success": True,
                "query": query,
                "understood_as": "capability_help",
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "confidence": 1.0,
                "data": {},
                "llm_enabled": self.use_llm,
                "session_id": session_key,
                "history_size": len(history),
                "output": output_payload,
            }

        # Follow-up person details like "give me all their details" should resolve
        # to the last referenced person from session state.
        if self._is_followup_person_detail_query(normalized_query):
            person_args: Dict[str, Any] = {}
            if isinstance(state.get("last_person_user_id"), str) and state.get("last_person_user_id"):
                person_args["user_id"] = state["last_person_user_id"]
            elif isinstance(state.get("last_person_name"), str) and state.get("last_person_name"):
                person_args["name"] = state["last_person_name"]

            if person_args:
                tool_name = "search_personnel"
                arguments = person_args
                understood_query = query
                confidence = 1.0
                result = await self.tool_handler.execute(tool_name, arguments, context)

                if self.use_llm and tool_name not in self._deterministic_format_tools:
                    response_text = await llm_format_response(query, tool_name, result, context_messages)
                else:
                    response_text = fallback_format_response(query, tool_name, arguments, result)

                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                if isinstance(result, dict) and result.get("success", False):
                    self._update_state(session_key, tool_name, arguments, result)

                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=tool_name,
                    arguments=arguments,
                    result=result,
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                return {
                    "success": bool(result.get("success")) if isinstance(result, dict) else True,
                    "query": query,
                    "understood_as": understood_query,
                    "response": response_text,
                    "routed_to": tool_name,
                    "arguments": arguments,
                    "confidence": confidence,
                    "data": result,
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

        # Attribute-only follow-ups like "What is the mobile number?" should
        # reuse the last resolved person in this chat without requiring "their".
        if self._is_attribute_only_person_followup_query(normalized_query, state):
            person_args: Dict[str, Any] = {}
            if isinstance(state.get("last_person_user_id"), str) and state.get("last_person_user_id"):
                person_args["user_id"] = state["last_person_user_id"]
            elif isinstance(state.get("last_person_name"), str) and state.get("last_person_name"):
                person_args["name"] = state["last_person_name"]

            if person_args:
                tool_name = "search_personnel"
                arguments = person_args
                understood_query = query
                confidence = 1.0
                result = await self.tool_handler.execute(tool_name, arguments, context)

                if self.use_llm and tool_name not in self._deterministic_format_tools:
                    response_text = await llm_format_response(query, tool_name, result, context_messages)
                else:
                    response_text = fallback_format_response(query, tool_name, arguments, result)

                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                if isinstance(result, dict) and result.get("success", False):
                    self._update_state(session_key, tool_name, arguments, result)

                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=tool_name,
                    arguments=arguments,
                    result=result,
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                return {
                    "success": bool(result.get("success")) if isinstance(result, dict) else True,
                    "query": query,
                    "understood_as": understood_query,
                    "response": response_text,
                    "routed_to": tool_name,
                    "arguments": arguments,
                    "confidence": confidence,
                    "data": result,
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

        # Follow-up unit details like "details on IT core (Special Wing)"
        # should resolve against units mentioned in previous hierarchy/list outputs.
        if self._is_unit_detail_query(normalized_query):
            resolved_unit = self._resolve_followup_unit_name(normalized_query, state)
            if resolved_unit:
                tool_name = "search_unit"
                arguments = {"name": resolved_unit}
                understood_query = query
                confidence = 1.0
                result = await self.tool_handler.execute(tool_name, arguments, context)

                if self.use_llm and tool_name not in self._deterministic_format_tools:
                    response_text = await llm_format_response(query, tool_name, result, context_messages)
                else:
                    response_text = fallback_format_response(query, tool_name, arguments, result)

                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                if isinstance(result, dict) and result.get("success", False):
                    self._update_state(session_key, tool_name, arguments, result)

                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=tool_name,
                    arguments=arguments,
                    result=result,
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                return {
                    "success": bool(result.get("success")) if isinstance(result, dict) else True,
                    "query": query,
                    "understood_as": understood_query,
                    "response": response_text,
                    "routed_to": tool_name,
                    "arguments": arguments,
                    "confidence": confidence,
                    "data": result,
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

        # Follow-up unit personnel like "list the personnel there"
        # should use the previously referenced unit context.
        if self._is_followup_unit_personnel_query(normalized_query):
            resolved_unit = self._resolve_followup_unit_name(normalized_query, state)
            if not resolved_unit:
                # Fallback to last explicit unit argument when available.
                last_args = state.get("last_arguments") or {}
                if isinstance(last_args, dict):
                    lu = last_args.get("unit_name")
                    if isinstance(lu, str) and lu.strip():
                        resolved_unit = lu.strip()
                    if not resolved_unit:
                        ln = last_args.get("name")
                        if isinstance(ln, str) and ln.strip():
                            resolved_unit = ln.strip()
            if resolved_unit:
                tool_name = "query_personnel_by_unit"
                arguments = {"unit_name": resolved_unit}
                understood_query = query
                confidence = 1.0
                result = await self.tool_handler.execute(tool_name, arguments, context)

                if self.use_llm and tool_name not in self._deterministic_format_tools:
                    response_text = await llm_format_response(query, tool_name, result, context_messages)
                else:
                    response_text = fallback_format_response(query, tool_name, arguments, result)

                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response_text})
                if isinstance(result, dict) and result.get("success", False):
                    self._update_state(session_key, tool_name, arguments, result)

                output_payload = build_output_payload(
                    query=query,
                    response_text=response_text,
                    routed_to=tool_name,
                    arguments=arguments,
                    result=result,
                    requested_format=effective_output_format,
                    allow_download=allow_download,
                )
                return {
                    "success": bool(result.get("success")) if isinstance(result, dict) else True,
                    "query": query,
                    "understood_as": understood_query,
                    "response": response_text,
                    "routed_to": tool_name,
                    "arguments": arguments,
                    "confidence": confidence,
                    "data": result,
                    "llm_enabled": self.use_llm,
                    "session_id": session_key,
                    "history_size": len(history),
                    "output": output_payload,
                }

        clarification_needed = needs_clarification(normalized_query)

        if self.use_llm:
            tool_name, arguments, understood_query, confidence, route_source = await llm_route_query(normalized_query, context_messages)
        else:
            tool_name, arguments = fallback_route_query(normalized_query)[:2]
            understood_query, confidence = normalized_query, 0.5
            route_source = "heuristic"
        force_deterministic_response = False

        arguments = self._inject_state_hints(session_key, normalized_query, arguments)
        tool_name, arguments = repair_route(
            query=normalized_query,
            tool_name=tool_name,
            arguments=arguments,
            last_user_query=last_user_query,
            last_assistant_response=last_assistant_response,
        )
        tool_name, arguments = self._apply_ordinal_route_overrides(normalized_query, tool_name, arguments, state)
        precomputed_result: Optional[Dict[str, Any]] = None

        # Force DIG/IGP role-of-unit queries to unit-personnel resolution first
        # so they don't drift into district/rank fallbacks.
        forced_senior = self._extract_forced_senior_role_unit(normalized_query)
        if forced_senior:
            resolved = await self._recover_role_of_unit_with_personnel_lookup(
                query=normalized_query,
                context=context,
            )
            if resolved:
                tool_name, arguments, precomputed_result = resolved
                force_deterministic_response = True
            else:
                tool_name = "query_personnel_by_unit"
                arguments = {"unit_name": forced_senior[1], "page_size": 500}

        # Let LLM try first on ambiguous queries. If LLM is unavailable/failed and
        # we only have a heuristic fallback, preserve the clarification behavior.
        if clarification_needed and route_source != "llm":
            response_text = (
                "Could you clarify what you want to know? "
                "Please include at least one target like a person name/user ID, district, unit, or rank. "
                "Example: 'Circle Inspectors in Chittoor district'."
            )
            output_payload = build_output_payload(
                query=query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=effective_output_format,
                allow_download=allow_download,
            )
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            return {
                "success": True,
                "query": query,
                "understood_as": "clarification_needed",
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "confidence": 1.0,
                "data": {},
                "llm_enabled": self.use_llm,
                "route_source": route_source,
                "session_id": session_key,
                "history_size": len(history),
                "output": output_payload,
            }

        if tool_name == "__help__":
            response_text = _capability_help_response_text()
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response_text})
            output_payload = build_output_payload(
                query=query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=effective_output_format,
                allow_download=allow_download,
            )
            return {
                "success": True,
                "query": query,
                "understood_as": "capability_help",
                "response": response_text,
                "routed_to": None,
                "arguments": {},
                "confidence": confidence,
                "data": {},
                "llm_enabled": self.use_llm,
                "route_source": route_source,
                "session_id": session_key,
                "history_size": len(history),
                "output": output_payload,
            }

        result = precomputed_result if precomputed_result is not None else await self.tool_handler.execute(tool_name, arguments, context)

        if not result.get("success", False):
            retry_tool, retry_args = repair_route(
                query=normalized_query,
                tool_name=tool_name,
                arguments=arguments,
                last_user_query=last_user_query,
                last_assistant_response=last_assistant_response,
            )
            if retry_tool != tool_name or retry_args != arguments:
                retry_result = await self.tool_handler.execute(retry_tool, retry_args, context)
                if retry_result.get("success", False):
                    tool_name, arguments, result = retry_tool, retry_args, retry_result
        if not result.get("success", False):
            tool_name, arguments, result = await self._recover_failed_result(
                query=normalized_query,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                state=state,
                context=context,
            )
        if tool_name == "get_unit_command_history":
            recovered = await self._recover_missing_command_with_personnel_lookup(
                query=normalized_query,
                arguments=arguments,
                result=result,
                context=context,
            )
            if recovered:
                tool_name, arguments, result = recovered
                force_deterministic_response = True
        if not force_deterministic_response:
            role_unit_recovered = await self._recover_role_of_unit_with_personnel_lookup(
                query=normalized_query,
                context=context,
            )
            if role_unit_recovered:
                tool_name, arguments, result = role_unit_recovered
                force_deterministic_response = True

        if (
            tool_name == "search_personnel"
            and result.get("success", False)
            and isinstance(result.get("data"), list)
            and len(result.get("data", [])) == 0
        ):
            candidate_name = (arguments or {}).get("name")
            if candidate_name:
                district_probe = await self.tool_handler.execute("list_districts", {"name": candidate_name}, context)
                districts = district_probe.get("data", []) if isinstance(district_probe, dict) else []
                if district_probe.get("success", False) and isinstance(districts, list) and len(districts) > 0:
                    tool_name = "list_units_in_district"
                    arguments = {"district_name": districts[0].get("district_name", candidate_name)}
                    result = await self.tool_handler.execute(tool_name, arguments, context)

        if is_followup_district_query(normalized_query):
            data_payload = result.get("data", []) if isinstance(result, dict) else []
            district_response = format_followup_district_response(data_payload)
            response_text = district_response or fallback_format_response(query, tool_name, arguments, result)
        elif tool_name in self._deterministic_format_tools or force_deterministic_response:
            response_text = fallback_format_response(query, tool_name, arguments, result)
        elif self.use_llm:
            response_text = await llm_format_response(query, tool_name, result, context_messages)
        else:
            response_text = fallback_format_response(query, tool_name, arguments, result)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response_text})
        if isinstance(result, dict) and result.get("success", False):
            self._update_state(session_key, tool_name, arguments, result)

        output_payload = build_output_payload(
            query=query,
            response_text=response_text,
            routed_to=tool_name,
            arguments=arguments,
            result=result,
            requested_format=effective_output_format,
            allow_download=allow_download,
        )
        return {
            "success": bool(result.get("success")) if isinstance(result, dict) else True,
            "query": query,
            "understood_as": understood_query,
            "response": response_text,
            "routed_to": tool_name,
            "arguments": arguments,
            "confidence": confidence,
            "data": result,
            "llm_enabled": self.use_llm,
            "route_source": route_source,
            "session_id": session_key,
            "history_size": len(history),
            "output": output_payload,
        }


_intelligent_handler: Optional[IntelligentQueryHandler] = None


def get_intelligent_handler() -> IntelligentQueryHandler:
    global _intelligent_handler
    if _intelligent_handler is None:
        _intelligent_handler = IntelligentQueryHandler()
    return _intelligent_handler
