"""LLM-powered query routing and response orchestration."""

import json
import logging
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple

from mcp.handlers.tool_handler import get_tool_handler
from mcp.utils.output_layer import build_output_payload
from mcp.router import (
    ROUTER_SYSTEM_PROMPT,
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
)

logger = logging.getLogger(__name__)


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
    if not response:
        logger.warning("LLM routing failed, using fallback")
        tool, args, understood_query, confidence = fallback_route_query(query)
        return tool, args, understood_query, confidence, "heuristic_fallback"

    try:
        json_match = __import__("re").search(r"\{[\s\S]*\}", response)
        if json_match:
            result = json.loads(json_match.group())
            return (
                result.get("tool", "search_personnel"),
                result.get("arguments", {}),
                result.get("understood_query", query),
                result.get("confidence", 0.5),
                "llm",
            )
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse LLM response: {exc}")

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
        "I can help with police personnel and unit reporting queries. Try asking:\n\n"
        "- 'What is the mobile number of A Ashok Kumar?'\n"
        "- 'List all SIs in Chittoor district'\n"
        "- 'How many personnel are in Guntur district?'\n"
        "- 'What is the unit hierarchy for Chittoor district?'\n"
        "- 'List units in Guntur district'\n"
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
            "list_units_in_district",
            "list_districts",
            "get_personnel_distribution",
            "query_personnel_by_rank",
            "get_unit_command_history",
            "get_unit_hierarchy",
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

    def _inject_state_hints(self, session_id: str, query: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        state = self._get_state(session_id)
        q = (query or "").lower()
        new_args = dict(arguments or {})
        if not new_args.get("user_id"):
            idx = extract_ordinal_index(q) or extract_list_reference_index(q)
            ids = state.get("last_list_user_ids") or []
            if isinstance(idx, int) and idx >= 1 and idx <= len(ids):
                new_args["user_id"] = ids[idx - 1]
        if not new_args.get("rank_name") and __import__("re").search(r"\b(their|them|they|those|these)\b", q):
            if state.get("last_rank"):
                new_args["rank_name"] = state["last_rank"]
        if not new_args.get("district_name") and __import__("re").search(r"\b(their|them|they|those|these)\b", q):
            if state.get("last_place"):
                new_args["district_name"] = state["last_place"]
        return new_args

    def _is_followup_person_detail_query(self, query: str) -> bool:
        q = (query or "").lower()
        if not q:
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
        district_intent = bool(__import__("re").search(r"\b(district|belong|belongs|belonging)\b", q))
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
        has_action = bool(__import__("re").search(r"\b(list|show|get|give)\b", q))
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
        history = self._get_history(session_key)
        context_messages = list(history)
        state = self._get_state(session_key)

        last_user_query = next((m.get("content") for m in reversed(context_messages) if m.get("role") == "user"), None)
        last_assistant_response = next((m.get("content") for m in reversed(context_messages) if m.get("role") == "assistant"), None)

        page_delta = self._pagination_delta(query)
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
                    requested_format=output_format,
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
                    requested_format=output_format,
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

            if is_followup_district_query(query):
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
                requested_format=output_format,
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

        if self._is_capability_help_query(query):
            response_text = _capability_help_response_text()
            output_payload = build_output_payload(
                query=query,
                response_text=response_text,
                routed_to=None,
                arguments={},
                result={},
                requested_format=output_format,
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
        if self._is_followup_person_detail_query(query):
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
                    requested_format=output_format,
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
        if self._is_attribute_only_person_followup_query(query, state):
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
                    requested_format=output_format,
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
        if self._is_unit_detail_query(query):
            resolved_unit = self._resolve_followup_unit_name(query, state)
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
                    requested_format=output_format,
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
        if self._is_followup_unit_personnel_query(query):
            resolved_unit = self._resolve_followup_unit_name(query, state)
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
                    requested_format=output_format,
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

        clarification_needed = needs_clarification(query)

        if self.use_llm:
            tool_name, arguments, understood_query, confidence, route_source = await llm_route_query(query, context_messages)
        else:
            tool_name, arguments = fallback_route_query(query)[:2]
            understood_query, confidence = query, 0.5
            route_source = "heuristic"

        arguments = self._inject_state_hints(session_key, query, arguments)
        tool_name, arguments = repair_route(
            query=query,
            tool_name=tool_name,
            arguments=arguments,
            last_user_query=last_user_query,
            last_assistant_response=last_assistant_response,
        )

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
                requested_format=output_format,
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
                requested_format=output_format,
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

        result = await self.tool_handler.execute(tool_name, arguments, context)

        if not result.get("success", False):
            retry_tool, retry_args = repair_route(
                query=query,
                tool_name=tool_name,
                arguments=arguments,
                last_user_query=last_user_query,
                last_assistant_response=last_assistant_response,
            )
            if retry_tool != tool_name or retry_args != arguments:
                retry_result = await self.tool_handler.execute(retry_tool, retry_args, context)
                if retry_result.get("success", False):
                    tool_name, arguments, result = retry_tool, retry_args, retry_result

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

        if is_followup_district_query(query):
            data_payload = result.get("data", []) if isinstance(result, dict) else []
            district_response = format_followup_district_response(data_payload)
            response_text = district_response or fallback_format_response(query, tool_name, arguments, result)
        elif tool_name in self._deterministic_format_tools:
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
            requested_format=output_format,
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
