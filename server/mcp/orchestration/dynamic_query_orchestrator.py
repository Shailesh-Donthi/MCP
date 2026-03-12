"""Dynamic Query Orchestrator.

Runs a multi-step LLM ↔ DB loop that lets the LLM describe collections,
construct find/aggregate queries, and produce a final natural-language answer.
All queries are read-only and scope-enforced via DynamicQueryExecutor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from mcp.core.cache import query_cache
from mcp.core.schema_scanner import generate_schema_response, generate_system_prompt, get_schema_info
from mcp.router.llm_client import call_openai_api
from mcp.schemas.context_schema import UserContext
from mcp.tools.dynamic_query_tool import DynamicQueryExecutor, QueryValidationError

logger = logging.getLogger(__name__)

_MAX_TURNS = 12


class DynamicQueryOrchestrator:
    """Runs the multi-step LLM ↔ DB loop for ad-hoc read-only queries."""

    def __init__(self, executor: DynamicQueryExecutor):
        self.executor = executor

    async def run(
        self,
        intent: str,
        context: UserContext,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        # Check cache first
        cached = await query_cache.get(intent)
        if cached is not None:
            logger.info("DynamicQuery cache HIT for: %.80s", intent)
            cached["cache_hit"] = True
            return cached

        try:
            result = await self._run(intent, context, conversation_context)
            # Cache successful results only
            if result.get("success"):
                await query_cache.set(intent, result)
            return result
        except Exception as exc:
            logger.exception("DynamicQueryOrchestrator unhandled error: %s", exc)
            return {
                "success": False,
                "data": {},
                "response": "I encountered an internal error while processing your query. Please try rephrasing.",
                "turns": 0,
                "route_source": "dynamic_query",
            }

    async def _run(
        self,
        intent: str,
        context: UserContext,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        schema = get_schema_info()
        system_prompt = generate_system_prompt(schema).replace("{max_turns}", str(_MAX_TURNS))

        messages: List[Dict[str, str]] = []
        if conversation_context:
            for msg in conversation_context[-4:]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    messages.append({"role": str(msg.get("role") or "user"), "content": msg["content"]})
        messages.append({"role": "user", "content": f"Intent: {intent}"})

        for turn in range(_MAX_TURNS):
            raw = await call_openai_api(messages, system_prompt, max_tokens=2048)
            if not raw:
                logger.warning("DynamicQuery turn %d: no LLM response", turn)
                break

            action_obj = _parse_action(raw)
            if action_obj is None:
                logger.warning("DynamicQuery turn %d: unparseable response: %.200s", turn, raw)
                break

            action = str(action_obj.get("action") or "")

            if action == "done":
                answer = str(action_obj.get("answer") or "").strip()
                if not answer:
                    answer = "I wasn't able to find that information in the database."
                return {
                    "success": True,
                    "data": {"answer": answer},
                    "response": answer,
                    "turns": turn + 1,
                    "route_source": "dynamic_query",
                }

            # Execute the operation and feed results back as the next user message
            logger.info("DynamicQuery turn %d action: %s", turn, json.dumps(action_obj, default=str)[:500])
            op_result = await self._dispatch(action_obj, context)
            logger.info("DynamicQuery turn %d result count: %s", turn, op_result.get("count", "n/a"))
            result_json = json.dumps(op_result, default=str)
            if len(result_json) > 8000:
                result_json = result_json[:8000] + "...[truncated]"

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Operation result:\n{result_json}",
            })

        return {
            "success": False,
            "data": {},
            "response": "I'm sorry, I was not able to find the information for that query. Please try rephrasing.",
            "turns": _MAX_TURNS,
            "route_source": "dynamic_query",
        }

    async def _dispatch(
        self, action_obj: Dict[str, Any], context: UserContext
    ) -> Dict[str, Any]:
        action = str(action_obj.get("action") or "")
        try:
            if action == "describe_collections":
                # Return pre-scanned schema immediately — no DB roundtrip needed.
                return generate_schema_response(get_schema_info())

            if action == "find":
                return await self.executor.find(
                    collection=str(action_obj.get("collection") or ""),
                    filter_doc=action_obj.get("filter") or {},
                    projection=action_obj.get("projection") or None,
                    limit=int(action_obj.get("limit") or 20),
                    context=context,
                )

            if action == "aggregate":
                return await self.executor.aggregate(
                    collection=str(action_obj.get("collection") or ""),
                    pipeline=action_obj.get("pipeline") or [],
                    context=context,
                )

            return {"error": f"Unknown action '{action}'. Use describe_collections, find, aggregate, or done."}

        except QueryValidationError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.exception("DynamicQuery dispatch error for action '%s': %s", action, exc)
            return {"error": "Query execution failed. Try a different approach."}


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from the LLM response.

    Handles cases where the model prepends prose before the JSON.
    Tries progressively shorter substrings to find a parseable object.
    """
    # Find all '{' positions and try to parse from each one
    for start in [m.start() for m in re.finditer(r"\{", raw)]:
        # Find the matching closing brace
        end = raw.rfind("}")
        while end > start:
            candidate = raw[start:end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            end = raw.rfind("}", start, end)
    return None
