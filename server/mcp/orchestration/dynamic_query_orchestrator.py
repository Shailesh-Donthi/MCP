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

from mcp.router.llm_client import call_openai_api
from mcp.schemas.context_schema import UserContext
from mcp.tools.dynamic_query_tool import DynamicQueryExecutor, QueryValidationError

logger = logging.getLogger(__name__)

_MAX_TURNS = 5

_SYSTEM_PROMPT = """\
You are a read-only MongoDB query assistant for a Police Personnel management system.
Your task: answer the user's intent by querying the database step-by-step.

## Available operations
Respond with exactly one JSON object per turn. No prose before or after the JSON.

### Describe collections — learn field names before querying
{"action": "describe_collections"}

### Find documents
{"action": "find", "collection": "<name>", "filter": {<MongoDB filter>}, "projection": {<fields or null>}, "limit": <int 1-100>}

### Aggregate
{"action": "aggregate", "collection": "<name>", "pipeline": [<stages>], "description": "<what this query does in plain English>"}

### Done — return the final answer to the user
{"action": "done", "answer": "<natural language answer>"}

## Rules
- Start with describe_collections if you are unsure of field names.
- Always include "isDelete": false in filters where applicable.
- Do NOT use $where, $function, $accumulator, $out, or $merge — they are blocked.
- Scope filters are applied automatically; do not add districtId/unitId filters yourself.
- You have at most {max_turns} turns total. Use "done" before running out.
- If the data cannot be found or the question is unanswerable, say so honestly in "answer".
- Output ONLY the JSON object. Nothing else.
"""


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
        system_prompt = _SYSTEM_PROMPT.format(max_turns=_MAX_TURNS)

        messages: List[Dict[str, str]] = []
        if conversation_context:
            messages.extend(conversation_context[-4:])
        messages.append({"role": "user", "content": f"Intent: {intent}"})

        for turn in range(_MAX_TURNS):
            raw = await call_openai_api(messages, system_prompt, max_tokens=1024)
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
            op_result = await self._dispatch(action_obj, context)
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
            "response": "I wasn't able to find the information in the database for that query.",
            "turns": _MAX_TURNS,
            "route_source": "dynamic_query",
        }

    async def _dispatch(
        self, action_obj: Dict[str, Any], context: UserContext
    ) -> Dict[str, Any]:
        action = str(action_obj.get("action") or "")
        try:
            if action == "describe_collections":
                return await self.executor.describe_collections()

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
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
