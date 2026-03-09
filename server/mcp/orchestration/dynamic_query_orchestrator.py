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
from mcp.router.llm_client import call_openai_api
from mcp.schemas.context_schema import UserContext
from mcp.tools.dynamic_query_tool import DynamicQueryExecutor, QueryValidationError

logger = logging.getLogger(__name__)

_MAX_TURNS = 12

# Static schema returned immediately when LLM calls describe_collections,
# avoiding a DB roundtrip and keeping the turn count low.
_STATIC_SCHEMA_RESPONSE: Dict[str, Any] = {
    "collections": {
        "personnel_master": {
            "fields": ["_id", "name", "badgeNo", "mobileNo", "rankId", "designationId", "departmentId", "isDelete", "isActive"],
            "relationships": {"rankId": "rank_master._id", "designationId": "designation_master._id", "departmentId": "department_master._id"},
        },
        "rank_master": {"fields": ["_id", "name", "shortCode", "order", "isDelete"]},
        "designation_master": {"fields": ["_id", "name", "isDelete"]},
        "department_master": {"fields": ["_id", "name", "isDelete"]},
        "assignment_master": {
            "fields": ["_id", "userId", "unitId", "designationId", "fromDate", "toDate", "isActive", "isDelete"],
            "relationships": {"userId": "personnel_master._id", "unitId": "unit_master._id", "designationId": "designation_master._id"},
        },
        "unit_master": {
            "fields": ["_id", "name", "shortCode", "districtId", "parentUnitId", "unitTypeId", "isDelete"],
            "relationships": {"districtId": "district_master._id", "parentUnitId": "unit_master._id", "unitTypeId": "unit_type_master._id"},
        },
        "unit_type_master": {"fields": ["_id", "name", "isDelete"]},
        "district_master": {"fields": ["_id", "name", "shortCode", "isDelete"]},
        "mandal_master": {
            "fields": ["_id", "name", "districtId", "isDelete"],
            "relationships": {"districtId": "district_master._id"},
        },
        "unit_villages_master": {
            "fields": ["_id", "unitId", "mandalId", "villageName", "isDelete"],
            "relationships": {"unitId": "unit_master._id", "mandalId": "mandal_master._id"},
        },
    }
}

_SYSTEM_PROMPT = """\
You are a read-only MongoDB query assistant for a Police Personnel management system.
Answer the user's intent by querying the database step-by-step.
Output ONLY a single JSON object per turn — no prose before or after.

## Actions
{"action":"find","collection":"<col>","filter":{...},"projection":{...},"limit":<1-500>}
{"action":"aggregate","collection":"<col>","pipeline":[...]}
{"action":"done","answer":"<natural language answer>"}

## Schema (fields)
personnel_master: _id,name,badgeNo,mobileNo,rankId,designationId,departmentId,isDelete,isActive
rank_master: _id,name,shortCode,order,isDelete  [shortCodes: Constable,HC,ASI,SI,Inspector,DSP,SP,DySP]
assignment_master: _id,userId,unitId,designationId,fromDate,toDate,isActive,isDelete
unit_master: _id,name,shortCode,districtId,parentUnitId,unitTypeId,isDelete
district_master: _id,name,shortCode,isDelete
mandal_master: _id,name,districtId,isDelete
unit_villages_master: _id,unitId,mandalId,villageName,isDelete
designation_master,department_master,unit_type_master: _id,name,isDelete

## Foreign keys
personnel_master.rankId→rank_master._id | personnel_master.designationId→designation_master._id
assignment_master.userId→personnel_master._id | assignment_master.unitId→unit_master._id
unit_master.districtId→district_master._id | unit_master.parentUnitId→unit_master._id
unit_villages_master.unitId→unit_master._id | unit_villages_master.mandalId→mandal_master._id
mandal_master.districtId→district_master._id

## Rules
1. Always add "isDelete":false to filters. ONLY add "isActive":true on assignment_master (for current postings). Do NOT filter by isActive on personnel_master — most personnel have isActive=false but are still valid records.
2. Use $lookup in aggregate for cross-collection joins — never multiple sequential finds.
3. For rank queries: $lookup rank_master on rankId, then filter by rank.shortCode or rank.name.
4. For district queries: use a SINGLE aggregate on assignment_master with $lookup unit_master (on unitId), $match unit's districtId, then $lookup personnel_master (on userId). This avoids multiple round-trips.
5. For missing mappings: $lookup unit_villages_master on unitId, $match villages size==0.
6. For transfers: filter assignment_master.fromDate >= date range.
7. Do NOT use $where,$function,$accumulator,$out,$merge — blocked.
8. Scope filters are applied automatically — do not add districtId/unitId scope yourself.
9. You have at most {max_turns} turns. Call "done" before running out.
10. If the query is unrelated to police personnel, call done immediately with a polite note.
11. If data is absent, say so clearly (e.g. "No vacancy data found in the system").
12. When the user asks to "list all" or "show all", return individual records with names/details — do NOT return counts or group-by summaries unless explicitly asked for counts. Use a $limit of 500. Do NOT use small limits like 10 or 50 for listing queries. For $count/$group queries, omit $limit entirely so all records are counted.
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
        system_prompt = _SYSTEM_PROMPT.replace("{max_turns}", str(_MAX_TURNS))

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
                # Return static schema immediately — no DB roundtrip needed.
                # This prevents the LLM from burning a turn on a DB call it doesn't need.
                return _STATIC_SCHEMA_RESPONSE

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
