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

_MAX_TURNS = 8

_SYSTEM_PROMPT = """\
You are a read-only MongoDB query assistant for a Police Personnel management system.
Your task: answer the user's intent by querying the database step-by-step.

## Available operations
Respond with exactly one JSON object per turn. No prose before or after the JSON.

### Describe collections — learn field names before querying
{"action": "describe_collections"}

### Find documents
{"action": "find", "collection": "<name>", "filter": {<MongoDB filter>}, "projection": {<fields>}, "limit": <int 1-500>}

### Aggregate (use for joins, grouping, counting, sorting across collections)
{"action": "aggregate", "collection": "<name>", "pipeline": [<stages>], "description": "<what this does>"}

### Done — return the final answer to the user
{"action": "done", "answer": "<natural language answer>"}

## Collection relationships (foreign keys)
Use these to write correct $lookup stages when the user's query spans multiple collections.

personnel_master:
  - rankId        → rank_master._id        (rank name, shortCode)
  - designationId → designation_master._id (designation/role title)
  - departmentId  → department_master._id  (department)

assignment_master:
  - userId        → personnel_master._id   (the officer)
  - unitId        → unit_master._id        (unit/station posted at)
  - designationId → designation_master._id (role within posting)

unit_master:
  - districtId    → district_master._id    (district the unit belongs to)
  - parentUnitId  → unit_master._id        (parent unit, self-referential)
  - unitTypeId    → unit_type_master._id   (type/category of unit)

unit_villages_master:
  - unitId        → unit_master._id        (unit responsible for village)
  - mandalId      → mandal_master._id      (sub-district)

mandal_master:
  - districtId    → district_master._id    (district)

## $lookup join patterns

### Pattern 1 — enrich personnel with rank name
{"action": "aggregate", "collection": "personnel_master", "pipeline": [
  {"$match": {"isDelete": false}},
  {"$lookup": {"from": "rank_master", "localField": "rankId", "foreignField": "_id", "as": "rank"}},
  {"$unwind": {"path": "$rank", "preserveNullAndEmptyArrays": true}},
  {"$project": {"name": 1, "badgeNo": 1, "rankName": "$rank.name", "shortCode": "$rank.shortCode"}},
  {"$limit": 50}
]}

### Pattern 2 — find officers posted at a named unit (join through assignment)
{"action": "aggregate", "collection": "unit_master", "pipeline": [
  {"$match": {"name": {"$regex": "Guntur", "$options": "i"}, "isDelete": false}},
  {"$lookup": {
    "from": "assignment_master",
    "let": {"uid": "$_id"},
    "pipeline": [
      {"$match": {"$expr": {"$eq": ["$unitId", "$$uid"]}, "isDelete": false, "isActive": true}},
      {"$lookup": {"from": "personnel_master", "localField": "userId", "foreignField": "_id", "as": "person"}},
      {"$unwind": "$person"},
      {"$lookup": {"from": "rank_master", "localField": "person.rankId", "foreignField": "_id", "as": "rank"}},
      {"$unwind": {"path": "$rank", "preserveNullAndEmptyArrays": true}},
      {"$project": {"_id": 0, "name": "$person.name", "badgeNo": "$person.badgeNo",
                    "rankName": "$rank.name", "designationId": 1}}
    ],
    "as": "officers"
  }},
  {"$limit": 20}
]}

### Pattern 3 — count personnel by rank across a district
{"action": "aggregate", "collection": "personnel_master", "pipeline": [
  {"$match": {"isDelete": false}},
  {"$lookup": {"from": "rank_master", "localField": "rankId", "foreignField": "_id", "as": "rank"}},
  {"$unwind": "$rank"},
  {"$group": {"_id": "$rank.name", "count": {"$sum": 1}}},
  {"$sort": {"count": -1}}
]}

## Rules
1. Start with describe_collections only if you are truly unsure of field names. For common lookups (rank, assignment, unit, district) use the relationship map above directly.
2. Always include "isDelete": false in filters where applicable.
3. Do NOT use $where, $function, $accumulator, $out, or $merge — they are blocked.
4. Scope filters (districtId, unitId) are applied automatically — do not add them yourself.
5. For cross-collection queries always use $lookup inside aggregate rather than multiple sequential find calls.
6. You have at most {max_turns} turns total. Use "done" before running out.
7. If the data cannot be found or the question is unanswerable, say so honestly in "answer".
8. Output ONLY the JSON object. Nothing else.
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
        try:
            return await self._run(intent, context, conversation_context)
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
