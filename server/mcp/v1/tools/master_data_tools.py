"""
Linked Master Data Tools

Provides discovery and linked querying across master collections.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bson import ObjectId

from mcp.schemas.context_schema import UserContext
from mcp.tools.base_tool import BaseTool
from mcp.utils.formatters import stringify_object_ids


REQUESTED_MASTER_COLLECTIONS: List[str] = [
    "approval_flow_master",
    "department_master",
    "district_master",
    "error_master",
    "jobs_master",
    "log_master",
    "mandal_master",
    "modules_master",
    "notification_master",
    "permissions_master",
    "permissions_mapping_master",
    "personnel_master",
    "prompt_master",
    "rank_master",
    "roles_master",
    "unit_master",
    "unit_type_master",
    "unit_villages_master",
    "user_role_permissions_master",
    "value_sets_master",
]


COLLECTION_ALIASES: Dict[str, List[str]] = {
    "approval_flow_master": ["approval_flow_master"],
    "department_master": ["department_master"],
    "district_master": ["district_master"],
    "error_master": ["error_master"],
    "jobs_master": ["jobs_master"],
    "log_master": ["log_master"],
    "mandal_master": ["mandal_master"],
    "modules_master": ["modules_master"],
    "notification_master": ["notification_master", "notifications", "notification"],
    "permissions_master": ["permissions_master"],
    "permissions_mapping_master": ["permissions_mapping_master"],
    "personnel_master": ["personnel_master"],
    "prompt_master": ["prompt_master"],
    "rank_master": ["rank_master"],
    "roles_master": ["roles_master", "role_master", "usecase_roles", "rank_roles"],
    "unit_master": ["unit_master"],
    "unit_type_master": ["unit_type_master"],
    "unit_villages_master": ["unit_villages_master"],
    "user_role_permissions_master": ["user_role_permissions_master"],
    "value_sets_master": ["value_sets_master"],
}


QUERY_HINT_TO_COLLECTION: List[Tuple[str, str]] = [
    (r"\bapproval\s*flow|approval\s*chain\b", "approval_flow_master"),
    (r"\bdepartment\b", "department_master"),
    (r"\bdistrict\b", "district_master"),
    (r"\berror(s)?\b", "error_master"),
    (r"\bjobs?\b", "jobs_master"),
    (r"\blogs?\b", "log_master"),
    (r"\bmandal\b", "mandal_master"),
    (r"\bnotification(s)?\b", "notification_master"),
    (r"\bmodules?\b", "modules_master"),
    (r"\bpermissions?\b", "permissions_master"),
    (r"\bpermission\s*mapping\b", "permissions_mapping_master"),
    (r"\bpersonnel|officers?|staff\b", "personnel_master"),
    (r"\bprompts?\b", "prompt_master"),
    (r"\branks?\b", "rank_master"),
    (r"\broles?\b", "roles_master"),
    (r"\bunits?|stations?\b", "unit_master"),
    (r"\bunit\s*types?\b", "unit_type_master"),
    (r"\bvillages?|village\s*mapping\b", "unit_villages_master"),
    (r"\buser\s*role\s*permissions?\b", "user_role_permissions_master"),
    (r"\bvalue\s*sets?\b", "value_sets_master"),
]


@dataclass(frozen=True)
class RelationDef:
    source: str
    field: str
    target: str
    target_field: str = "_id"


RELATION_DEFS: List[RelationDef] = [
    RelationDef("personnel_master", "rankId", "rank_master"),
    RelationDef("personnel_master", "departmentId", "department_master"),
    RelationDef("unit_master", "districtId", "district_master"),
    RelationDef("unit_master", "departmentId", "department_master"),
    RelationDef("unit_master", "unitTypeId", "unit_type_master"),
    RelationDef("mandal_master", "districtId", "district_master"),
    RelationDef("unit_villages_master", "unitId", "unit_master"),
    RelationDef("unit_villages_master", "mandalId", "mandal_master"),
    RelationDef("approval_flow_master", "districtId", "district_master"),
    RelationDef("approval_flow_master", "finalApprovalUnitId", "unit_master"),
    RelationDef("approval_flow_master", "moduleId", "modules_master"),
    RelationDef("error_master", "moduleId", "modules_master"),
    RelationDef("log_master", "moduleId", "modules_master"),
    RelationDef("notification_master", "moduleId", "modules_master"),
    RelationDef("prompt_master", "moduleId", "modules_master"),
]


def infer_master_collection_from_query(query: str) -> Optional[str]:
    q = (query or "").strip().lower()
    if not q:
        return None
    for pattern, collection in QUERY_HINT_TO_COLLECTION:
        if re.search(pattern, q, re.IGNORECASE):
            return collection
    return None


class QueryLinkedMasterDataTool(BaseTool):
    """
    Query master collections with relationship awareness.
    """

    name = "query_linked_master_data"
    description = (
        "Discover and query linked master data across approval, module, "
        "notification, prompt, personnel, unit, rank, district, and value-set collections."
    )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "discover or query",
                    "enum": ["discover", "query"],
                    "default": "query",
                },
                "collection": {
                    "type": "string",
                    "description": (
                        "Root collection name. Accepts canonical names "
                        "(for example: roles_master) and known aliases."
                    ),
                },
                "filters": {
                    "type": "object",
                    "description": "Optional Mongo-style equality filters on root collection fields.",
                },
                "search_text": {
                    "type": "string",
                    "description": "Optional broad text filter applied on common string fields.",
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Include linked documents using known foreign-key relations.",
                    "default": True,
                },
                "include_reverse": {
                    "type": "boolean",
                    "description": "Include reverse-linked documents that reference the root records.",
                    "default": True,
                },
                "related_collections": {
                    "type": "array",
                    "description": "Optional allow-list of related canonical collections to expand.",
                    "items": {"type": "string"},
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort field on the root collection.",
                    "default": "createdAt",
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "default": "asc",
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "default": 50,
                },
                "include_integrity": {
                    "type": "boolean",
                    "description": "For discover mode: compute relation integrity summary.",
                    "default": True,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        mode = str(arguments.get("mode") or "query").strip().lower()
        if mode == "discover":
            return await self._discover_relations(arguments)
        return await self._query_with_links(arguments, context)

    async def _discover_relations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        include_integrity = bool(arguments.get("include_integrity", True))
        existing = set(await self.db.list_collection_names())

        collections_payload: List[Dict[str, Any]] = []
        for canonical in REQUESTED_MASTER_COLLECTIONS:
            resolved_actual = self._resolve_collection_name(canonical, existing)[0]
            collections_payload.append(
                {
                    "canonical": canonical,
                    "resolved_collection": resolved_actual,
                    "exists": bool(resolved_actual),
                    "aliases_checked": COLLECTION_ALIASES.get(canonical, [canonical]),
                }
            )

        relation_payload: List[Dict[str, Any]] = []
        for rel in RELATION_DEFS:
            src_actual = self._resolve_collection_name(rel.source, existing)[0]
            dst_actual = self._resolve_collection_name(rel.target, existing)[0]
            entry: Dict[str, Any] = {
                "source": rel.source,
                "source_collection": src_actual,
                "source_field": rel.field,
                "target": rel.target,
                "target_collection": dst_actual,
                "target_field": rel.target_field,
                "active": bool(src_actual and dst_actual),
            }
            if include_integrity and src_actual and dst_actual:
                entry["integrity"] = await self._relation_integrity(
                    source_collection=src_actual,
                    source_field=rel.field,
                    target_collection=dst_actual,
                    target_field=rel.target_field,
                )
            relation_payload.append(entry)

        missing = [c["canonical"] for c in collections_payload if not c["exists"]]
        metadata = {
            "requested_collection_count": len(REQUESTED_MASTER_COLLECTIONS),
            "existing_requested_collections": len(REQUESTED_MASTER_COLLECTIONS) - len(missing),
            "missing_requested_collections": missing,
            "active_relation_count": len([r for r in relation_payload if r.get("active")]),
        }

        return self.format_success_response(
            query_type="linked_master_data_discovery",
            data={
                "requested_collections": collections_payload,
                "relations": relation_payload,
            },
            metadata=metadata,
        )

    async def _query_with_links(
        self,
        arguments: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        existing = set(await self.db.list_collection_names())

        raw_collection = str(arguments.get("collection") or "").strip()
        if not raw_collection:
            inferred = infer_master_collection_from_query(str(arguments.get("search_text") or ""))
            raw_collection = inferred or ""

        if not raw_collection:
            return self.format_error_response(
                "MISSING_PARAMETER",
                "Collection is required for query mode.",
                {
                    "hint": (
                        "Set collection to one of: "
                        + ", ".join(REQUESTED_MASTER_COLLECTIONS)
                    )
                },
            )

        resolved_collection, canonical_collection, candidates = self._resolve_collection_name(
            raw_collection,
            existing,
        )
        if not resolved_collection:
            suggestions = get_close_matches(
                str(raw_collection).lower(),
                sorted(existing),
                n=5,
                cutoff=0.6,
            )
            return self.format_error_response(
                "NOT_FOUND",
                f"Collection not found in database: {raw_collection}",
                {
                    "canonical_collection": canonical_collection,
                    "aliases_checked": candidates,
                    "available_similar_collections": suggestions,
                },
            )

        filters = arguments.get("filters") or {}
        if not isinstance(filters, dict):
            return self.format_error_response(
                "VALIDATION_ERROR",
                "filters must be an object/dictionary.",
            )

        search_text = str(arguments.get("search_text") or "").strip()
        include_related = bool(arguments.get("include_related", True))
        include_reverse = bool(arguments.get("include_reverse", True))
        related_collections = arguments.get("related_collections") or []
        if not isinstance(related_collections, list):
            related_collections = []
        related_allow_set = {
            str(item).strip().lower()
            for item in related_collections
            if isinstance(item, str) and item.strip()
        }

        sort_by = str(arguments.get("sort_by") or "createdAt").strip()
        sort_order = str(arguments.get("sort_order") or "asc").strip().lower()
        sort_dir = -1 if sort_order == "desc" else 1
        page, page_size, skip = self.get_pagination_params(arguments)

        base_query = self._build_query_from_filters(filters)
        base_query = await self._apply_scope(base_query, canonical_collection, context)

        if search_text:
            text_fields = await self._guess_text_fields(resolved_collection)
            if text_fields:
                search_or = [
                    {field: {"$regex": re.escape(search_text), "$options": "i"}}
                    for field in text_fields
                ]
                if "$and" in base_query:
                    base_query["$and"].append({"$or": search_or})
                elif base_query:
                    base_query = {"$and": [base_query, {"$or": search_or}]}
                else:
                    base_query = {"$or": search_or}

        total = await self.db[resolved_collection].count_documents(base_query)
        docs = await self.db[resolved_collection].find(base_query).sort(sort_by, sort_dir).skip(skip).limit(page_size).to_list(length=page_size)
        docs = [self._compact_document(stringify_object_ids(doc)) for doc in docs]

        link_metadata: Dict[str, Any] = {
            "forward_links": [],
            "reverse_links": [],
            "collection_resolution": {
                "input": raw_collection,
                "canonical": canonical_collection,
                "resolved_collection": resolved_collection,
            },
        }

        if include_related and docs:
            await self._attach_forward_links(
                docs=docs,
                root_canonical=canonical_collection,
                existing_collections=existing,
                allow_set=related_allow_set,
                link_metadata=link_metadata,
            )

        if include_reverse and docs:
            await self._attach_reverse_links(
                docs=docs,
                root_canonical=canonical_collection,
                existing_collections=existing,
                allow_set=related_allow_set,
                link_metadata=link_metadata,
            )

        return self.format_success_response(
            query_type="linked_master_data_query",
            data=docs,
            total=total,
            page=page,
            page_size=page_size,
            metadata=link_metadata,
        )

    async def _relation_integrity(
        self,
        *,
        source_collection: str,
        source_field: str,
        target_collection: str,
        target_field: str,
    ) -> Dict[str, Any]:
        pipeline = [
            {"$match": {source_field: {"$exists": True, "$ne": None}}},
            {
                "$lookup": {
                    "from": target_collection,
                    "localField": source_field,
                    "foreignField": target_field,
                    "as": "__rel",
                }
            },
            {
                "$project": {
                    "ok": {"$gt": [{"$size": "$__rel"}, 0]},
                }
            },
            {
                "$group": {
                    "_id": "$ok",
                    "count": {"$sum": 1},
                }
            },
        ]
        rows = await self.db[source_collection].aggregate(pipeline).to_list(length=None)
        resolved = 0
        unresolved = 0
        for row in rows:
            if row.get("_id") is True:
                resolved = int(row.get("count") or 0)
            else:
                unresolved = int(row.get("count") or 0)
        return {
            "referencing_rows": resolved + unresolved,
            "resolved_rows": resolved,
            "unresolved_rows": unresolved,
        }

    def _build_query_from_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        query: Dict[str, Any] = {}
        for key, raw_value in filters.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                value = raw_value.strip()
                if not value:
                    continue
                if value.startswith("~"):
                    query[key] = {"$regex": re.escape(value[1:]), "$options": "i"}
                    continue
                if self._looks_like_object_id_field(key) and ObjectId.is_valid(value):
                    query[key] = ObjectId(value)
                    continue
                query[key] = value
                continue
            if isinstance(raw_value, list):
                values = [
                    self._coerce_object_id_if_needed(key, item)
                    for item in raw_value
                    if item is not None
                ]
                query[key] = {"$in": values}
                continue
            query[key] = self._coerce_object_id_if_needed(key, raw_value)
        return query

    async def _apply_scope(
        self,
        base_query: Dict[str, Any],
        canonical_collection: str,
        context: UserContext,
    ) -> Dict[str, Any]:
        if context.has_state_access:
            return base_query

        if canonical_collection == "personnel_master":
            return await self.apply_scope_filter(base_query, context, "personnel")
        if canonical_collection == "unit_master":
            return await self.apply_scope_filter(base_query, context, "unit")
        if canonical_collection == "unit_villages_master":
            return await self.apply_scope_filter(base_query, context, "unit_villages")

        query = dict(base_query)

        # Generic district scope for master collections carrying districtId.
        district_ids = context.get_accessible_district_ids()
        if district_ids:
            district_oids = [ObjectId(did) for did in district_ids if ObjectId.is_valid(did)]
            if district_oids:
                query["districtId"] = {"$in": district_oids}
                return query

        unit_ids = context.get_accessible_unit_ids()
        if unit_ids:
            unit_oids = [ObjectId(uid) for uid in unit_ids if ObjectId.is_valid(uid)]
            if unit_oids:
                query["unitId"] = {"$in": unit_oids}
                return query

        return {"_id": {"$exists": False}}

    async def _guess_text_fields(self, collection_name: str) -> List[str]:
        sample = await self.db[collection_name].find_one({})
        if not sample:
            return []
        preferred_order = [
            "name",
            "shortCode",
            "description",
            "flowName",
            "errorCode",
            "moduleName",
            "notificationType",
            "category",
            "key",
            "userId",
            "title",
        ]
        candidates = [field for field in preferred_order if isinstance(sample.get(field), str)]
        if candidates:
            return candidates[:6]
        fallback = [
            key
            for key, value in sample.items()
            if isinstance(value, str) and key not in {"_id", "createdBy", "updatedBy", "createdIp", "updatedIp"}
        ]
        return fallback[:6]

    async def _attach_forward_links(
        self,
        *,
        docs: List[Dict[str, Any]],
        root_canonical: str,
        existing_collections: Set[str],
        allow_set: Set[str],
        link_metadata: Dict[str, Any],
    ) -> None:
        root_relations = [rel for rel in RELATION_DEFS if rel.source == root_canonical]
        for rel in root_relations:
            target_actual, target_canonical, _ = self._resolve_collection_name(rel.target, existing_collections)
            if not target_actual or not target_canonical:
                continue
            if allow_set and target_canonical not in allow_set:
                continue

            join_values: Set[str] = set()
            for doc in docs:
                join_values.update(self._extract_join_values(doc.get(rel.field)))

            if not join_values:
                continue

            fetch_in = self._build_fetch_values(rel.target_field, join_values)
            if not fetch_in:
                continue

            linked_docs = await self.db[target_actual].find(
                {rel.target_field: {"$in": fetch_in}}
            ).limit(5000).to_list(length=5000)
            linked_docs = [self._compact_document(stringify_object_ids(item)) for item in linked_docs]

            by_key: Dict[str, Dict[str, Any]] = {}
            for item in linked_docs:
                key = str(item.get(rel.target_field))
                if key:
                    by_key[key] = item

            attached_count = 0
            for doc in docs:
                local = doc.get(rel.field)
                if isinstance(local, list):
                    refs = [by_key.get(str(value)) for value in local if by_key.get(str(value))]
                    if refs:
                        doc[f"{rel.field}_ref"] = refs[:10]
                        attached_count += 1
                else:
                    ref = by_key.get(str(local))
                    if ref:
                        doc[f"{rel.field}_ref"] = ref
                        attached_count += 1

            link_metadata["forward_links"].append(
                {
                    "source": root_canonical,
                    "field": rel.field,
                    "target": target_canonical,
                    "target_collection": target_actual,
                    "linked_document_count": len(linked_docs),
                    "rows_with_link_attached": attached_count,
                }
            )

    async def _attach_reverse_links(
        self,
        *,
        docs: List[Dict[str, Any]],
        root_canonical: str,
        existing_collections: Set[str],
        allow_set: Set[str],
        link_metadata: Dict[str, Any],
    ) -> None:
        reverse_relations = [rel for rel in RELATION_DEFS if rel.target == root_canonical]
        if not reverse_relations:
            return

        root_ids = [str(doc.get("_id")) for doc in docs if doc.get("_id")]
        if not root_ids:
            return

        for rel in reverse_relations:
            source_actual, source_canonical, _ = self._resolve_collection_name(rel.source, existing_collections)
            if not source_actual or not source_canonical:
                continue
            if allow_set and source_canonical not in allow_set:
                continue

            fetch_in = self._build_fetch_values(rel.field, set(root_ids))
            if not fetch_in:
                continue

            source_docs = await self.db[source_actual].find(
                {rel.field: {"$in": fetch_in}}
            ).limit(3000).to_list(length=3000)
            source_docs = [self._compact_document(stringify_object_ids(item)) for item in source_docs]

            attached = 0
            for root_doc in docs:
                rid = str(root_doc.get("_id"))
                if not rid:
                    continue
                matching: List[Dict[str, Any]] = []
                for src_doc in source_docs:
                    src_value = src_doc.get(rel.field)
                    src_values = src_value if isinstance(src_value, list) else [src_value]
                    if rid in {str(v) for v in src_values if v is not None}:
                        matching.append(src_doc)
                if matching:
                    root_doc.setdefault("_linked_from", {})
                    root_doc["_linked_from"][source_canonical] = matching[:10]
                    attached += 1

            link_metadata["reverse_links"].append(
                {
                    "source": source_canonical,
                    "field": rel.field,
                    "target": root_canonical,
                    "rows_with_reverse_link_attached": attached,
                    "linked_document_count": len(source_docs),
                }
            )

    def _resolve_collection_name(
        self,
        collection_name: str,
        existing_collections: Set[str],
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        raw = str(collection_name or "").strip().lower()
        if not raw:
            return None, None, []

        canonical = raw
        if raw in COLLECTION_ALIASES:
            canonical = raw
        else:
            for key, aliases in COLLECTION_ALIASES.items():
                if raw == key or raw in {alias.lower() for alias in aliases}:
                    canonical = key
                    break

        candidates = [raw]
        for alias in COLLECTION_ALIASES.get(canonical, []):
            if alias not in candidates:
                candidates.append(alias)
        if canonical not in candidates:
            candidates.insert(0, canonical)

        for candidate in candidates:
            if candidate in existing_collections:
                return candidate, canonical, candidates
        return None, canonical, candidates

    @staticmethod
    def _looks_like_object_id_field(field_name: str) -> bool:
        key = (field_name or "").lower()
        return key == "_id" or key.endswith("id")

    def _coerce_object_id_if_needed(self, field_name: str, value: Any) -> Any:
        if isinstance(value, str) and self._looks_like_object_id_field(field_name) and ObjectId.is_valid(value):
            return ObjectId(value)
        return value

    @staticmethod
    def _extract_join_values(value: Any) -> Set[str]:
        out: Set[str] = set()
        if isinstance(value, list):
            for item in value:
                if item is not None:
                    out.add(str(item))
            return out
        if value is not None:
            out.add(str(value))
        return out

    @staticmethod
    def _build_fetch_values(field_name: str, raw_values: Iterable[str]) -> List[Any]:
        values: List[Any] = []
        seen: Set[str] = set()
        for raw in raw_values:
            v = str(raw)
            if not v or v in seen:
                continue
            seen.add(v)
            values.append(v)
            if (field_name == "_id" or field_name.lower().endswith("id")) and ObjectId.is_valid(v):
                values.append(ObjectId(v))
        return values

    def _compact_document(self, value: Any, *, depth: int = 0) -> Any:
        if depth > 3:
            return None
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= 24:
                    compact["__truncated_keys"] = True
                    break
                if k in {"json", "template", "taskExample", "settingsJson"}:
                    compact[k] = "<omitted_large_field>"
                    continue
                compact[k] = self._compact_document(v, depth=depth + 1)
            return compact
        if isinstance(value, list):
            clipped = value[:12]
            out = [self._compact_document(v, depth=depth + 1) for v in clipped]
            if len(value) > 12:
                out.append({"__truncated_items": len(value) - 12})
            return out
        if isinstance(value, str):
            if len(value) > 320:
                return value[:320] + "...<truncated>"
            return value
        return value
