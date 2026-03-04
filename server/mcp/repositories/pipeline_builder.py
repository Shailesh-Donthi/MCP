"""Aggregation pipeline builder for V2 relationship-aware enrichment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from mcp.config import mcp_settings
from mcp.repositories.relationship_mapper import (
    RelationshipSpec,
    get_relationship,
    normalize_enrichments,
)


class PipelineBuilder:
    """Builds MongoDB aggregation pipelines with optional relationship joins."""

    MAX_ENRICHMENT_DEPTH = int(getattr(mcp_settings, "MCP_MAX_ENRICHMENT_DEPTH", 4) or 4)

    def build_enriched_query(
        self,
        *,
        base_collection: str,
        filters: Optional[Dict[str, Any]] = None,
        enrichments: Optional[List[str]] = None,
        scope_filter: Optional[Dict[str, Any]] = None,
        sort: Optional[Dict[str, int]] = None,
        skip: int = 0,
        limit: Optional[int] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build an aggregation pipeline for an enriched query."""
        match_query = self._merge_match(filters or {}, scope_filter or {})
        normalized_paths = normalize_enrichments(
            base_collection,
            enrichments,
            max_depth=self.MAX_ENRICHMENT_DEPTH,
        )

        pipeline: List[Dict[str, Any]] = []
        if match_query:
            pipeline.append({"$match": match_query})

        path_meta: Dict[str, Dict[str, Any]] = {
            "": {
                "collection": base_collection,
                "doc_path": "",
                "is_array": False,
                "container_array_path": None,
            }
        }

        for path in normalized_paths:
            self._add_lookup_for_path(
                pipeline=pipeline,
                path=path,
                path_meta=path_meta,
            )

        if sort:
            pipeline.append({"$sort": sort})
        if skip > 0:
            pipeline.append({"$skip": max(0, int(skip))})
        if limit is not None and int(limit) > 0:
            pipeline.append({"$limit": int(limit)})
        if projection:
            pipeline.append({"$project": projection})

        return pipeline

    def build_count_query(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        scope_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a fast count_documents-style query."""
        return self._merge_match(filters or {}, scope_filter or {})

    def _add_lookup_for_path(
        self,
        *,
        pipeline: List[Dict[str, Any]],
        path: str,
        path_meta: Dict[str, Dict[str, Any]],
    ) -> None:
        tokens = path.split(".")
        parent_path = ".".join(tokens[:-1])
        relationship_key = tokens[-1]
        parent = path_meta.get(parent_path)
        if not parent:
            return

        spec = get_relationship(parent["collection"], relationship_key)
        if spec is None:
            return

        parent_doc_path = str(parent["doc_path"] or "")
        target_doc_path = spec.as_field if not parent_doc_path else f"{parent_doc_path}.{spec.as_field}"
        temp_field = f"__join_{path.replace('.', '_')}"
        local_field = spec.local_field if not parent_doc_path else f"{parent_doc_path}.{spec.local_field}"

        pipeline.append(
            {
                "$lookup": {
                    "from": spec.collection,
                    "localField": local_field,
                    "foreignField": spec.foreign_field,
                    "as": temp_field,
                }
            }
        )

        container_array_path = parent.get("container_array_path")
        if parent.get("is_array"):
            container_array_path = parent_doc_path

        if container_array_path:
            self._merge_lookup_into_array_path(
                pipeline=pipeline,
                temp_field=temp_field,
                parent_doc_path=parent_doc_path,
                container_array_path=container_array_path,
                spec=spec,
            )
        elif parent_doc_path:
            local_value_expr = f"${parent_doc_path}.{spec.local_field}"
            joined_expr = self._joined_value_expression(
                temp_field=temp_field,
                local_value_expression=local_value_expr,
                foreign_field=spec.foreign_field,
                is_one=spec.is_one,
            )
            pipeline.append(
                {
                    "$addFields": {
                        parent_doc_path: {
                            "$mergeObjects": [
                                {"$ifNull": [f"${parent_doc_path}", {}]},
                                {spec.as_field: joined_expr},
                            ]
                        }
                    }
                }
            )
        else:
            pipeline.append({"$addFields": {spec.as_field: f"${temp_field}"}})
            if spec.is_one:
                pipeline.append(
                    {
                        "$unwind": {
                            "path": f"${spec.as_field}",
                            "preserveNullAndEmptyArrays": spec.preserve_null,
                        }
                    }
                )

        pipeline.append({"$unset": temp_field})

        path_meta[path] = {
            "collection": spec.collection,
            "doc_path": target_doc_path,
            "is_array": not spec.is_one,
            "container_array_path": container_array_path,
        }

    def _merge_lookup_into_array_path(
        self,
        *,
        pipeline: List[Dict[str, Any]],
        temp_field: str,
        parent_doc_path: str,
        container_array_path: str,
        spec: RelationshipSpec,
    ) -> None:
        relative_path = self._relative_path(
            full_path=parent_doc_path,
            container_path=container_array_path,
        )
        relative_tokens = [t for t in relative_path.split(".") if t] if relative_path else []

        item_alias = "item"
        local_value_expression = self._scoped_value_expression(
            alias=item_alias,
            nested_tokens=relative_tokens + [spec.local_field],
        )
        joined_expr = self._joined_value_expression(
            temp_field=temp_field,
            local_value_expression=local_value_expression,
            foreign_field=spec.foreign_field,
            is_one=spec.is_one,
        )
        nested_patch = self._build_nested_patch(
            alias=item_alias,
            path_tokens=relative_tokens,
            field_name=spec.as_field,
            field_value=joined_expr,
        )

        pipeline.append(
            {
                "$addFields": {
                    container_array_path: {
                        "$map": {
                            "input": {"$ifNull": [f"${container_array_path}", []]},
                            "as": item_alias,
                            "in": {
                                "$mergeObjects": [
                                    f"$${item_alias}",
                                    nested_patch,
                                ]
                            },
                        }
                    }
                }
            }
        )

    def _joined_value_expression(
        self,
        *,
        temp_field: str,
        local_value_expression: Any,
        foreign_field: str,
        is_one: bool,
    ) -> Dict[str, Any]:
        filtered = {
            "$filter": {
                "input": f"${temp_field}",
                "as": "joined",
                "cond": {
                    "$eq": [
                        f"$$joined.{foreign_field}",
                        local_value_expression,
                    ]
                },
            }
        }
        if is_one:
            return {"$arrayElemAt": [filtered, 0]}
        return filtered

    def _build_nested_patch(
        self,
        *,
        alias: str,
        path_tokens: List[str],
        field_name: str,
        field_value: Any,
    ) -> Dict[str, Any]:
        if not path_tokens:
            return {field_name: field_value}

        head = path_tokens[0]
        remainder = path_tokens[1:]
        base_path = self._scoped_value_expression(alias=alias, nested_tokens=[head])
        return {
            head: {
                "$mergeObjects": [
                    {"$ifNull": [base_path, {}]},
                    self._build_nested_patch(
                        alias=alias,
                        path_tokens=remainder,
                        field_name=field_name,
                        field_value=field_value,
                    ),
                ]
            }
        }

    def _scoped_value_expression(self, *, alias: str, nested_tokens: List[str]) -> str:
        if not nested_tokens:
            return f"$${alias}"
        return "$$" + alias + "." + ".".join(nested_tokens)

    def _relative_path(self, *, full_path: str, container_path: str) -> str:
        if not full_path or full_path == container_path:
            return ""
        if full_path.startswith(f"{container_path}."):
            return full_path[len(container_path) + 1:]
        return full_path

    def _merge_match(self, base: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
        base_query = deepcopy(base) if base else {}
        scope_query = deepcopy(scope) if scope else {}
        if not base_query:
            return scope_query
        if not scope_query:
            return base_query
        return {"$and": [base_query, scope_query]}

