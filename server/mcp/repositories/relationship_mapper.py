"""Relationship registry used by V2 repository enrichment pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from mcp.constants import Collections


@dataclass(frozen=True)
class RelationshipSpec:
    """Defines how two collections are joined during enrichment."""

    key: str
    collection: str
    local_field: str
    foreign_field: str
    as_field: str
    cardinality: str = "many"  # one | many
    enrich_by_default: bool = False
    preserve_null: bool = True

    @property
    def is_one(self) -> bool:
        return self.cardinality == "one"


RELATIONSHIP_MAP: Dict[str, Dict[str, RelationshipSpec]] = {
    Collections.PERSONNEL_MASTER: {
        "assignments": RelationshipSpec(
            key="assignments",
            collection=Collections.ASSIGNMENT_MASTER,
            local_field="_id",
            foreign_field="userId",
            as_field="assignments",
            cardinality="many",
            enrich_by_default=True,
        ),
        "rank": RelationshipSpec(
            key="rank",
            collection=Collections.RANK_MASTER,
            local_field="rankId",
            foreign_field="_id",
            as_field="rank",
            cardinality="one",
            enrich_by_default=True,
        ),
        "designation": RelationshipSpec(
            key="designation",
            collection=Collections.DESIGNATION_MASTER,
            local_field="designationId",
            foreign_field="_id",
            as_field="designation",
            cardinality="one",
            enrich_by_default=False,
        ),
        "department": RelationshipSpec(
            key="department",
            collection=Collections.DEPARTMENT,
            local_field="departmentId",
            foreign_field="_id",
            as_field="department",
            cardinality="one",
            enrich_by_default=True,
        ),
    },
    Collections.ASSIGNMENT_MASTER: {
        "personnel": RelationshipSpec(
            key="personnel",
            collection=Collections.PERSONNEL_MASTER,
            local_field="userId",
            foreign_field="_id",
            as_field="personnel",
            cardinality="one",
            enrich_by_default=True,
        ),
        "unit": RelationshipSpec(
            key="unit",
            collection=Collections.UNIT,
            local_field="unitId",
            foreign_field="_id",
            as_field="unit",
            cardinality="one",
            enrich_by_default=True,
        ),
        "designation": RelationshipSpec(
            key="designation",
            collection=Collections.DESIGNATION_MASTER,
            local_field="designationId",
            foreign_field="_id",
            as_field="designation",
            cardinality="one",
            enrich_by_default=True,
        ),
    },
    Collections.UNIT: {
        "district": RelationshipSpec(
            key="district",
            collection=Collections.DISTRICT,
            local_field="districtId",
            foreign_field="_id",
            as_field="district",
            cardinality="one",
            enrich_by_default=True,
        ),
        "parent_unit": RelationshipSpec(
            key="parent_unit",
            collection=Collections.UNIT,
            local_field="parentUnitId",
            foreign_field="_id",
            as_field="parentUnit",
            cardinality="one",
            enrich_by_default=False,
        ),
        "unit_type": RelationshipSpec(
            key="unit_type",
            collection=Collections.UNIT_TYPE,
            local_field="unitTypeId",
            foreign_field="_id",
            as_field="unitType",
            cardinality="one",
            enrich_by_default=True,
        ),
        "responsible_user": RelationshipSpec(
            key="responsible_user",
            collection=Collections.PERSONNEL_MASTER,
            local_field="responsibleUserId",
            foreign_field="_id",
            as_field="responsibleUser",
            cardinality="one",
            enrich_by_default=False,
        ),
    },
}


def get_relationship(
    collection_name: str,
    relationship_key: str,
) -> Optional[RelationshipSpec]:
    """Fetch a relationship definition for a collection."""
    return RELATIONSHIP_MAP.get(collection_name, {}).get(relationship_key)


def get_default_enrichments(collection_name: str) -> List[str]:
    """Return the default enrichment list for a collection."""
    relationships = RELATIONSHIP_MAP.get(collection_name, {})
    return [
        spec.key
        for spec in relationships.values()
        if spec.enrich_by_default
    ]


def iter_relationships(collection_name: str) -> Iterable[RelationshipSpec]:
    """Iterate relationships for a collection."""
    return RELATIONSHIP_MAP.get(collection_name, {}).values()


def normalize_enrichments(
    base_collection: str,
    enrichments: Optional[List[str]],
    *,
    max_depth: int = 4,
) -> List[str]:
    """
    Normalize enrichment paths.

    - Applies defaults when enrichments is empty/None.
    - Keeps only valid relationships.
    - Ensures parent paths exist before nested paths.
    - Caps depth to prevent expensive pipelines.
    """
    requested = enrichments or get_default_enrichments(base_collection)
    ordered: List[str] = []
    seen: set[str] = set()

    for path in requested:
        if not isinstance(path, str):
            continue
        tokens = [t.strip() for t in path.split(".") if t.strip()]
        if not tokens:
            continue
        if len(tokens) > max_depth:
            tokens = tokens[:max_depth]

        current_collection = base_collection
        built: List[str] = []
        for token in tokens:
            spec = get_relationship(current_collection, token)
            if spec is None:
                break
            built.append(token)
            normalized_path = ".".join(built)
            if normalized_path not in seen:
                seen.add(normalized_path)
                ordered.append(normalized_path)
            current_collection = spec.collection

    return ordered

