"""
MongoDB Aggregation Pipeline Builder

Provides a fluent API for building aggregation pipelines with safety controls.
"""

from typing import Any, Dict, List, Optional, Union
from bson import ObjectId

from mcp.config import mcp_settings
from mcp.constants import Collections


class AggregationBuilder:
    """
    Safe MongoDB aggregation pipeline builder.
    Enforces stage limits and validates pipeline structure.
    """

    MAX_STAGES = mcp_settings.MCP_MAX_AGGREGATION_STAGES

    # Allowed lookup targets (whitelist)
    ALLOWED_LOOKUPS = {
        Collections.RANK_MASTER,
        Collections.DISTRICT,
        Collections.UNIT,
        Collections.PERSONNEL_MASTER,
        Collections.DESIGNATION_MASTER,
        Collections.MANDAL,
        Collections.UNIT_VILLAGES,
        Collections.DEPARTMENT,
        Collections.UNIT_TYPE,
        Collections.ROLES,
        Collections.ROLES_MASTER,
        Collections.APPROVAL_FLOW_MASTER,
        Collections.ERROR_MASTER,
        Collections.LOG_MASTER,
        Collections.MODULES_MASTER,
        Collections.NOTIFICATION_MASTER,
        Collections.PROMPT_MASTER,
        Collections.VALUE_SETS_MASTER,
    }

    def __init__(self):
        self._stages: List[Dict[str, Any]] = []

    def match(self, query: Dict[str, Any]) -> "AggregationBuilder":
        """Add $match stage"""
        if query:
            self._stages.append({"$match": query})
        return self

    def lookup(
        self,
        from_collection: str,
        local_field: str,
        foreign_field: str,
        as_field: str,
    ) -> "AggregationBuilder":
        """Add $lookup stage with validation"""
        if from_collection not in self.ALLOWED_LOOKUPS:
            raise ValueError(f"Lookup to '{from_collection}' not allowed")

        self._stages.append(
            {
                "$lookup": {
                    "from": from_collection,
                    "localField": local_field,
                    "foreignField": foreign_field,
                    "as": as_field,
                }
            }
        )
        return self

    def lookup_pipeline(
        self,
        from_collection: str,
        let_vars: Dict[str, str],
        pipeline: List[Dict[str, Any]],
        as_field: str,
    ) -> "AggregationBuilder":
        """Add $lookup stage with sub-pipeline"""
        if from_collection not in self.ALLOWED_LOOKUPS:
            raise ValueError(f"Lookup to '{from_collection}' not allowed")

        self._stages.append(
            {
                "$lookup": {
                    "from": from_collection,
                    "let": let_vars,
                    "pipeline": pipeline,
                    "as": as_field,
                }
            }
        )
        return self

    def unwind(
        self, path: str, preserve_null: bool = True
    ) -> "AggregationBuilder":
        """Add $unwind stage"""
        path_str = f"${path}" if not path.startswith("$") else path
        self._stages.append(
            {
                "$unwind": {
                    "path": path_str,
                    "preserveNullAndEmptyArrays": preserve_null,
                }
            }
        )
        return self

    def unwind_simple(self, path: str) -> "AggregationBuilder":
        """Add simple $unwind stage (filters out nulls)"""
        path_str = f"${path}" if not path.startswith("$") else path
        self._stages.append({"$unwind": path_str})
        return self

    def group(
        self, group_by: Any, aggregations: Dict[str, Any]
    ) -> "AggregationBuilder":
        """Add $group stage"""
        group_spec = {"_id": group_by}
        group_spec.update(aggregations)
        self._stages.append({"$group": group_spec})
        return self

    def project(self, projection: Dict[str, Any]) -> "AggregationBuilder":
        """Add $project stage"""
        if projection:
            self._stages.append({"$project": projection})
        return self

    def add_fields(self, fields: Dict[str, Any]) -> "AggregationBuilder":
        """Add $addFields stage"""
        if fields:
            self._stages.append({"$addFields": fields})
        return self

    def sort(self, sort_spec: Dict[str, int]) -> "AggregationBuilder":
        """Add $sort stage"""
        if sort_spec:
            self._stages.append({"$sort": sort_spec})
        return self

    def skip(self, count: int) -> "AggregationBuilder":
        """Add $skip stage"""
        if count > 0:
            self._stages.append({"$skip": count})
        return self

    def limit(self, count: int) -> "AggregationBuilder":
        """Add $limit stage"""
        # Enforce maximum limit
        count = min(count, mcp_settings.MCP_MAX_RESULTS)
        if count > 0:
            self._stages.append({"$limit": count})
        return self

    def count(self, field_name: str = "total") -> "AggregationBuilder":
        """Add $count stage"""
        self._stages.append({"$count": field_name})
        return self

    def facet(self, facets: Dict[str, List[Dict[str, Any]]]) -> "AggregationBuilder":
        """Add $facet stage for multiple aggregations"""
        self._stages.append({"$facet": facets})
        return self

    def replace_root(self, new_root: Union[str, Dict[str, Any]]) -> "AggregationBuilder":
        """Add $replaceRoot stage"""
        if isinstance(new_root, str):
            new_root = {"newRoot": f"${new_root}" if not new_root.startswith("$") else new_root}
        else:
            new_root = {"newRoot": new_root}
        self._stages.append({"$replaceRoot": new_root})
        return self

    def build(self) -> List[Dict[str, Any]]:
        """Build and validate the pipeline"""
        if len(self._stages) > self.MAX_STAGES:
            raise ValueError(
                f"Pipeline exceeds maximum stages ({self.MAX_STAGES}), "
                f"current: {len(self._stages)}"
            )
        return self._stages.copy()

    def reset(self) -> "AggregationBuilder":
        """Reset the pipeline for reuse"""
        self._stages = []
        return self

    def __len__(self) -> int:
        """Return number of stages"""
        return len(self._stages)


class AggregationHelpers:
    """Common aggregation expressions and patterns"""

    @staticmethod
    def to_string(field: str) -> Dict[str, Any]:
        """Convert ObjectId to string"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$toString": field_ref}

    @staticmethod
    def if_null(field: str, default: Any) -> Dict[str, Any]:
        """Provide default value if field is null"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$ifNull": [field_ref, default]}

    @staticmethod
    def cond(condition: Dict, if_true: Any, if_false: Any) -> Dict[str, Any]:
        """Conditional expression"""
        return {"$cond": {"if": condition, "then": if_true, "else": if_false}}

    @staticmethod
    def array_size(field: str) -> Dict[str, Any]:
        """Get size of array field"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$size": {"$ifNull": [field_ref, []]}}

    @staticmethod
    def concat_arrays(*arrays: str) -> Dict[str, Any]:
        """Concatenate arrays"""
        return {"$concatArrays": [f"${a}" if not a.startswith("$") else a for a in arrays]}

    @staticmethod
    def in_array(element: Any, array: str) -> Dict[str, Any]:
        """Check if element is in array"""
        array_ref = f"${array}" if not array.startswith("$") else array
        return {"$in": [element, array_ref]}

    @staticmethod
    def sum_field(field: str) -> Dict[str, Any]:
        """Sum aggregation"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$sum": field_ref}

    @staticmethod
    def count_docs() -> Dict[str, Any]:
        """Count documents in group"""
        return {"$sum": 1}

    @staticmethod
    def first(field: str) -> Dict[str, Any]:
        """First value in group"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$first": field_ref}

    @staticmethod
    def push(field: str) -> Dict[str, Any]:
        """Push values to array"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$push": field_ref}

    @staticmethod
    def add_to_set(field: str) -> Dict[str, Any]:
        """Add unique values to array"""
        field_ref = f"${field}" if not field.startswith("$") else field
        return {"$addToSet": field_ref}

