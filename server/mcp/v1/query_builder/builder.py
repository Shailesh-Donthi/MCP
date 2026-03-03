"""
Safe Query Builder for MongoDB

Provides safe parameterized query construction to prevent injection attacks.
All user inputs are validated and sanitized before being used in queries.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from bson import ObjectId


class SafeQueryBuilder:
    """
    Safe MongoDB query builder with parameterization.
    Prevents injection by validating and sanitizing all inputs.
    """

    ALLOWED_OPERATORS = {
        "$eq",
        "$ne",
        "$gt",
        "$gte",
        "$lt",
        "$lte",
        "$in",
        "$nin",
        "$exists",
        "$regex",
        "$and",
        "$or",
        "$not",
        "$elemMatch",
    }

    ALLOWED_SORT_ORDERS = {1, -1}

    @staticmethod
    def is_valid_object_id(value: str) -> bool:
        """Check if a string is a valid ObjectId"""
        return ObjectId.is_valid(value)

    @staticmethod
    def validate_object_id(value: str) -> ObjectId:
        """Validate and convert string to ObjectId"""
        if not ObjectId.is_valid(value):
            raise ValueError(f"Invalid ObjectId: {value}")
        return ObjectId(value)

    @staticmethod
    def try_parse_object_id(value: str) -> Union[ObjectId, str]:
        """Try to parse as ObjectId, return original string if invalid"""
        if ObjectId.is_valid(value):
            return ObjectId(value)
        return value

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Expected string value")
        # Truncate to max length
        value = value[:max_length]
        # Remove null bytes
        value = value.replace("\x00", "")
        return value

    @staticmethod
    def escape_regex(pattern: str) -> str:
        """Escape special regex characters in a pattern"""
        return re.escape(pattern)

    @classmethod
    def build_regex_pattern(
        cls, search_term: str, case_insensitive: bool = True
    ) -> Dict[str, Any]:
        """Build safe regex pattern for text search"""
        escaped = cls.escape_regex(search_term)
        options = "i" if case_insensitive else ""
        return {"$regex": escaped, "$options": options}

    @classmethod
    def build_contains_pattern(
        cls, search_term: str, case_insensitive: bool = True
    ) -> Dict[str, Any]:
        """Build pattern to match if field contains search term"""
        escaped = cls.escape_regex(search_term)
        options = "i" if case_insensitive else ""
        return {"$regex": f".*{escaped}.*", "$options": options}

    @classmethod
    def build_starts_with_pattern(
        cls, search_term: str, case_insensitive: bool = True
    ) -> Dict[str, Any]:
        """Build pattern to match if field starts with search term"""
        escaped = cls.escape_regex(search_term)
        options = "i" if case_insensitive else ""
        return {"$regex": f"^{escaped}", "$options": options}

    @classmethod
    def build_match_query(
        cls,
        filters: Dict[str, Any],
        allowed_fields: List[str],
        include_soft_delete_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a safe $match query from filters.
        Only allows querying on whitelisted fields.

        Args:
            filters: Dictionary of field -> value filters
            allowed_fields: List of field names that can be queried
            include_soft_delete_filter: Whether to add isDelete: False

        Returns:
            Safe MongoDB query dictionary
        """
        query: Dict[str, Any] = {}

        for field, value in filters.items():
            # Only allow whitelisted fields
            if field not in allowed_fields:
                continue

            # Handle different value types
            if value is None:
                continue
            elif isinstance(value, str):
                if ObjectId.is_valid(value):
                    query[field] = cls.validate_object_id(value)
                else:
                    query[field] = cls.sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                query[field] = value
            elif isinstance(value, datetime):
                query[field] = value
            elif isinstance(value, ObjectId):
                query[field] = value
            elif isinstance(value, dict):
                # Handle operators
                operator_query = cls._process_operator_dict(value)
                if operator_query:
                    query[field] = operator_query
            elif isinstance(value, list):
                # Handle $in by default for lists
                processed_list = []
                for item in value:
                    if isinstance(item, str) and ObjectId.is_valid(item):
                        processed_list.append(cls.validate_object_id(item))
                    else:
                        processed_list.append(item)
                query[field] = {"$in": processed_list}

        # Always add soft delete filter unless explicitly disabled
        if include_soft_delete_filter:
            query["isDelete"] = False

        return query

    @classmethod
    def _process_operator_dict(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Process a dictionary containing MongoDB operators"""
        operator_query = {}

        for op, op_value in value.items():
            if op not in cls.ALLOWED_OPERATORS:
                continue

            if op in ("$in", "$nin"):
                if isinstance(op_value, list):
                    processed = []
                    for v in op_value:
                        if isinstance(v, str) and ObjectId.is_valid(v):
                            processed.append(cls.validate_object_id(v))
                        else:
                            processed.append(v)
                    operator_query[op] = processed
            elif op == "$regex":
                # Escape the regex pattern for safety
                if isinstance(op_value, str):
                    operator_query[op] = cls.escape_regex(op_value)
            elif op in ("$and", "$or"):
                # Recursively process logical operators
                if isinstance(op_value, list):
                    operator_query[op] = [
                        cls._process_operator_dict(item)
                        if isinstance(item, dict)
                        else item
                        for item in op_value
                    ]
            else:
                operator_query[op] = op_value

        return operator_query

    @classmethod
    def build_date_range_query(
        cls,
        field: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Build date range query"""
        if not start_date and not end_date:
            return {}

        date_query: Dict[str, Any] = {}
        if start_date:
            date_query["$gte"] = start_date
        if end_date:
            date_query["$lte"] = end_date

        return {field: date_query} if date_query else {}

    @classmethod
    def build_text_search_query(
        cls, search_term: str, fields: List[str]
    ) -> Dict[str, Any]:
        """
        Build a multi-field text search query using $or.

        Args:
            search_term: The search term
            fields: List of fields to search in

        Returns:
            Query with $or conditions for each field
        """
        if not search_term or not fields:
            return {}

        pattern = cls.build_contains_pattern(search_term)
        return {"$or": [{field: pattern} for field in fields]}

    @classmethod
    def build_sort_spec(
        cls, sort_field: str, sort_order: int = -1, allowed_fields: List[str] = None
    ) -> Dict[str, int]:
        """
        Build a safe sort specification.

        Args:
            sort_field: Field to sort by
            sort_order: 1 for ascending, -1 for descending
            allowed_fields: Optional whitelist of sortable fields

        Returns:
            Sort specification dictionary
        """
        # Validate sort order
        if sort_order not in cls.ALLOWED_SORT_ORDERS:
            sort_order = -1

        # Validate field if whitelist provided
        if allowed_fields and sort_field not in allowed_fields:
            sort_field = "createdAt"  # Default fallback

        return {sort_field: sort_order}

    @classmethod
    def build_projection(
        cls, include_fields: List[str] = None, exclude_fields: List[str] = None
    ) -> Optional[Dict[str, int]]:
        """
        Build a projection specification.

        Args:
            include_fields: Fields to include (uses inclusion projection)
            exclude_fields: Fields to exclude (uses exclusion projection)

        Returns:
            Projection dictionary or None
        """
        if include_fields:
            return {field: 1 for field in include_fields}
        elif exclude_fields:
            return {field: 0 for field in exclude_fields}
        return None

    @classmethod
    def merge_queries(cls, *queries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple queries using $and.

        Args:
            *queries: Variable number of query dictionaries

        Returns:
            Merged query
        """
        non_empty = [q for q in queries if q]
        if not non_empty:
            return {}
        if len(non_empty) == 1:
            return non_empty[0]
        return {"$and": non_empty}

