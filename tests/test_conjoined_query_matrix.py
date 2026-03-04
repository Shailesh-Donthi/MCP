import unittest
from typing import Any
import re

from mcp.router import repair_route
from mcp.v1.server_http import route_query_to_tool as route_query_to_tool_v1
from mcp.v2.server_http import route_query_to_tool as route_query_to_tool_v2
try:
    from tests.query_case_loader import load_query_cases_with_expected_results
except ModuleNotFoundError:
    from query_case_loader import load_query_cases_with_expected_results


def _route_v1_repaired(query: str):
    tool, args = route_query_to_tool_v1(query)
    return repair_route(query, tool, args)


class ConjoinedQueryMatrixTests(unittest.TestCase):
    CASES = load_query_cases_with_expected_results()

    _BANNED_DISTRICT_VALUES = {"and", "also", "then", "count", "give", "them", "many", "find"}

    @staticmethod
    def _normalize_value_for_compare(key: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        normalized = " ".join(value.split())
        if key == "name":
            normalized = re.sub(r"^(?:personnel|person|officer)\s+", "", normalized, flags=re.IGNORECASE)

        if key in {"name", "unit_name", "district_name", "designation_name", "rank_name"}:
            return normalized.lower()

        return normalized

    def _assert_subset(self, actual: dict, expected_subset: dict, query: str) -> None:
        for key, expected_value in expected_subset.items():
            self.assertIn(key, actual, msg=f"Missing key '{key}' for query: {query}")
            actual_value = actual.get(key)
            if isinstance(expected_value, list):
                normalized_expected = [self._normalize_value_for_compare(key, item) for item in expected_value]
                normalized_actual = [self._normalize_value_for_compare(key, item) for item in (actual_value or [])]
                self.assertEqual(
                    sorted(normalized_expected),
                    sorted(normalized_actual),
                    msg=f"Unexpected list value for '{key}' on query: {query}",
                )
            else:
                self.assertEqual(
                    self._normalize_value_for_compare(key, expected_value),
                    self._normalize_value_for_compare(key, actual_value),
                    msg=f"Unexpected '{key}' for query: {query}",
                )

    def _assert_no_clause_pollution(self, args: dict, query: str) -> None:
        for key in ("district_name", "unit_name", "name", "designation_name"):
            value = args.get(key)
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            self.assertNotRegex(
                lowered,
                r"\b(and|also|then)\b\s+(show|list|get|give|provide|find|where|who|what|which|how|compare|count)\b",
                msg=f"Conjoined clause leakage in '{key}' for query: {query}",
            )
            if key == "district_name":
                self.assertNotIn(lowered.strip(), self._BANNED_DISTRICT_VALUES, msg=f"Bad district parse for query: {query}")

        district_names = args.get("district_names")
        if isinstance(district_names, list):
            for item in district_names:
                text = str(item or "").strip().lower()
                self.assertTrue(text, msg=f"Empty district_names item for query: {query}")
                self.assertNotIn(text, self._BANNED_DISTRICT_VALUES, msg=f"Bad district_names item for query: {query}")
                self.assertNotRegex(
                    text,
                    r"\b(and|also|then)\b",
                    msg=f"Conjunction leakage in district_names for query: {query}",
                )

    def _extract_expected_result(self, case: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[str]]:
        case_id = str(case.get("id", "")).strip()
        query = str(case.get("query", "")).strip()
        self.assertTrue(query, msg="Each query case must include a non-empty 'query'")
        self.assertTrue(case_id, msg=f"Case for query '{query}' is missing 'id'")

        expected = case.get("expected_result")
        self.assertIsInstance(expected, dict, msg=f"Case '{query}' must include expected_result object")

        expected_tool = str(expected.get("tool", "")).strip()
        self.assertTrue(expected_tool, msg=f"Case '{query}' expected_result.tool is required")

        expected_subset = expected.get("args_subset", {})
        self.assertIsInstance(expected_subset, dict, msg=f"Case '{query}' expected_result.args_subset must be a dict")

        absent_keys = expected.get("absent_keys", [])
        self.assertIsInstance(absent_keys, list, msg=f"Case '{query}' expected_result.absent_keys must be a list")
        return case_id, expected_tool, expected_subset, absent_keys

    def test_expected_result_schema(self) -> None:
        ids: set[str] = set()
        for case in self.CASES:
            case_id = str(case.get("id", "")).strip()
            query = str(case.get("query", "")).strip()
            self.assertTrue(case_id, msg=f"Case for query '{query}' is missing 'id'")
            self.assertNotIn(case_id, ids, msg=f"Duplicate query case id found: {case_id}")
            ids.add(case_id)

            expected = case.get("expected_result", {})
            self.assertIsInstance(expected, dict, msg=f"Case '{case_id}' expected_result must be a dict")
            for key in ("response_should_contain", "response_should_not_contain"):
                value = expected.get(key, [])
                self.assertIsInstance(value, list, msg=f"Case '{case_id}' expected_result.{key} must be a list")

    def test_conjoined_query_matrix_v1_and_v2(self) -> None:
        for case in self.CASES:
            query = case["query"]
            case_id, expected_tool, expected_subset, absent_keys = self._extract_expected_result(case)

            with self.subTest(query=f"v1::{case_id}::{query}"):
                v1_tool, v1_args = _route_v1_repaired(query)
                self.assertEqual(expected_tool, v1_tool, msg=f"Unexpected V1 route for query: {query}")
                self._assert_subset(v1_args, expected_subset, query)
                for key in absent_keys:
                    self.assertNotIn(key, v1_args, msg=f"Key '{key}' should be absent for query: {query}")
                self._assert_no_clause_pollution(v1_args, query)

            with self.subTest(query=f"v2::{case_id}::{query}"):
                v2_tool, v2_args = route_query_to_tool_v2(query)
                self.assertEqual(expected_tool, v2_tool, msg=f"Unexpected V2 route for query: {query}")
                self._assert_subset(v2_args, expected_subset, query)
                for key in absent_keys:
                    self.assertNotIn(key, v2_args, msg=f"Key '{key}' should be absent for query: {query}")
                self._assert_no_clause_pollution(v2_args, query)

            with self.subTest(query=f"parity::{case_id}::{query}"):
                v1_tool, v1_args = _route_v1_repaired(query)
                v2_tool, v2_args = route_query_to_tool_v2(query)
                self.assertEqual(v1_tool, v2_tool, msg=f"V1/V2 tool mismatch for query: {query}")
                self.assertEqual(v1_args, v2_args, msg=f"V1/V2 args mismatch for query: {query}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
