import unittest
from unittest.mock import AsyncMock, patch

from mcp.llm_router import IntelligentQueryHandler, _parse_route_response, llm_route_query
from mcp.router.extractors import (
    extract_person_hint,
    extract_place_hints,
    extract_rank_hints,
    extract_unit_hint,
    is_followup_district_query,
    normalize_common_query_typos,
)
from mcp.router.routing_rules import repair_route
from mcp.server_http import route_query_to_tool


class RouteQueryToToolTests(unittest.TestCase):
    def assert_route(self, query: str, expected_tool: str, expected_args_subset: dict | None = None) -> None:
        tool, args = route_query_to_tool(query)
        self.assertEqual(expected_tool, tool, msg=f"Unexpected tool for query: {query}")
        if expected_args_subset:
            for key, value in expected_args_subset.items():
                self.assertIn(key, args, msg=f"Expected arg '{key}' missing for query: {query}")
                self.assertEqual(value, args[key], msg=f"Unexpected arg '{key}' for query: {query}")

    def test_list_districts_query(self) -> None:
        self.assert_route("which districts are available", "list_districts")

    def test_help_query(self) -> None:
        self.assert_route("what can you help me with", "__help__")

    def test_master_data_query(self) -> None:
        tool, args = route_query_to_tool("show notification master entries linked to modules")
        self.assertEqual("query_linked_master_data", tool)
        self.assertEqual("notification_master", args.get("collection"))
        self.assertTrue(args.get("include_related"))
        self.assertTrue(args.get("include_reverse"))

    def test_master_data_discover_query(self) -> None:
        tool, args = route_query_to_tool("how are roles and permissions interlinked")
        self.assertEqual("query_linked_master_data", tool)
        self.assertEqual("discover", args.get("mode"))
        self.assertTrue(args.get("include_integrity"))

    def test_missing_village_mapping_query(self) -> None:
        self.assert_route(
            "find missing village mappings in chittoor district",
            "find_missing_village_mappings",
            {"district_name": "Chittoor"},
        )

    def test_village_coverage_query(self) -> None:
        tool, args = route_query_to_tool("which villages are mapped to k v palli ps")
        self.assertEqual("get_village_coverage", tool)
        self.assertIn("unit_name", args)
        self.assertTrue(str(args["unit_name"]).strip())

    def test_hierarchy_query(self) -> None:
        self.assert_route(
            "show hierarchy of chittoor district",
            "get_unit_hierarchy",
            {"district_name": "Chittoor"},
        )

    def test_transfer_query_with_days(self) -> None:
        self.assert_route(
            "show recent transfers for 15 days",
            "query_recent_transfers",
            {"days": 15},
        )

    @unittest.expectedFailure
    def test_transfer_query_with_bare_place_should_capture_district(self) -> None:
        # Known gap: bare place names in transfer queries (without the word "district")
        # are not always captured as district_name by route_query_to_tool.
        self.assert_route(
            "show recent transfers in guntur for 15 days",
            "query_recent_transfers",
            {"district_name": "Guntur", "days": 15},
        )

    def test_command_history_query(self) -> None:
        tool, args = route_query_to_tool("who was the sho of guntur ps in last 15 days")
        self.assertEqual("get_unit_command_history", tool)
        self.assertIn("unit_name", args)

    def test_person_attribute_query(self) -> None:
        self.assert_route(
            "what is the mobile number of A Ashok Kumar",
            "search_personnel",
            {"name": "A Ashok Kumar"},
        )

    def test_responsible_user_query(self) -> None:
        tool, args = route_query_to_tool("who is the sdpo of kuppam")
        self.assertEqual("get_unit_command_history", tool)
        self.assertEqual("Kuppam SDPO", args.get("unit_name"))

    def test_list_units_query(self) -> None:
        self.assert_route(
            "list all units in guntur district",
            "list_units_in_district",
            {"district_name": "Guntur"},
        )

    def test_distribution_query(self) -> None:
        self.assert_route(
            "how many personnel are in guntur district",
            "get_personnel_distribution",
            {"district_name": "Guntur", "group_by": "rank"},
        )

    def test_vacancy_query(self) -> None:
        self.assert_route(
            "show vacancies in guntur district",
            "count_vacancies_by_unit_rank",
            {"district_name": "Guntur"},
        )

    def test_personnel_by_rank_query(self) -> None:
        self.assert_route(
            "list all SI in guntur district",
            "query_personnel_by_rank",
            {"district_name": "Guntur"},
        )

    def test_personnel_in_district_falls_back_to_distribution(self) -> None:
        self.assert_route(
            "list personnel in guntur district",
            "get_personnel_distribution",
            {"district_name": "Guntur", "group_by": "rank"},
        )

    def test_search_unit_query(self) -> None:
        tool, args = route_query_to_tool("where is kuppam sdpo")
        self.assertEqual("search_unit", tool)
        self.assertIn("name", args)

    def test_person_specific_query(self) -> None:
        self.assert_route(
            "tell me about person Ravi Kumar",
            "search_personnel",
            {"name": "Ravi Kumar"},
        )

    def test_default_query_route(self) -> None:
        self.assert_route(
            "abracadabra",
            "get_personnel_distribution",
            {"group_by": "rank"},
        )

    def test_sort_phrase_not_treated_as_district(self) -> None:
        tool, args = route_query_to_tool("list all SI in alphabetical order")
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertNotIn("district_name", args)


class RepairRouteTests(unittest.TestCase):
    def test_multi_district_rank_contact_query(self) -> None:
        query = "Find all Circle Inspectors in Guntur and Chittoor districts and show their contact details"
        tool, args = repair_route(query, "get_personnel_distribution", {})
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertEqual("Circle Inspector", args.get("rank_name"))
        self.assertEqual(["Guntur", "Chittoor"], args.get("district_names"))
        self.assertEqual(200, args.get("page_size"))

    def test_sort_phrase_removed_from_rank_query(self) -> None:
        tool, args = repair_route(
            "list all SI in alphabetical order",
            "query_personnel_by_rank",
            {"rank_name": "Sub Inspector", "district_name": "Alphabetical Order"},
        )
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertNotIn("district_name", args)

    def test_sp_of_district_keeps_district(self) -> None:
        tool, args = repair_route(
            "who is the SP of guntur",
            "query_personnel_by_rank",
            {"rank_name": "Superintendent of Police", "rank_relation": "exact"},
        )
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertEqual("Guntur", args.get("district_name"))

    def test_master_relation_forces_discover_mode(self) -> None:
        tool, args = repair_route(
            "how are modules and notifications interlinked",
            "list_districts",
            {},
        )
        self.assertEqual("query_linked_master_data", tool)
        self.assertEqual("discover", args.get("mode"))
        self.assertTrue(args.get("include_integrity"))

    def test_role_unit_query_maps_to_command_history(self) -> None:
        tool, args = repair_route(
            "who is the SDPO of kuppam",
            "search_personnel",
            {},
        )
        self.assertEqual("get_unit_command_history", tool)
        self.assertEqual("kuppam sdpo", str(args.get("unit_name", "")).lower())

    def test_followup_district_repair_from_previous_rank(self) -> None:
        tool, args = repair_route(
            "which districts do they belong to",
            "search_personnel",
            {},
            last_user_query="list all SI in guntur district",
            last_assistant_response="",
        )
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertEqual("Sub-Inspector", args.get("rank_name"))
        self.assertEqual("Guntur", args.get("district_name"))


class ExtractorTests(unittest.TestCase):
    def test_normalize_common_query_typos(self) -> None:
        self.assertEqual(
            "list personnel in district",
            normalize_common_query_typos("list personell in disctrict"),
        )

    def test_extract_place_hints_multi_district(self) -> None:
        places = extract_place_hints("Find all SIs in Guntur and Chittoor districts")
        self.assertEqual(["Guntur", "Chittoor"], places)

    def test_extract_rank_hints(self) -> None:
        ranks = extract_rank_hints("list circle inspectors and constables")
        self.assertIn("Circle Inspector", ranks)
        self.assertIn("Constable", ranks)

    def test_extract_unit_hint(self) -> None:
        unit = extract_unit_hint("which villages are mapped to k v palli ps")
        self.assertIsNotNone(unit)
        self.assertIn("ps", unit.lower())

    def test_extract_person_hint_ignores_role_phrasing(self) -> None:
        person = extract_person_hint("who is the sdpo of kuppam")
        self.assertIsNone(person)

    def test_followup_district_query_positive(self) -> None:
        self.assertTrue(is_followup_district_query("which districts do they belong to"))

    def test_followup_district_query_negative_for_contact_request(self) -> None:
        self.assertFalse(
            is_followup_district_query(
                "find all circle inspectors in guntur and chittoor districts and show their contact details"
            )
        )


class ParseRouteResponseTests(unittest.TestCase):
    def test_parse_valid_response(self) -> None:
        parsed = _parse_route_response(
            '{"tool":"list_districts","arguments":{},"understood_query":"list districts","confidence":0.95}',
            "list districts",
        )
        self.assertIsNotNone(parsed)
        tool, args, understood, confidence = parsed
        self.assertEqual("list_districts", tool)
        self.assertEqual({}, args)
        self.assertEqual("list districts", understood)
        self.assertEqual(0.95, confidence)

    def test_parse_rejects_missing_required_arguments(self) -> None:
        parsed = _parse_route_response(
            '{"tool":"query_personnel_by_rank","arguments":{},"confidence":0.8}',
            "list SI",
        )
        self.assertIsNone(parsed)

    def test_parse_uses_query_as_understood_fallback(self) -> None:
        parsed = _parse_route_response(
            '{"tool":"list_districts","arguments":{},"confidence":0.8}',
            "which districts are available",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual("which districts are available", parsed[2])

    def test_parse_clamps_confidence(self) -> None:
        parsed = _parse_route_response(
            '{"tool":"list_districts","arguments":{},"understood_query":"x","confidence":5}',
            "x",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(1.0, parsed[3])


class LLMRouteQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_route_query_uses_strict_retry(self) -> None:
        with (
            patch("mcp.llm_router.call_claude_api", new=AsyncMock(side_effect=["not-json", '{"tool":"list_districts","arguments":{},"confidence":0.81}'])),
            patch("mcp.llm_router.call_openai_api", new=AsyncMock(return_value=None)),
        ):
            tool, args, understood, confidence, source = await llm_route_query("which districts are available")
        self.assertEqual("list_districts", tool)
        self.assertEqual({}, args)
        self.assertEqual("which districts are available", understood)
        self.assertEqual(0.81, confidence)
        self.assertEqual("llm_retry_no_context", source)

    async def test_llm_route_query_falls_back_to_heuristics(self) -> None:
        with (
            patch("mcp.llm_router.call_claude_api", new=AsyncMock(return_value=None)),
            patch("mcp.llm_router.call_openai_api", new=AsyncMock(return_value=None)),
            patch("mcp.llm_router.fallback_route_query", return_value=("list_districts", {}, "which districts are available", 0.5)),
        ):
            tool, args, understood, confidence, source = await llm_route_query("which districts are available")
        self.assertEqual("list_districts", tool)
        self.assertEqual({}, args)
        self.assertEqual("which districts are available", understood)
        self.assertEqual(0.5, confidence)
        self.assertEqual("heuristic_fallback", source)


class IntelligentQueryHandlerFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_query_uses_rank_flow_for_multi_district_contact_request(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            return_value={
                "success": True,
                "data": [
                    {
                        "name": "Officer One",
                        "userId": "00000001",
                        "districtName": "Guntur",
                        "mobile": "9000000001",
                        "email": "one@example.com",
                    }
                ],
                "pagination": {"page": 1, "page_size": 50, "total": 1, "total_pages": 1},
            }
        )

        query = "Find all Circle Inspectors in Guntur and Chittoor districts and show their contact details"
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=False),
            patch("mcp.llm_router.fallback_format_response", return_value="FORMATTED_RESULT") as fallback_fmt,
            patch("mcp.llm_router.format_followup_district_response", return_value="DISTRICT_ONLY_RESULT") as district_fmt,
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(query, context=None, session_id="s1")

        self.assertEqual("query_personnel_by_rank", result.get("routed_to"))
        self.assertEqual("FORMATTED_RESULT", result.get("response"))
        self.assertFalse(district_fmt.called, msg="Contact query should not be collapsed to district follow-up formatter")
        self.assertTrue(fallback_fmt.called)

        exec_call = fake_handler.execute.await_args
        self.assertEqual("query_personnel_by_rank", exec_call.args[0])
        executed_args = exec_call.args[1]
        self.assertEqual("Circle Inspector", executed_args.get("rank_name"))
        self.assertEqual(["Guntur", "Chittoor"], executed_args.get("district_names"))

    async def test_process_query_uses_followup_district_formatter_when_requested(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": [
                        {"name": "Officer One", "userId": "00000001", "districtName": "Guntur"},
                        {"name": "Officer Two", "userId": "00000002", "districtName": "Guntur"},
                    ],
                    "pagination": {"page": 1, "page_size": 50, "total": 2, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {"name": "Officer One", "userId": "00000001", "districtName": "Guntur"},
                        {"name": "Officer Two", "userId": "00000002", "districtName": "Guntur"},
                    ],
                    "pagination": {"page": 1, "page_size": 50, "total": 2, "total_pages": 1},
                },
            ]
        )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=False),
            patch("mcp.llm_router.fallback_format_response", return_value="RANK_LIST"),
            patch("mcp.llm_router.format_followup_district_response", return_value="DISTRICT_FOLLOWUP") as district_fmt,
        ):
            handler = IntelligentQueryHandler()
            await handler.process_query("list all SI in guntur district", context=None, session_id="s2")
            followup = await handler.process_query("which districts do they belong to", context=None, session_id="s2")

        self.assertEqual("DISTRICT_FOLLOWUP", followup.get("response"))
        self.assertTrue(district_fmt.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
