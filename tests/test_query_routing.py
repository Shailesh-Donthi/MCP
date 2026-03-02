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
from mcp.router.routing_rules import needs_clarification, repair_route
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

    def test_sp_of_dpo_routes_to_command_history(self) -> None:
        tool, args = repair_route(
            "SP of Guntur DPO",
            "query_personnel_by_rank",
            {"rank_name": "Superintendent of Police", "district_name": "Guntur"},
        )
        self.assertEqual("get_unit_command_history", tool)
        self.assertEqual("Guntur DPO", args.get("unit_name"))

    def test_sp_of_gpo_typo_routes_to_command_history(self) -> None:
        tool, args = repair_route(
            "SP of Guntur GPO",
            "query_personnel_by_rank",
            {"rank_name": "Superintendent of Police", "district_name": "Guntur"},
        )
        self.assertEqual("get_unit_command_history", tool)
        self.assertEqual("Guntur DPO", args.get("unit_name"))

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

    def test_designation_query_maps_to_search_personnel_designation(self) -> None:
        tool, args = repair_route(
            "who has the designation of SPDO",
            "query_personnel_by_rank",
            {"rank_name": "SDPO"},
        )
        self.assertEqual("search_personnel", tool)
        self.assertEqual("SDPO", args.get("designation_name"))

    def test_plural_spdos_query_maps_to_search_personnel_designation(self) -> None:
        tool, args = repair_route(
            "list all SPDOs",
            "query_personnel_by_rank",
            {"rank_name": "Superintendent of Police"},
        )
        self.assertEqual("search_personnel", tool)
        self.assertEqual("SDPO", args.get("designation_name"))

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

    def test_extract_place_hints_hierarchy_query_strips_leading_get(self) -> None:
        places = extract_place_hints("get unit hierarchy for guntur district")
        self.assertEqual(["Guntur"], places)

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


class ClarificationRuleTests(unittest.TestCase):
    def test_rank_and_above_query_not_marked_for_clarification(self) -> None:
        self.assertFalse(needs_clarification("show SI and above in guntur"))


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

    async def test_process_query_does_not_leak_selected_person_into_explicit_rank_place_query(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": [
                        {"name": "User 00001437", "userId": "00001437", "rank": {"name": "Sub Inspector"}},
                    ],
                    "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {"name": "User 00000738", "userId": "00000738", "rankName": "Circle Inspector", "districtName": "Guntur"},
                    ],
                    "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
                },
            ]
        )

        async def _fake_route(query: str, conversation_context=None):
            if "who has user id 00001437" in query.lower():
                return "search_personnel", {"user_id": "00001437"}, query, 0.99, "llm"
            return (
                "query_personnel_by_rank",
                {
                    "rank_name": "Circle Inspector",
                    "district_names": ["Guntur", "Chittoor"],
                    "page_size": 200,
                },
                query,
                0.95,
                "llm",
            )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch("mcp.llm_router.llm_route_query", new=AsyncMock(side_effect=_fake_route)),
            patch("mcp.llm_router.fallback_format_response", return_value="OK"),
        ):
            handler = IntelligentQueryHandler()
            await handler.process_query("who has user id 00001437", context=None, session_id="s3")
            await handler.process_query(
                "list all Circle Inspectors in guntur and chittoor districts and show their contact details",
                context=None,
                session_id="s3",
            )

        second_call = fake_handler.execute.await_args_list[1]
        self.assertEqual("query_personnel_by_rank", second_call.args[0])
        second_args = second_call.args[1]
        self.assertNotIn("user_id", second_args)
        self.assertEqual("Circle Inspector", second_args.get("rank_name"))

    async def test_process_query_recovers_designation_prompt_from_rank_not_found(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": False,
                    "error": {"code": "NOT_FOUND", "message": "Rank not found: SDPO"},
                },
                {
                    "success": True,
                    "data": [],
                    "pagination": {"page": 1, "page_size": 50, "total": 0, "total_pages": 1},
                },
            ]
        )

        async def _fake_route(query: str, conversation_context=None):
            return (
                "query_personnel_by_rank",
                {"rank_name": "SDPO", "rank_relation": "exact"},
                query,
                0.92,
                "llm",
            )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch("mcp.llm_router.llm_route_query", new=AsyncMock(side_effect=_fake_route)),
            patch("mcp.llm_router.llm_format_response", new=AsyncMock(return_value="NO_MATCH_RESULT")),
            patch("mcp.llm_router.fallback_format_response", return_value="NO_MATCH_RESULT"),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "who has the designation of SPDO",
                context=None,
                session_id="s4",
            )

        self.assertEqual("search_personnel", result.get("routed_to"))
        self.assertEqual("SDPO", result.get("arguments", {}).get("designation_name"))
        self.assertEqual("NO_MATCH_RESULT", result.get("response"))

    async def test_process_query_recovers_sp_of_unit_when_command_data_missing(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": {
                        "unitId": "u1",
                        "unitName": "Guntur DPO",
                        "currentResponsibleUser": None,
                        "history": [],
                    },
                    "metadata": {
                        "command_data_missing": True,
                        "resolved_unit_name": "Guntur DPO",
                    },
                    "pagination": {"page": 1, "page_size": 20, "total": 0, "total_pages": 0},
                },
                {
                    "success": True,
                    "data": [
                        {
                            "_id": "p1",
                            "name": "Vakul Jindal",
                            "userId": "14402876",
                            "rankName": "Superintendent of Police",
                            "unitName": "Guntur DPO",
                            "districtName": "Guntur",
                        },
                        {
                            "_id": "p2",
                            "name": "Other SP",
                            "userId": "10000001",
                            "rankName": "Superintendent of Police",
                            "unitName": "Technical Services",
                            "districtName": "Guntur",
                        },
                    ],
                    "pagination": {"page": 1, "page_size": 200, "total": 2, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {
                            "_id": "p1",
                            "name": "Vakul Jindal",
                            "userId": "14402876",
                            "rankName": "Superintendent of Police",
                        }
                    ],
                    "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
                },
            ]
        )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch(
                "mcp.llm_router.llm_route_query",
                new=AsyncMock(
                    return_value=(
                        "get_unit_command_history",
                        {"unit_name": "Guntur DPO"},
                        "SP of Guntur DPO",
                        0.98,
                        "llm",
                    )
                ),
            ),
            patch(
                "mcp.llm_router.llm_format_response",
                new=AsyncMock(return_value="LLM_SHOULD_NOT_BE_USED"),
            ) as llm_fmt,
            patch(
                "mcp.llm_router.fallback_format_response",
                return_value="Vakul Jindal is the Superintendent of Police for Guntur DPO.",
            ),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query("SP of Guntur DPO", context=None, session_id="s5")

        self.assertEqual("search_personnel", result.get("routed_to"))
        self.assertEqual("14402876", result.get("arguments", {}).get("user_id"))
        self.assertEqual("Vakul Jindal is the Superintendent of Police for Guntur DPO.", result.get("response"))
        self.assertFalse(llm_fmt.called)

    async def test_process_query_resolves_role_of_unit_via_unit_personnel_filter(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": [
                        {"name": "Unrelated", "userId": "00000001", "rankName": "Administrative Officer"},
                    ],
                    "pagination": {"page": 1, "page_size": 20, "total": 1, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {"name": "Ambati Prabhudas", "userId": "14388807", "rankName": "Office Superintendent", "rankShortCode": "OS"},
                        {"name": "Tirumalasetty Ranga Rao", "userId": "14408286", "rankName": "Administrative Officer", "rankShortCode": "AO"},
                    ],
                    "pagination": {"page": 1, "page_size": 500, "total": 2, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {
                            "_id": "p_ao",
                            "name": "Tirumalasetty Ranga Rao",
                            "userId": "14408286",
                            "rankName": "Administrative Officer",
                        }
                    ],
                    "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
                },
            ]
        )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch(
                "mcp.llm_router.llm_route_query",
                new=AsyncMock(
                    return_value=(
                        "query_personnel_by_rank",
                        {"rank_name": "Administrative Officer", "district_name": "Guntur"},
                        "who is the AO of Sixth Btn APSP Mangalagiri?",
                        0.93,
                        "llm",
                    )
                ),
            ),
            patch(
                "mcp.llm_router.llm_format_response",
                new=AsyncMock(return_value="LLM_SHOULD_NOT_BE_USED"),
            ) as llm_fmt,
            patch(
                "mcp.llm_router.fallback_format_response",
                return_value="AO for Sixth Btn APSP Mangalagiri is Tirumalasetty Ranga Rao.",
            ),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "who is the AO of Sixth Btn APSP Mangalagiri?",
                context=None,
                session_id="s6",
            )

        self.assertEqual("search_personnel", result.get("routed_to"))
        self.assertEqual("14408286", result.get("arguments", {}).get("user_id"))
        self.assertEqual("AO for Sixth Btn APSP Mangalagiri is Tirumalasetty Ranga Rao.", result.get("response"))
        self.assertFalse(llm_fmt.called)

    async def test_process_query_resolves_igp_of_unit_via_unit_personnel_filter(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": [
                        {"name": "Admin", "userId": "10000001", "rankName": "Sub Inspector", "rankShortCode": "SI"},
                        {"name": "B. Raja Kumari", "userId": "9440796479", "rankName": "Inspector General of Police", "rankShortCode": "IGP"},
                    ],
                    "pagination": {"page": 1, "page_size": 500, "total": 2, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {
                            "_id": "p_igp",
                            "name": "B. Raja Kumari",
                            "userId": "9440796479",
                            "rankName": "Inspector General of Police",
                        }
                    ],
                    "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
                },
            ]
        )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch(
                "mcp.llm_router.llm_route_query",
                new=AsyncMock(
                    return_value=(
                        "get_unit_command_history",
                        {"unit_name": "APSP Head Office"},
                        "who is the IGP of APSP Head Office?",
                        0.96,
                        "llm",
                    )
                ),
            ),
            patch(
                "mcp.llm_router.llm_format_response",
                new=AsyncMock(return_value="LLM_SHOULD_NOT_BE_USED"),
            ) as llm_fmt,
            patch(
                "mcp.llm_router.fallback_format_response",
                return_value="IGP for APSP Head Office is B. Raja Kumari.",
            ),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "who is the IGP of APSP Head Office?",
                context=None,
                session_id="s7",
            )

        self.assertEqual("search_personnel", result.get("routed_to"))
        self.assertEqual("9440796479", result.get("arguments", {}).get("user_id"))
        self.assertEqual("IGP for APSP Head Office is B. Raja Kumari.", result.get("response"))
        self.assertFalse(llm_fmt.called)

    async def test_process_query_returns_candidate_list_for_ambiguous_role_in_unit(self) -> None:
        fake_handler = type("FakeHandler", (), {})()
        fake_handler.execute = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "data": [
                        {"name": "Any PC", "userId": "10000001", "rankName": "Police Constable", "districtName": "Guntur"},
                    ],
                    "pagination": {"page": 1, "page_size": 20, "total": 1, "total_pages": 1},
                },
                {
                    "success": True,
                    "data": [
                        {"name": "Abdul Fareed", "userId": "14166170", "rankName": "Police Constable", "rankShortCode": "PC"},
                        {"name": "Kasukurthi Vijayakuamr", "userId": "14164121", "rankName": "Police Constable", "rankShortCode": "PC"},
                        {"name": "Suraboyina Venkata Ravi", "userId": "14474547", "rankName": "Sub Inspector", "rankShortCode": "SI"},
                    ],
                    "pagination": {"page": 1, "page_size": 500, "total": 3, "total_pages": 1},
                },
            ]
        )

        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch(
                "mcp.llm_router.llm_route_query",
                new=AsyncMock(
                    return_value=(
                        "query_personnel_by_rank",
                        {"rank_name": "Police Constable", "district_name": "Guntur"},
                        "who is the PC of Duggirala PS?",
                        0.9,
                        "llm",
                    )
                ),
            ),
            patch(
                "mcp.llm_router.llm_format_response",
                new=AsyncMock(return_value="LLM_SHOULD_NOT_BE_USED"),
            ) as llm_fmt,
            patch(
                "mcp.llm_router.fallback_format_response",
                return_value="MULTIPLE_MATCHES",
            ),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "who is the PC of Duggirala PS?",
                context=None,
                session_id="s8",
            )

        self.assertEqual("query_personnel_by_unit", result.get("routed_to"))
        self.assertEqual("Duggirala PS", result.get("arguments", {}).get("unit_name"))
        self.assertEqual("MULTIPLE_MATCHES", result.get("response"))
        data_payload = result.get("data", {})
        self.assertTrue(isinstance(data_payload, dict))
        self.assertTrue(data_payload.get("success"))
        rows = data_payload.get("data") if isinstance(data_payload, dict) else None
        self.assertTrue(isinstance(rows, list))
        self.assertEqual(2, len(rows))
        self.assertFalse(llm_fmt.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
