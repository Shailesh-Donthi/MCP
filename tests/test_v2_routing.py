import asyncio
import unittest

from mcp.llm_router import llm_route_query
from mcp.orchestration.enriched_router import route_query_to_tool_enriched
from mcp.server_http import route_query_to_tool as route_query_to_tool_v2


class V2RoutingTests(unittest.TestCase):
    def assert_route(self, query: str, tool: str, subset: dict | None = None) -> None:
        routed_tool, args = route_query_to_tool_enriched(query)
        self.assertEqual(tool, routed_tool, msg=f"Unexpected tool for query: {query}")
        if subset:
            for key, value in subset.items():
                self.assertIn(key, args, msg=f"Missing argument '{key}' for query: {query}")
                self.assertEqual(value, args[key], msg=f"Unexpected value for '{key}' on query: {query}")

    def test_sp_in_district_routes_to_search_personnel(self) -> None:
        self.assert_route(
            "SP in Guntur",
            "search_personnel",
            {"designation_name": "Superintendent of Police", "district_name": "Guntur"},
        )

    def test_in_charge_of_unit_routes_to_search_unit(self) -> None:
        tool, args = route_query_to_tool_enriched("who is in charge of Guntur PS")
        self.assertEqual("get_unit_command_history", tool)
        self.assertIn("unit_name", args)
        self.assertIn("Guntur", str(args.get("unit_name")))

    def test_sho_query_with_time_window_cleans_unit_name(self) -> None:
        tool, args = route_query_to_tool_v2("who was the sho of guntur ps in last 15 days")
        self.assertEqual("get_unit_command_history", tool)
        self.assertEqual("guntur ps", str(args.get("unit_name", "")).lower())

    def test_where_is_generic_target_routes_to_search_unit(self) -> None:
        tool, args = route_query_to_tool_v2("where is Addl DGP LO")
        self.assertEqual("search_unit", tool)
        self.assertEqual("Addl DGP LO", args.get("name"))

    def test_vacancy_tool_alias_normalized_in_v2(self) -> None:
        tool, args = route_query_to_tool_v2("show vacancies in annamayya district", hint="count_vacancies_by_unit")
        self.assertEqual("count_vacancies_by_unit_rank", tool)
        self.assertEqual("Annamayya", args.get("district_name"))

    def test_followup_si_attachment_prefers_rank_route(self) -> None:
        tool, args, _, _, source = asyncio.run(
            llm_route_query(
                "what unit are these SIs attached to",
                conversation_context=[
                    {"role": "assistant", "content": "Sub-Inspector Personnel in Guntur district:\n\n1. A Person"}
                ],
                available_tools={"query_personnel_by_rank", "search_assignment"},
            )
        )
        self.assertEqual("query_personnel_by_rank", tool)
        self.assertEqual("Sub-Inspector", args.get("rank_name"))
        self.assertEqual("Guntur", args.get("district_name"))
        self.assertEqual("heuristic_followup", source)

    def test_assignment_query_routes_to_search_assignment(self) -> None:
        self.assert_route(
            "assignments for user id 14402876",
            "search_assignment",
            {"user_id": "14402876"},
        )

    def test_responsible_user_check_routes_correctly(self) -> None:
        self.assert_route(
            "is Ravi Kumar a responsible user",
            "check_responsible_user",
            {"name": "Ravi Kumar"},
        )

    def test_unknown_query_returns_help(self) -> None:
        tool, args = route_query_to_tool_enriched("???")
        self.assertEqual("__help__", tool)
        self.assertIsInstance(args, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
