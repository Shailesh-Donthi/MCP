import unittest

from mcp.handlers.tool_handler import TOOL_CLASSES as V1_TOOL_CLASSES
from mcp.router.routing_rules import repair_route
from mcp.v1.server_http import route_query_to_tool as route_v1
from mcp.v2.handlers.tool_handler import TOOL_CLASSES as V2_TOOL_CLASSES
from mcp.v2.server_http import route_query_to_tool as route_v2


def _tool_names(classes):
    names = []
    for cls in classes:
        value = getattr(cls, "name", None)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    return set(names)


class V2ParityTests(unittest.TestCase):
    def test_v2_tool_registry_covers_v1(self):
        v1_tools = _tool_names(V1_TOOL_CLASSES)
        v2_tools = _tool_names(V2_TOOL_CLASSES)
        missing = sorted(v1_tools - v2_tools)
        self.assertEqual([], missing, msg=f"V2 is missing V1 tools: {missing}")
        self.assertIn("search_assignment", v2_tools, msg="V2-specific search_assignment tool is missing")

    def test_v2_route_matches_v1_for_core_queries(self):
        # Representative cross-domain queries from V1 behavior.
        queries = [
            "which districts are available",
            "show notification master entries linked to modules",
            "find missing village mappings in chittoor district",
            "which villages are mapped to k v palli ps",
            "show hierarchy of chittoor district",
            "show recent transfers for 15 days",
            "who was the sho of guntur ps in last 15 days",
            "who is the sdpo of kuppam",
            "list all units in guntur district",
            "how many personnel are in guntur district",
            "show vacancies in guntur district",
            "list all SI in guntur district",
            "where is kuppam sdpo",
            "tell me about person Ravi Kumar",
        ]
        for query in queries:
            with self.subTest(query=query):
                v1_tool, v1_args = route_v1(query)
                v1_tool, v1_args = repair_route(query, v1_tool, v1_args)
                v2_tool, v2_args = route_v2(query)
                self.assertEqual(v1_tool, v2_tool, msg=f"Tool mismatch for query: {query}")
                # Spot-check a few key arguments where V1 provides them.
                for key in ("district_name", "unit_name", "rank_name", "days", "collection"):
                    if key in v1_args:
                        self.assertIn(key, v2_args, msg=f"Missing arg '{key}' for query: {query}")
                        self.assertEqual(v1_args[key], v2_args[key], msg=f"Arg '{key}' mismatch for query: {query}")

    def test_v2_supports_assignment_intent(self):
        tool, args = route_v2("assignments for user id 14402876")
        self.assertEqual("search_assignment", tool)
        self.assertEqual("14402876", args.get("user_id"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
