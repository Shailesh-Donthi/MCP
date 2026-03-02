import unittest
from unittest.mock import AsyncMock, patch

from mcp.llm_router import IntelligentQueryHandler
from mcp.utils.formatters import generate_natural_language_response


class ComplexQueryExecutionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _result(
        data,
        *,
        page: int = 1,
        page_size: int = 50,
        total: int = 0,
        total_pages: int = 1,
    ):
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
            },
        }

    def _build_fake_handler(self):
        fake_handler = type("FakeHandler", (), {})()

        async def _execute(tool_name, arguments, context=None):
            args = arguments or {}

            if tool_name == "query_personnel_by_rank":
                page = int(args.get("page", 1) or 1)
                if page <= 1:
                    data = [
                        {"name": "Officer One", "userId": "00000001", "districtName": "Guntur"},
                        {"name": "Officer Two", "userId": "00000002", "districtName": "Guntur"},
                    ]
                else:
                    data = [
                        {"name": "Officer Three", "userId": "00000003", "districtName": "Chittoor"},
                        {"name": "Officer Four", "userId": "00000004", "districtName": "Chittoor"},
                    ]
                return self._result(data, page=page, page_size=2, total=4, total_pages=2)

            if tool_name == "search_personnel":
                user_id = str(args.get("user_id") or "00000002")
                district = "Guntur" if user_id in {"00000001", "00000002"} else "Chittoor"
                data = [
                    {
                        "name": f"Officer {user_id}",
                        "userId": user_id,
                        "districtName": district,
                        "mobile": "9000000000",
                        "email": f"{user_id}@example.com",
                    }
                ]
                return self._result(data, page=1, page_size=1, total=1, total_pages=1)

            if tool_name == "query_linked_master_data":
                data = [
                    {
                        "module": "Investigation",
                        "notifications": ["CaseAssigned", "CaseClosed"],
                        "reverse_links": ["approval_flow_master"],
                    }
                ]
                return self._result(data, page=1, page_size=50, total=1, total_pages=1)

            return self._result([], page=1, page_size=50, total=0, total_pages=1)

        fake_handler.execute = AsyncMock(side_effect=_execute)
        return fake_handler

    async def test_chain_query_with_pagination_and_ordinal_detail_lookup(self):
        fake_handler = self._build_fake_handler()
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=False),
            patch("mcp.llm_router.fallback_format_response", side_effect=lambda q, t, a, r: f"{t}:{a}"),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "list all SI in guntur district and then next page and then show details of the 2nd one",
                context=None,
                session_id="complex-chain",
            )

        self.assertTrue(result.get("success"))
        self.assertEqual("complex_query_chain", result.get("understood_as"))
        steps = result.get("data", {}).get("steps", [])
        self.assertEqual(3, len(steps))
        self.assertEqual("query_personnel_by_rank", steps[0].get("routed_to"))
        self.assertEqual(2, steps[1].get("arguments", {}).get("page"))
        self.assertEqual("search_personnel", steps[2].get("routed_to"))
        self.assertEqual("00000004", steps[2].get("arguments", {}).get("user_id"))

    async def test_followup_district_query_after_rank_list_uses_district_formatter(self):
        fake_handler = self._build_fake_handler()
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=False),
            patch("mcp.llm_router.fallback_format_response", side_effect=lambda q, t, a, r: f"{t}:{a}"),
            patch("mcp.llm_router.format_followup_district_response", return_value="DISTRICT_FOLLOWUP") as district_fmt,
        ):
            handler = IntelligentQueryHandler()
            await handler.process_query("list all SI in guntur district", context=None, session_id="district-followup")
            followup = await handler.process_query(
                "which districts do they belong to",
                context=None,
                session_id="district-followup",
            )

        self.assertEqual("query_personnel_by_rank", followup.get("routed_to"))
        self.assertEqual("DISTRICT_FOLLOWUP", followup.get("response"))
        self.assertTrue(district_fmt.called)

    async def test_attribute_only_followup_reuses_last_person(self):
        fake_handler = self._build_fake_handler()
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=False),
            patch("mcp.llm_router.fallback_format_response", side_effect=lambda q, t, a, r: f"{t}:{a}"),
        ):
            handler = IntelligentQueryHandler()
            first = await handler.process_query("who has user id 00000002", context=None, session_id="person-followup")
            second = await handler.process_query("mobile number?", context=None, session_id="person-followup")

        self.assertEqual("search_personnel", first.get("routed_to"))
        self.assertEqual("search_personnel", second.get("routed_to"))
        self.assertEqual("00000002", second.get("arguments", {}).get("user_id"))

    async def test_llm_routed_relationship_query_avoids_clarification(self):
        fake_handler = self._build_fake_handler()
        llm_route = (
            "query_linked_master_data",
            {
                "collection": "modules_master",
                "mode": "discover",
                "include_related": True,
                "include_reverse": True,
                "include_integrity": True,
            },
            "discover relationships between modules and notifications",
            0.94,
            "llm",
        )
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch("mcp.llm_router.llm_route_query", new=AsyncMock(return_value=llm_route)),
            patch("mcp.llm_router.fallback_format_response", side_effect=lambda q, t, a, r: f"{t}:{a}"),
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query(
                "how are modules and notifications interlinked",
                context=None,
                session_id="llm-master",
            )

        self.assertTrue(result.get("success"))
        self.assertEqual("query_linked_master_data", result.get("routed_to"))
        self.assertEqual("discover", result.get("arguments", {}).get("mode"))
        self.assertEqual("llm", result.get("route_source"))

    async def test_llm_enabled_person_lookup_uses_deterministic_formatter(self):
        fake_handler = self._build_fake_handler()
        with (
            patch("mcp.llm_router.get_tool_handler", return_value=fake_handler),
            patch("mcp.llm_router.has_llm_api_key", return_value=True),
            patch(
                "mcp.llm_router.llm_route_query",
                new=AsyncMock(
                    return_value=(
                        "search_personnel",
                        {"name": "Officer 00000002"},
                        "who is officer 00000002",
                        0.95,
                        "llm",
                    )
                ),
            ),
            patch("mcp.llm_router.llm_format_response", new=AsyncMock(return_value="LLM_FORMATTED")) as llm_fmt,
            patch("mcp.llm_router.fallback_format_response", return_value="FALLBACK_FORMATTED") as fallback_fmt,
        ):
            handler = IntelligentQueryHandler()
            result = await handler.process_query("who is officer 00000002", context=None, session_id="det-person")

        self.assertEqual("search_personnel", result.get("routed_to"))
        self.assertEqual("FALLBACK_FORMATTED", result.get("response"))
        self.assertTrue(fallback_fmt.called)
        self.assertFalse(llm_fmt.called)


class PersonnelFormatterTests(unittest.TestCase):
    def test_person_profile_includes_assignment_details(self):
        result = {
            "success": True,
            "data": [
                {
                    "name": "Chitikina Murali Krishna",
                    "userId": "14402876",
                    "badgeNo": None,
                    "rank": {"name": "Deputy Superintendent of Police", "shortCode": "DSP"},
                    "department": "Law & Order Wing",
                    "isActive": True,
                    "gender": "Male",
                    "dateOfBirth": "1966-01-18T00:00:00",
                    "mobile": "8096024000",
                    "email": "chmurali.dsp@gmail.com",
                    "address": None,
                    "bloodGroup": None,
                    "fatherName": None,
                    "dateOfJoining": None,
                    "dateOfRetirement": None,
                    "primary_unit": "Not assigned",
                    "assignments": [
                        {
                            "unitId": "u1",
                            "unitName": "North Sub Division",
                            "districtName": "Guntur",
                            "designationName": "DSP",
                        }
                    ],
                }
            ],
            "pagination": {"page": 1, "page_size": 1, "total": 1, "total_pages": 1},
            "metadata": {},
        }

        response = generate_natural_language_response(
            "who is chitikina murali krishna",
            "search_personnel",
            {"name": "Chitikina Murali Krishna"},
            result,
        )

        self.assertIn("full profile", response.lower())
        self.assertIn("North Sub Division", response)
        self.assertIn("Guntur", response)
        self.assertIn("Active assignments", response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
