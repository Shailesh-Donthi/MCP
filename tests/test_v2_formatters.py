import unittest

from mcp.v2.utils.formatters import generate_natural_language_response


class V2FormatterTests(unittest.TestCase):
    def test_search_personnel_multi_match_returns_full_profile_lines(self) -> None:
        result = {
            "success": True,
            "data": [
                {
                    "name": "Chaitanya Kumar",
                    "userId": "12345678",
                    "badgeNo": "B-100",
                    "rank": {"name": "Sub Inspector", "shortCode": "SI"},
                    "department": "Law & Order Wing",
                    "isActive": True,
                    "gender": "Male",
                    "dateOfBirth": "1990-02-01T00:00:00",
                    "mobile": "9000000000",
                    "email": "ckumar@example.com",
                    "address": "Guntur",
                    "bloodGroup": "O+",
                    "fatherName": "Ramesh",
                    "dateOfJoining": "2015-01-01T00:00:00",
                    "dateOfRetirement": None,
                    "assignments": [
                        {
                            "unitName": "Guntur PS",
                            "districtName": "Guntur",
                            "designationName": "SHO",
                        }
                    ],
                },
                {
                    "name": "Chaitanya Rao",
                    "userId": "87654321",
                    "badgeNo": "B-101",
                    "rank": {"name": "Head Constable", "shortCode": "HC"},
                    "department": "Traffic",
                    "isActive": True,
                    "gender": "Male",
                    "dateOfBirth": "1992-06-15T00:00:00",
                    "mobile": "9000000001",
                    "email": "crao@example.com",
                    "address": "Vijayawada",
                    "bloodGroup": "A+",
                    "fatherName": "Suresh",
                    "dateOfJoining": "2016-04-01T00:00:00",
                    "dateOfRetirement": None,
                    "assignments": [
                        {
                            "unitName": "Vijayawada Traffic PS",
                            "districtName": "NTR",
                            "designationName": "HC",
                        }
                    ],
                },
            ],
            "pagination": {"total": 2, "page": 1, "page_size": 20, "total_pages": 1},
        }

        response = generate_natural_language_response(
            "find personnel with name Chaitanya",
            "search_personnel",
            {"name": "Chaitanya"},
            result,
        )

        self.assertIn("Found 2 matching personnel record(s) with full details:", response)
        self.assertIn("1. Chaitanya Kumar", response)
        self.assertIn("- User ID: 12345678", response)
        self.assertIn("- Active assignments:", response)
        self.assertIn("2. Chaitanya Rao", response)
        self.assertIn("- Mobile: 9000000001", response)

    def test_search_personnel_adds_pagination_note_when_more_records_exist(self) -> None:
        result = {
            "success": True,
            "data": [
                {
                    "name": "Chaitanya Kumar",
                    "userId": "12345678",
                    "rank": {"name": "Sub Inspector", "shortCode": "SI"},
                }
            ],
            "pagination": {"total": 3, "page": 1, "page_size": 1, "total_pages": 3},
        }

        response = generate_natural_language_response(
            "find personnel with name Chaitanya",
            "search_personnel",
            {"name": "Chaitanya"},
            result,
        )

        self.assertIn("Showing 1 record(s) on page 1/3. 2 more record(s) are available.", response)

    def test_search_personnel_department_dict_is_rendered_human_readable(self) -> None:
        result = {
            "success": True,
            "data": [
                {
                    "name": "K. Chaitanya",
                    "userId": "15000290",
                    "badgeNo": None,
                    "rank": {"name": "Police Constable", "shortCode": "PC"},
                    "department": {
                        "_id": "693914d3df665309d24dba3f",
                        "name": "Law & Order Wing",
                        "createdAt": "2025-12-10T12:06:03.053000",
                        "shortCode": "LO",
                    },
                    "isActive": True,
                    "gender": "Male",
                    "dateOfBirth": "1990-08-22T00:00:00",
                    "mobile": "+919273275091",
                    "email": "chaitanya.kanna1211@gmail.com",
                    "address": None,
                    "bloodGroup": None,
                    "fatherName": None,
                    "dateOfJoining": None,
                    "dateOfRetirement": None,
                }
            ],
            "pagination": {"total": 1, "page": 1, "page_size": 20, "total_pages": 1},
        }

        response = generate_natural_language_response(
            "search personnel with name chaitanya",
            "search_personnel",
            {"name": "chaitanya"},
            result,
        )

        self.assertIn("- Department: Law & Order Wing", response)
        self.assertNotIn("createdAt", response)
        self.assertNotIn("'name': 'Law & Order Wing'", response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
