import re
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
from bson import ObjectId
from pymongo import MongoClient


def _load_env_file(path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


class DBRelationshipIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = _load_env_file()
        uri = env.get("MONGODB_URI")
        db_name = env.get("MONGODB_DB_NAME")
        if not uri or not db_name:
            raise unittest.SkipTest("MONGODB_URI/MONGODB_DB_NAME not configured in .env")

        cls._mongo_client = MongoClient(uri, serverSelectionTimeoutMS=20000)
        cls.db = cls._mongo_client[db_name]
        cls.db.command("ping")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_mongo_client", None) is not None:
            cls._mongo_client.close()

    def test_required_collections_exist_and_non_empty(self) -> None:
        required_collections = [
            "personnel_master",
            "assignment_master",
            "unit_master",
            "district_master",
            "rank_master",
            "designation_master",
            "unit_type_master",
        ]
        existing = set(self.db.list_collection_names())
        for collection in required_collections:
            self.assertIn(collection, existing, msg=f"Missing collection: {collection}")
            count = self.db[collection].estimated_document_count()
            self.assertGreater(count, 0, msg=f"Collection {collection} is empty")

    def test_assignment_foreign_keys_integrity_rate(self) -> None:
        assignment_docs = list(
            self.db["assignment_master"].find(
                {"isDelete": False},
                {"_id": 1, "userId": 1, "unitId": 1, "designationId": 1},
            )
        )
        total = len(assignment_docs)
        self.assertGreater(total, 0, "No active assignment records found")

        unit_ids = set(self.db["unit_master"].distinct("_id", {"isDelete": False}))
        personnel_ids = set(self.db["personnel_master"].distinct("_id", {"isDelete": False}))
        designation_ids = set(self.db["designation_master"].distinct("_id", {"isDelete": False}))

        missing_user = 0
        missing_unit = 0
        missing_designation = 0
        for row in assignment_docs:
            if row.get("userId") not in personnel_ids:
                missing_user += 1
            if row.get("unitId") not in unit_ids:
                missing_unit += 1
            if row.get("designationId") is not None and row.get("designationId") not in designation_ids:
                missing_designation += 1

        # Keep strict on unit/designation references; allow a small residual on user links.
        self.assertEqual(0, missing_unit, "Found assignments with missing unit references")
        self.assertEqual(0, missing_designation, "Found assignments with missing designation references")
        self.assertLessEqual(
            missing_user / total,
            0.02,
            msg=f"Assignment -> personnel orphan rate too high: {missing_user}/{total}",
        )

    def test_unit_foreign_keys_integrity_rate(self) -> None:
        unit_docs = list(
            self.db["unit_master"].find(
                {"isDelete": False},
                {"_id": 1, "districtId": 1, "unitTypeId": 1, "parentUnitId": 1, "responsibleUserId": 1},
            )
        )
        total = len(unit_docs)
        self.assertGreater(total, 0, "No active unit records found")

        district_ids = set(self.db["district_master"].distinct("_id", {"isDelete": False}))
        unit_type_ids = set(self.db["unit_type_master"].distinct("_id", {"isDelete": False}))
        unit_ids = {row["_id"] for row in unit_docs if isinstance(row.get("_id"), ObjectId)}
        personnel_ids = set(self.db["personnel_master"].distinct("_id", {"isDelete": False}))

        missing_district = 0
        missing_unit_type = 0
        bad_parent = 0
        bad_responsible_user = 0

        for row in unit_docs:
            if row.get("districtId") not in district_ids:
                missing_district += 1
            if row.get("unitTypeId") is not None and row.get("unitTypeId") not in unit_type_ids:
                missing_unit_type += 1
            if row.get("parentUnitId") is not None and row.get("parentUnitId") not in unit_ids:
                bad_parent += 1
            if row.get("responsibleUserId") is not None and row.get("responsibleUserId") not in personnel_ids:
                bad_responsible_user += 1

        self.assertEqual(0, missing_unit_type, "Found units with missing unit_type references")
        self.assertEqual(0, bad_parent, "Found units with missing parent unit references")
        self.assertEqual(0, bad_responsible_user, "Found units with missing responsible-user references")
        self.assertLessEqual(
            missing_district / total,
            0.02,
            msg=f"Unit -> district orphan rate too high: {missing_district}/{total}",
        )

    def test_personnel_rank_and_designation_links_are_consistent(self) -> None:
        personnel_docs = list(
            self.db["personnel_master"].find(
                {"isDelete": False},
                {"_id": 1, "rankId": 1, "designationId": 1},
            )
        )
        self.assertGreater(len(personnel_docs), 0, "No active personnel records found")

        rank_ids = set(self.db["rank_master"].distinct("_id", {"isDelete": False}))
        designation_ids = set(self.db["designation_master"].distinct("_id", {"isDelete": False}))

        bad_rank = 0
        bad_designation = 0
        for row in personnel_docs:
            if row.get("rankId") is not None and row.get("rankId") not in rank_ids:
                bad_rank += 1
            if row.get("designationId") is not None and row.get("designationId") not in designation_ids:
                bad_designation += 1

        self.assertEqual(0, bad_rank, f"Found personnel with invalid rank links: {bad_rank}")
        self.assertEqual(0, bad_designation, f"Found personnel with invalid designation links: {bad_designation}")


class QueryMatrixRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = _load_env_file()
        uri = env.get("MONGODB_URI")
        db_name = env.get("MONGODB_DB_NAME")
        if not uri or not db_name:
            raise unittest.SkipTest("MONGODB_URI/MONGODB_DB_NAME not configured in .env")

        cls._mongo_client = MongoClient(uri, serverSelectionTimeoutMS=20000)
        cls.db = cls._mongo_client[db_name]
        cls.db.command("ping")

        cls.base_url = "http://127.0.0.1:8090/api/v1/query"
        cls.http = httpx.Client(timeout=60.0)
        try:
            health = cls.http.get("http://127.0.0.1:8090/health")
            if health.status_code != 200:
                raise unittest.SkipTest(f"API health endpoint returned {health.status_code}")
        except Exception as exc:
            raise unittest.SkipTest(f"API not reachable on 127.0.0.1:8090: {exc}") from exc

        cls.district_names = [
            row.get("name")
            for row in cls.db["district_master"].find({"isDelete": False}, {"name": 1}).sort("name", 1).limit(5)
            if isinstance(row.get("name"), str) and row.get("name").strip()
        ]
        cls.unit_names = [
            row.get("name")
            for row in cls.db["unit_master"].find({"isDelete": False}, {"name": 1}).sort("name", 1).limit(8)
            if isinstance(row.get("name"), str) and row.get("name").strip()
        ]
        cls.rank_names = [
            row.get("name")
            for row in cls.db["rank_master"].find({"isDelete": False}, {"name": 1}).sort("name", 1).limit(5)
            if isinstance(row.get("name"), str) and row.get("name").strip()
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "http", None) is not None:
            cls.http.close()
        if getattr(cls, "_mongo_client", None) is not None:
            cls._mongo_client.close()

    def _query(self, query: str, session_id: str, chat_id: str) -> Dict:
        response = self.http.post(
            self.base_url,
            json={"query": query, "session_id": session_id, "chat_id": chat_id},
        )
        self.assertEqual(200, response.status_code, msg=f"HTTP failed for query: {query}")
        payload = response.json()
        self.assertIsInstance(payload, dict, msg=f"Non-dict API payload for query: {query}")
        self.assertTrue(payload.get("success", False), msg=f"Query failed: {query} -> {payload.get('response')}")
        rendered = str(payload.get("response") or "")
        self.assertNotIn("Error code:", rendered, msg=f"Tool error response for query: {query}")
        return payload

    def test_core_queries(self) -> None:
        queries = [
            "which districts are available",
            "how many ranges in AP",
            "show me units in guntur range",
            "show recent transfers for 30 days",
            "find missing village mappings in guntur district",
            "who was the sho of guntur ps in last 15 days",
            "where is Addl DGP LO",
        ]
        for q in queries:
            with self.subTest(query=q):
                self._query(q, "matrix_core_session", "matrix_core_chat")

    def test_district_rank_unit_matrix_queries(self) -> None:
        generated: List[str] = []
        for district in self.district_names:
            generated.extend(
                [
                    f"list all units in {district} district",
                    f"how many personnel are in {district} district",
                    f"list all SI in {district} district",
                    f"show vacancies in {district} district",
                ]
            )
        for unit in self.unit_names:
            generated.extend(
                [
                    f"where is {unit}",
                    f"who is in charge of {unit}",
                    f"personnel in {unit}",
                ]
            )
        for rank in self.rank_names:
            generated.extend(
                [
                    f"list all {rank} in guntur district",
                    f"how many {rank} are in guntur district",
                ]
            )

        seen = set()
        deduped: List[str] = []
        for q in generated:
            key = q.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)

        for q in deduped:
            with self.subTest(query=q):
                self._query(q, "matrix_bulk_session", "matrix_bulk_chat")

    def test_followup_si_attachment_response_contains_units(self) -> None:
        first = self._query(
            "list all SI in guntur district",
            "si_attachment_session",
            "si_attachment_chat",
        )
        self.assertIn("Sub-Inspector", str(first.get("response", "")))

        followup = self._query(
            "what unit are these SIs attached to",
            "si_attachment_session",
            "si_attachment_chat",
        )
        rendered = str(followup.get("response") or "")
        # Validate that list entries include " - <unit>" for at least one line.
        self.assertRegex(
            rendered,
            r"\n\s*\d+\.\s+.+\s-\s+.+",
            msg=f"Follow-up SI attachment response missing unit mapping:\n{rendered[:800]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

