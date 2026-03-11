"""
Complex query integration tests for the hybrid query assistant.
Tests multi-collection joins, aggregations, comparative queries,
and edge cases that stress both pre-built tools and the dynamic orchestrator.

Run with:
    python tests/test_complex_queries_live.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://127.0.0.1:8090"
TIMEOUT = 120
INTER_TEST_DELAY_S = 3


@dataclass
class TestCase:
    name: str
    query: str
    expect_any: List[str] = field(default_factory=list)
    expect_none: List[str] = field(default_factory=list)
    expect_nonempty: bool = True
    expect_success: bool = True
    # Optional: require the response to contain at least N of the expect_any tokens
    min_matches: int = 1


TEST_CASES: List[TestCase] = [

    # =========================================================================
    # 1. Multi-collection join queries (personnel + rank + unit + district)
    # =========================================================================
    TestCase(
        name="ci_in_guntur",
        query="List all Circle Inspectors posted in Guntur district with their names",
        expect_any=["Circle Inspector", "CI", "Guntur", "Brahmam", "Ashok", "Srinivas"],
        expect_none=["500"],
    ),
    TestCase(
        name="dsp_with_units",
        query="Show all DSPs with their current posting units and districts",
        expect_any=["DSP", "Deputy Superintendent", "unit", "Unit", "district", "District"],
        expect_none=["500"],
    ),
    TestCase(
        name="hc_count_per_district",
        query="How many Head Constables are there in each district?",
        expect_any=["Head Constable", "HC", "Guntur", "Chittoor"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 2. Comparative / analytical queries
    # =========================================================================
    TestCase(
        name="district_comparison",
        query="Compare the number of officers in Guntur vs Chittoor district",
        expect_any=["Guntur", "Chittoor"],
        expect_none=["500"],
    ),
    TestCase(
        name="rank_distribution_chittoor",
        query="What is the rank-wise distribution of officers in Chittoor district?",
        expect_any=["Chittoor", "Constable", "Inspector", "rank", "Rank"],
        expect_none=["500"],
    ),
    TestCase(
        name="largest_unit_by_strength",
        query="Which unit has the most personnel posted?",
        expect_any=["unit", "Unit", "personnel", "officer", "most", "highest"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 3. Specific person + detail queries (names from real data)
    # =========================================================================
    TestCase(
        name="person_by_name_detail",
        query="Find all details of Nakka Venkata Srinivasarao",
        expect_any=["Nakka", "Srinivasarao", "Venkata"],
        expect_none=["500", "I wasn't able"],
    ),
    TestCase(
        name="person_by_badge",
        query="Who has badge number 158?",
        expect_any=["158", "badge", "Badge", "officer", "personnel"],
        expect_none=["500"],
    ),
    TestCase(
        name="sp_of_chittoor",
        query="Who is the Superintendent of Police in Chittoor?",
        expect_any=["Superintendent", "SP", "Chittoor"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 4. Unit hierarchy and geography queries
    # =========================================================================
    TestCase(
        name="units_in_visakhapatnam",
        query="List all police stations in Visakhapatnam district",
        expect_any=["Visakhapatnam", "PS", "station", "Station", "unit", "Unit"],
        expect_none=["500"],
    ),
    TestCase(
        name="unit_personnel_count",
        query="How many officers are posted at the Guntur Traffic PS unit?",
        expect_any=["Guntur", "Traffic", "officer", "personnel", "posted"],
        expect_none=["500", "3,266", "3266"],
    ),
    TestCase(
        name="districts_list_all",
        query="List all districts in the system",
        expect_any=["Guntur", "Chittoor", "Visakhapatnam", "district", "District"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 5. Department-based queries
    # =========================================================================
    TestCase(
        name="law_order_personnel",
        query="How many personnel are in the Law & Order Wing department?",
        expect_any=["Law", "Order", "personnel", "officer"],
        expect_none=["500"],
    ),
    TestCase(
        name="department_distribution",
        query="How many personnel are in each department? Group by department name and show the count",
        expect_any=["Law & Order", "department", "Department", "count", "Count", "Wing"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 6. Complex aggregation queries
    # =========================================================================
    TestCase(
        name="rank_per_district_matrix",
        query="Show how many SIs and CIs are in Guntur and Chittoor districts",
        expect_any=["Guntur", "Chittoor"],
        expect_none=["500"],
    ),
    TestCase(
        name="asi_in_annamayya",
        query="List all ASIs posted in Annamayya district with their unit names",
        expect_any=["ASI", "Assistant", "Annamayya"],
        expect_none=["500"],
    ),
    TestCase(
        name="constable_count_top_districts",
        query="Which 3 districts have the most Police Constables?",
        expect_any=["Constable", "PC", "Guntur", "district", "District"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 7. Conversational / natural phrasing queries
    # =========================================================================
    TestCase(
        name="natural_who_heads_guntur",
        query="Who is the SP of Guntur district?",
        expect_any=["Guntur", "SP", "Superintendent", "Jindal", "Malika"],
        expect_none=["500"],
    ),
    TestCase(
        name="natural_female_officers",
        query="Are there any female officers in Chittoor?",
        expect_any=["female", "Female", "woman", "Woman", "officer", "yes", "Yes", "Chittoor"],
        expect_none=["500"],
    ),
    TestCase(
        name="natural_contact_info",
        query="What is the mobile number of Vakul Jindal?",
        expect_any=["mobile", "Mobile", "phone", "Phone", "8688831300", "Vakul"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 8. Edge cases and stress tests
    # =========================================================================
    TestCase(
        name="empty_district_query",
        query="List all officers in Polavaram district",
        expect_any=["Polavaram", "officer", "personnel", "no ", "No "],
        expect_none=["500"],
    ),
    TestCase(
        name="misspelled_district",
        query="Show officers in Chitoor district",
        expect_any=["Chittoor", "Chitoor", "officer", "personnel", "did you mean", "not found", "No district"],
        expect_none=["500"],
    ),
    TestCase(
        name="multi_rank_query",
        query="How many Inspectors and DSPs are there in total?",
        expect_any=["Inspector", "DSP", "Deputy", "total"],
        expect_none=["500"],
    ),
    TestCase(
        name="no_objectids_in_response",
        query="Give me full details of the SP of Guntur including unit and district",
        expect_any=["Guntur", "SP", "Superintendent"],
        # Must NOT contain raw ObjectIDs (24-char hex strings from training DB)
        expect_none=["500", "693915bb87e2a550106d4fbf", "697a60defddfcfc418c8f6288"],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    query: str
    status_code: int
    response_text: str
    duration_s: float
    passed: bool
    failures: List[str]
    route_source: Optional[str] = None
    turns: Optional[int] = None
    routed_to: Optional[str] = None


async def run_test(client: httpx.AsyncClient, tc: TestCase) -> TestResult:
    failures: List[str] = []
    t0 = time.time()
    status_code = 0
    response_text = ""
    route_source = None
    turns = None
    routed_to = None

    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/ask",
            json={"query": tc.query, "output_format": "text"},
            timeout=TIMEOUT,
        )
        status_code = resp.status_code
        body: Dict[str, Any] = resp.json()
        response_text = str(body.get("response") or body.get("answer") or "")
        route_source = body.get("route_source")
        routed_to = body.get("routed_to")
        data_field = body.get("data")
        turns = body.get("turns") or (data_field.get("turns") if isinstance(data_field, dict) else None)

        if tc.expect_success and status_code >= 500:
            failures.append(f"HTTP {status_code} (expected success)")

        if tc.expect_nonempty and not response_text.strip():
            failures.append("Response text is empty")

        # Check expect_any: at least min_matches tokens must appear
        resp_lower = response_text.lower()
        matches = sum(1 for token in tc.expect_any if token.lower() in resp_lower)
        if matches < tc.min_matches:
            failures.append(
                f"Expected at least {tc.min_matches} of {tc.expect_any!r} in response (found {matches})"
            )

        for token in tc.expect_none:
            if token.lower() in resp_lower:
                failures.append(f"Found forbidden token {token!r} in response")

    except httpx.TimeoutException:
        failures.append(f"Request timed out after {TIMEOUT}s")
        status_code = -1
        response_text = "TIMEOUT"
    except Exception as exc:
        failures.append(f"Client error: {exc}")
        status_code = -1

    duration = time.time() - t0
    return TestResult(
        name=tc.name,
        query=tc.query,
        status_code=status_code,
        response_text=response_text,
        duration_s=duration,
        passed=len(failures) == 0,
        failures=failures,
        route_source=route_source,
        turns=turns,
        routed_to=routed_to,
    )


async def main():
    print(f"\n{'='*72}")
    print(f"  Complex Query Integration Tests")
    print(f"  Target: {BASE_URL}  |  Tests: {len(TEST_CASES)}")
    print(f"{'='*72}\n")

    results: List[TestResult] = []

    async with httpx.AsyncClient() as client:
        # Quick connectivity check
        try:
            health = await client.get(f"{BASE_URL}/docs", timeout=5)
            if health.status_code != 200:
                print(f"  WARNING: Server returned {health.status_code} on /docs")
        except Exception:
            print(f"  ERROR: Cannot reach server at {BASE_URL}. Is it running?")
            return

        for i, tc in enumerate(TEST_CASES):
            if i > 0:
                await asyncio.sleep(INTER_TEST_DELAY_S)
            print(f"  [{tc.name}] {tc.query[:60]}...", end="", flush=True)
            r = await run_test(client, tc)
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            route_info = f"via {r.routed_to or '?'}" if r.routed_to else ""
            print(f"  {status}  ({r.duration_s:.1f}s, {r.route_source or 'unknown'} {route_info})")
            if not r.passed:
                for f in r.failures:
                    print(f"         FAIL: {f}")
            preview = r.response_text[:140].replace("\n", " ").encode("ascii", "replace").decode()
            print(f"         Response: {preview}")
            print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{'='*72}")
    print(f"  Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"{'='*72}\n")

    # Route breakdown
    route_counts: Dict[str, int] = {}
    for r in results:
        key = r.route_source or "unknown"
        route_counts[key] = route_counts.get(key, 0) + 1
    print("  ROUTING BREAKDOWN:")
    for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
        print(f"    {route}: {count} queries")
    print()

    # Timing breakdown
    avg_time = sum(r.duration_s for r in results) / len(results) if results else 0
    fast = [r for r in results if r.duration_s < 5]
    medium = [r for r in results if 5 <= r.duration_s < 20]
    slow = [r for r in results if r.duration_s >= 20]
    print("  TIMING BREAKDOWN:")
    print(f"    Average: {avg_time:.1f}s")
    print(f"    Fast (<5s): {len(fast)} queries")
    print(f"    Medium (5-20s): {len(medium)} queries")
    print(f"    Slow (>20s): {len(slow)} queries")
    if slow:
        print(f"    Slow queries: {[r.name for r in slow]}")
    print()

    if failed:
        print("  FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.failures}")
        print()

    # Save results
    out_path = "tests/complex_query_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "name": r.name,
                    "query": r.query,
                    "passed": r.passed,
                    "failures": r.failures,
                    "status_code": r.status_code,
                    "duration_s": round(r.duration_s, 2),
                    "route_source": r.route_source,
                    "routed_to": r.routed_to,
                    "turns": r.turns,
                    "response": r.response_text,
                }
                for r in results
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Full results saved to {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
