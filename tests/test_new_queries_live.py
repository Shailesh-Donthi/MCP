"""
Fresh query verification tests — queries NOT in existing test suites.
Tests novel phrasing, edge cases, and multi-step reasoning.

Run with:
    python tests/test_new_queries_live.py
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
    min_matches: int = 1


TEST_CASES: List[TestCase] = [

    # =========================================================================
    # 1. Novel phrasing / synonym tests
    # =========================================================================
    TestCase(
        name="synonym_cops_in_guntur",
        query="How many cops are posted in Guntur?",
        expect_any=["Guntur", "officer", "personnel", "Police", "Constable", "total", "1,393", "1393"],
        expect_none=["500"],
    ),
    TestCase(
        name="synonym_boss_of_chittoor",
        query="Who is the boss of Chittoor police?",
        expect_any=["Chittoor", "SP", "Superintendent", "Tushar", "Dudi"],
        expect_none=["500"],
    ),
    TestCase(
        name="informal_phone_lookup",
        query="Can you give me Malika Garg's phone number?",
        expect_any=["Malika", "Garg", "mobile", "Mobile", "phone", "Phone", "96180"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 2. Counting and math queries
    # =========================================================================
    TestCase(
        name="total_personnel_count",
        query="What is the total number of personnel in the entire system?",
        expect_any=["3,266", "3266", "total", "personnel", "officer"],
        expect_none=["500"],
    ),
    TestCase(
        name="count_sis_in_guntur",
        query="How many Sub-Inspectors are in Guntur district?",
        expect_any=["Guntur", "Sub-Inspector", "Sub Inspector", "SI"],
        expect_none=["500"],
    ),
    TestCase(
        name="count_districts_total",
        query="How many districts are there in the system?",
        expect_any=["district", "District", "26", "27", "28"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 3. Cross-collection / join queries
    # =========================================================================
    TestCase(
        name="officers_at_specific_station",
        query="Who is posted at Kuppam PS?",
        expect_any=["Kuppam", "PS", "officer", "personnel", "posted", "no ", "No "],
        expect_none=["500"],
    ),
    TestCase(
        name="units_under_guntur_dpo",
        query="What units fall under Guntur DPO?",
        expect_any=["Guntur", "DPO", "unit", "Unit", "PS"],
        expect_none=["500"],
    ),
    TestCase(
        name="rank_of_specific_person",
        query="What is the rank of Sake Mahendra?",
        expect_any=["Sake", "Mahendra", "DSP", "Deputy", "rank", "Rank"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 4. Negative / boundary tests
    # =========================================================================
    TestCase(
        name="nonexistent_person",
        query="Find details of John Smith in the system",
        expect_any=["no ", "No ", "not found", "Not found", "match", "0 ", "sorry", "Sorry"],
        expect_none=["500"],
    ),
    TestCase(
        name="nonexistent_rank",
        query="List all Brigadiers in the system",
        expect_any=["no ", "No ", "not found", "Not found", "0", "Brigadier", "sorry", "Sorry", "unavailable"],
        expect_none=["500"],
    ),
    TestCase(
        name="empty_result_graceful",
        query="Show all transfers in Srikakulam district in the last week",
        expect_any=["no ", "No ", "transfer", "Transfer", "Srikakulam", "found", "record", "sorry", "Sorry"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 5. Multi-entity / comparative
    # =========================================================================
    TestCase(
        name="compare_three_districts",
        query="Compare officer counts in Guntur, Chittoor, and Visakhapatnam",
        expect_any=["Guntur", "Chittoor"],
        expect_none=["500"],
    ),
    TestCase(
        name="top_ranks_systemwide",
        query="How many SPs, DSPs, and Inspectors are there in total?",
        expect_any=["SP", "DSP", "Inspector", "total", "count"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 6. Specific data point queries
    # =========================================================================
    TestCase(
        name="badge_number_lookup",
        query="Who has badge number 3001?",
        expect_any=["3001", "Malika", "Garg", "badge", "Badge"],
        expect_none=["500"],
    ),
    TestCase(
        name="user_id_lookup",
        query="Find the officer with user ID 14402876",
        expect_any=["14402876", "Vakul", "Jindal"],
        expect_none=["500"],
    ),
    TestCase(
        name="email_lookup",
        query="Which officer has the mobile number 8688831300?",
        expect_any=["8688831300", "Vakul", "Jindal", "mobile", "phone"],
        expect_none=["500"],
    ),

    # =========================================================================
    # 7. Natural language / conversational
    # =========================================================================
    TestCase(
        name="natural_smallest_district",
        query="Which district has the fewest officers?",
        expect_any=["district", "District", "officer", "fewest", "least", "smallest", "lowest"],
        expect_none=["500"],
    ),
    TestCase(
        name="natural_recent_activity",
        query="Have there been any recent transfers?",
        expect_any=["transfer", "Transfer", "recent", "no ", "No ", "found", "last", "sorry", "Sorry"],
        expect_none=["500"],
    ),
    TestCase(
        name="natural_help_request",
        query="What kind of questions can I ask you?",
        expect_any=["personnel", "unit", "district", "search", "query", "help", "can"],
        expect_none=["500"],
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
    routed_to: Optional[str] = None


async def run_test(client: httpx.AsyncClient, tc: TestCase) -> TestResult:
    failures: List[str] = []
    t0 = time.time()
    status_code = 0
    response_text = ""
    route_source = None
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

        if tc.expect_success and status_code >= 500:
            failures.append(f"HTTP {status_code} (expected success)")

        if tc.expect_nonempty and not response_text.strip():
            failures.append("Response text is empty")

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
        routed_to=routed_to,
    )


async def main():
    print(f"\n{'='*72}")
    print(f"  New Query Verification Tests (GPT 5.2)")
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

        # Check active model
        try:
            model_resp = await client.get(f"{BASE_URL}/api/v1/llm-model", timeout=5)
            model_data = model_resp.json()
            active = model_data.get("active", "?")
            label = model_data.get("profiles", {}).get(active, {}).get("label", "?")
            print(f"  Active model: {label} ({active})\n")
        except Exception:
            print(f"  Could not determine active model\n")

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

    # Timing
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
    out_path = "tests/new_query_test_results.json"
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
