"""
Live integration tests for the dynamic_only query assistant.
Sends real HTTP requests to the running server at localhost:8090 and
grades each response for correctness / completeness.

Run with:
    python tests/test_dynamic_query_live.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Pause between tests to avoid hitting LLM API rate limits
INTER_TEST_DELAY_S = 3

import httpx

BASE_URL = "http://127.0.0.1:8090"
TIMEOUT = 120  # seconds per query — LLM + DB can be slow


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    query: str
    # At least one of these strings must appear in the response text
    expect_any: List[str] = field(default_factory=list)
    # None of these strings should appear
    expect_none: List[str] = field(default_factory=list)
    # Response must not be empty
    expect_nonempty: bool = True
    # Should succeed (no HTTP 500)
    expect_success: bool = True


TEST_CASES: List[TestCase] = [
    # --- Basic rank queries -----------------------------------------------
    TestCase(
        name="list_all_SIs",
        query="List all Sub-Inspectors",
        expect_any=["Sub-Inspector", "SI", "sub-inspector"],
        expect_none=["Internal Server Error", "I wasn't able"],
    ),
    TestCase(
        name="list_all_SPs",
        query="List all Superintendents of Police",
        expect_any=["Superintendent", "SP"],
        expect_none=["Internal Server Error"],
    ),
    TestCase(
        name="count_personnel_by_rank",
        query="How many personnel are there in each rank?",
        expect_any=["Sub-Inspector", "Inspector", "Constable", "rank"],
        expect_none=["500"],
    ),

    # --- District-based queries -------------------------------------------
    TestCase(
        name="personnel_in_district",
        query="How many personnel are in Guntur district?",
        expect_any=["Guntur", "personnel", "officer"],
        expect_none=["500", "I wasn't able"],
    ),
    TestCase(
        name="sp_of_guntur",
        query="Who is the SP of Guntur?",
        expect_any=["Superintendent", "SP", "Guntur"],
        expect_none=["500", "No personnel records"],
    ),
    TestCase(
        name="officers_in_chittoor",
        query="List officers posted in Chittoor district",
        expect_any=["Chittoor", "officer", "personnel"],
        expect_none=["500"],
    ),

    # --- Unit-based queries -----------------------------------------------
    TestCase(
        name="unit_hierarchy",
        query="What is the unit hierarchy for Chittoor?",
        expect_any=["Chittoor", "unit", "Unit"],
        expect_none=["500"],
    ),
    TestCase(
        name="personnel_at_unit",
        query="Who are the officers posted at Guntur Urban PS?",
        expect_any=["officer", "personnel", "posted", "Guntur"],
        expect_none=["500"],
    ),

    # --- Cross-collection join queries ------------------------------------
    TestCase(
        name="rank_and_unit_join",
        query="Show me all Inspectors with their current posting unit",
        expect_any=["Inspector", "unit", "Unit", "posted"],
        expect_none=["500"],
    ),
    TestCase(
        name="recent_transfers",
        query="Show recent transfers in the last 30 days",
        expect_any=["transfer", "Transfer", "assignment", "posted", "days", "recent", "no recent", "No recent"],
        expect_none=["500"],
    ),

    # --- Distribution / statistics ----------------------------------------
    TestCase(
        name="personnel_distribution",
        query="Show personnel distribution by rank",
        expect_any=["rank", "Rank", "count", "Count", "Sub-Inspector", "Inspector"],
        expect_none=["500"],
    ),
    TestCase(
        name="vacancy_report",
        query="List all vacancies by rank",
        expect_any=["vacanc", "Vacanc", "rank", "sanctioned", "actual", "no vacanc", "No vacanc"],
        expect_none=["500"],
    ),

    # --- Specific person lookups ------------------------------------------
    TestCase(
        name="search_by_name",
        query="Find personnel named Ravi Kumar",
        expect_any=["Ravi", "Kumar", "officer", "personnel", "found", "no personnel", "No personnel"],
        expect_none=["500"],
    ),

    # --- Edge / tricky cases ---------------------------------------------
    TestCase(
        name="ambiguous_query",
        query="who is in charge",
        expect_any=["officer", "charge", "unit", "personnel", "clarif", "sorry", "Sorry", "please", "Please"],
        expect_none=["500"],
    ),
    TestCase(
        name="nonsense_query",
        query="xzqywm abcdef",
        expect_success=True,  # should not 500, just return graceful message
        expect_any=["sorry", "Sorry", "unable", "Unable", "not", "No", "clarif"],
    ),
    TestCase(
        name="missing_mappings",
        query="Show missing village mappings",
        expect_any=["village", "Village", "mapping", "Mapping", "unit", "no missing", "No missing"],
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
    turns: Optional[int] = None


async def run_test(client: httpx.AsyncClient, tc: TestCase) -> TestResult:
    failures: List[str] = []
    t0 = time.time()
    status_code = 0
    response_text = ""
    route_source = None
    turns = None

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
        turns = body.get("turns") or (body.get("data") or {}).get("turns")

        if tc.expect_success and status_code >= 500:
            failures.append(f"HTTP {status_code} (expected success)")

        if tc.expect_nonempty and not response_text.strip():
            failures.append("Response text is empty")

        for token in tc.expect_any:
            if not any(token.lower() in response_text.lower() for token in tc.expect_any):
                failures.append(f"Expected one of {tc.expect_any!r} in response")
                break

        for token in tc.expect_none:
            if token.lower() in response_text.lower():
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
    )


async def main():
    print(f"\n{'='*72}")
    print(f"  Dynamic Query Assistant — Live Integration Tests")
    print(f"  Target: {BASE_URL}  |  Mode: dynamic_only")
    print(f"{'='*72}\n")

    results: List[TestResult] = []

    # Run tests sequentially to avoid overwhelming the LLM API
    async with httpx.AsyncClient() as client:
        for i, tc in enumerate(TEST_CASES):
            if i > 0:
                await asyncio.sleep(INTER_TEST_DELAY_S)
            print(f"  [{tc.name}] {tc.query[:60]}...", end="", flush=True)
            r = await run_test(client, tc)
            results.append(r)
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  ({r.duration_s:.1f}s, HTTP {r.status_code}, turns={r.turns})")
            if not r.passed:
                for f in r.failures:
                    print(f"         FAIL: {f}")
            # Truncate long responses in summary (strip non-ASCII for console)
            preview = r.response_text[:120].replace("\n", " ").encode("ascii", "replace").decode()
            print(f"         Response: {preview}")
            print()

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{'='*72}")
    print(f"  Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"{'='*72}\n")

    if failed:
        print("  FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"    • {r.name}: {r.failures}")
        print()

    # Detailed issue analysis
    print("  ISSUE ANALYSIS:")
    slow = [r for r in results if r.duration_s > 20]
    if slow:
        print(f"    Slow queries (>20s): {[r.name for r in slow]}")

    timeouts = [r for r in results if r.status_code == -1]
    if timeouts:
        print(f"    Timeouts: {[r.name for r in timeouts]}")

    errors = [r for r in results if r.status_code >= 500]
    if errors:
        print(f"    Server errors (5xx): {[r.name for r in errors]}")

    empty = [r for r in results if r.passed and not r.response_text.strip()]
    if empty:
        print(f"    Empty responses: {[r.name for r in empty]}")

    no_issues = not (slow or timeouts or errors or empty)
    if no_issues and not failed:
        print("    No issues detected.")
    elif not (slow or timeouts or errors or empty):
        print("    See failed tests above.")
    print()

    # Save full results to file
    out_path = "tests/dynamic_query_test_results.json"
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
