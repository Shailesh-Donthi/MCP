"""
Side-by-side comparison: Smart AI (GPT 5.2) vs MongoDB MCP mode.

Runs every test query in both routing modes and produces a comparison
table showing pass/fail, response quality, timing, and accuracy.

Run with:
    python tests/test_mode_comparison.py
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

BASE_URL = "http://127.0.0.1:8090"
TIMEOUT = 120
INTER_TEST_DELAY_S = 2


@dataclass
class TestCase:
    name: str
    query: str
    expect_any: List[str] = field(default_factory=list)
    expect_none: List[str] = field(default_factory=list)
    expect_nonempty: bool = True
    expect_success: bool = True
    min_matches: int = 1
    category: str = ""


# Same 20 test cases from test_new_queries_live.py
TEST_CASES: List[TestCase] = [
    # 1. Novel phrasing
    TestCase(
        name="synonym_cops_in_guntur",
        query="How many cops are posted in Guntur?",
        expect_any=["Guntur", "officer", "personnel", "Police", "Constable", "total", "1,393", "1393"],
        category="Phrasing",
    ),
    TestCase(
        name="synonym_boss_of_chittoor",
        query="Who is the boss of Chittoor police?",
        expect_any=["Chittoor", "SP", "Superintendent", "Tushar", "Dudi"],
        category="Phrasing",
    ),
    TestCase(
        name="informal_phone_lookup",
        query="Can you give me Malika Garg's phone number?",
        expect_any=["Malika", "Garg", "mobile", "Mobile", "phone", "Phone", "96180"],
        category="Phrasing",
    ),

    # 2. Counting
    TestCase(
        name="total_personnel_count",
        query="What is the total number of personnel in the entire system?",
        expect_any=["3,266", "3266", "total", "personnel", "officer"],
        category="Counting",
    ),
    TestCase(
        name="count_sis_in_guntur",
        query="How many Sub-Inspectors are in Guntur district?",
        expect_any=["Guntur", "Sub-Inspector", "Sub Inspector", "SI"],
        category="Counting",
    ),
    TestCase(
        name="count_districts_total",
        query="How many districts are there in the system?",
        expect_any=["district", "District", "26", "27", "28"],
        category="Counting",
    ),

    # 3. Cross-collection
    TestCase(
        name="officers_at_specific_station",
        query="Who is posted at Kuppam PS?",
        expect_any=["Kuppam", "PS", "officer", "personnel", "posted", "no ", "No "],
        category="Cross-collection",
    ),
    TestCase(
        name="units_under_guntur_dpo",
        query="What units fall under Guntur DPO?",
        expect_any=["Guntur", "DPO", "unit", "Unit", "PS"],
        category="Cross-collection",
    ),
    TestCase(
        name="rank_of_specific_person",
        query="What is the rank of Sake Mahendra?",
        expect_any=["Sake", "Mahendra", "DSP", "Deputy", "rank", "Rank"],
        category="Cross-collection",
    ),

    # 4. Negative / boundary
    TestCase(
        name="nonexistent_person",
        query="Find details of John Smith in the system",
        expect_any=["no ", "No ", "not found", "Not found", "match", "0 ", "sorry", "Sorry"],
        category="Edge case",
    ),
    TestCase(
        name="nonexistent_rank",
        query="List all Brigadiers in the system",
        expect_any=["no ", "No ", "not found", "Not found", "0", "Brigadier", "sorry", "Sorry", "unavailable"],
        category="Edge case",
    ),
    TestCase(
        name="empty_result_graceful",
        query="Show all transfers in Srikakulam district in the last week",
        expect_any=["no ", "No ", "transfer", "Transfer", "Srikakulam", "found", "record", "sorry", "Sorry"],
        category="Edge case",
    ),

    # 5. Multi-entity
    TestCase(
        name="compare_three_districts",
        query="Compare officer counts in Guntur, Chittoor, and Visakhapatnam",
        expect_any=["Guntur", "Chittoor"],
        category="Multi-entity",
    ),
    TestCase(
        name="top_ranks_systemwide",
        query="How many SPs, DSPs, and Inspectors are there in total?",
        expect_any=["SP", "DSP", "Inspector", "total", "count"],
        category="Multi-entity",
    ),

    # 6. Specific data
    TestCase(
        name="badge_number_lookup",
        query="Who has badge number 3001?",
        expect_any=["3001", "Malika", "Garg", "badge", "Badge"],
        category="Data lookup",
    ),
    TestCase(
        name="user_id_lookup",
        query="Find the officer with user ID 14402876",
        expect_any=["14402876", "Vakul", "Jindal"],
        category="Data lookup",
    ),
    TestCase(
        name="email_lookup",
        query="Which officer has the mobile number 8688831300?",
        expect_any=["8688831300", "Vakul", "Jindal", "mobile", "phone"],
        category="Data lookup",
    ),

    # 7. Natural language
    TestCase(
        name="natural_smallest_district",
        query="Which district has the fewest officers?",
        expect_any=["district", "District", "officer", "fewest", "least", "smallest", "lowest"],
        category="Natural lang",
    ),
    TestCase(
        name="natural_recent_activity",
        query="Have there been any recent transfers?",
        expect_any=["transfer", "Transfer", "recent", "no ", "No ", "found", "last", "sorry", "Sorry"],
        category="Natural lang",
    ),
    TestCase(
        name="natural_help_request",
        query="What kind of questions can I ask you?",
        expect_any=["personnel", "unit", "district", "search", "query", "help", "can"],
        category="Natural lang",
    ),
]


@dataclass
class SingleResult:
    mode: str
    name: str
    query: str
    passed: bool
    failures: List[str]
    response_text: str
    duration_s: float
    route_source: str
    routed_to: str
    status_code: int


async def run_one(
    client: httpx.AsyncClient,
    tc: TestCase,
    mode: str,
    session_id: str,
) -> SingleResult:
    failures: List[str] = []
    t0 = time.time()
    status_code = 0
    response_text = ""
    route_source = ""
    routed_to = ""

    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/ask",
            json={
                "query": tc.query,
                "output_format": "text",
                "session_id": session_id,
                "routing_mode": mode,
            },
            timeout=TIMEOUT,
        )
        status_code = resp.status_code
        body: Dict[str, Any] = resp.json()
        response_text = str(body.get("response") or body.get("answer") or "")
        route_source = body.get("route_source", "")
        routed_to = body.get("routed_to", "")

        if tc.expect_success and status_code >= 500:
            failures.append(f"HTTP {status_code}")

        if tc.expect_nonempty and not response_text.strip():
            failures.append("Empty response")

        resp_lower = response_text.lower()
        matches = sum(1 for token in tc.expect_any if token.lower() in resp_lower)
        if matches < tc.min_matches:
            failures.append(
                f"Missing keywords ({matches}/{tc.min_matches})"
            )

        for token in tc.expect_none:
            if token.lower() in resp_lower:
                failures.append(f"Forbidden: {token!r}")

    except httpx.TimeoutException:
        failures.append(f"Timeout ({TIMEOUT}s)")
        status_code = -1
        response_text = "TIMEOUT"
    except Exception as exc:
        failures.append(f"Error: {exc}")
        status_code = -1

    return SingleResult(
        mode=mode,
        name=tc.name,
        query=tc.query,
        passed=len(failures) == 0,
        failures=failures,
        response_text=response_text,
        duration_s=time.time() - t0,
        route_source=route_source,
        routed_to=routed_to,
        status_code=status_code,
    )


async def main():
    print(f"\n{'='*80}")
    print(f"  Smart AI (GPT 5.2) vs MongoDB MCP  --  Side-by-Side Comparison")
    print(f"  Target: {BASE_URL}  |  Tests: {len(TEST_CASES)}")
    print(f"{'='*80}")

    async with httpx.AsyncClient() as client:
        # Connectivity check
        try:
            health = await client.get(f"{BASE_URL}/docs", timeout=5)
            if health.status_code != 200:
                print(f"  WARNING: Server returned {health.status_code}")
        except Exception:
            print(f"  ERROR: Cannot reach server at {BASE_URL}. Is it running?")
            return

        # Check model
        try:
            model_resp = await client.get(f"{BASE_URL}/api/v1/llm-model", timeout=5)
            model_data = model_resp.json()
            active = model_data.get("active", "?")
            label = model_data.get("profiles", {}).get(active, {}).get("label", "?")
            print(f"  Active LLM: {label} ({active})")
        except Exception:
            pass

        print()

        smart_results: List[SingleResult] = []
        mcp_results: List[SingleResult] = []

        for i, tc in enumerate(TEST_CASES):
            print(f"  [{i+1:2d}/{len(TEST_CASES)}] {tc.query[:62]}")

            # Run Smart AI first
            r_smart = await run_one(client, tc, "smart_ai", f"cmp_smart_{i}")
            smart_results.append(r_smart)

            await asyncio.sleep(INTER_TEST_DELAY_S)

            # Run MCP mode
            r_mcp = await run_one(client, tc, "mcp_mode", f"cmp_mcp_{i}")
            mcp_results.append(r_mcp)

            s_icon = "PASS" if r_smart.passed else "FAIL"
            m_icon = "FAIL" if not r_mcp.passed else "PASS"

            print(f"         Smart AI: {s_icon} ({r_smart.duration_s:.1f}s) | MCP: {m_icon} ({r_mcp.duration_s:.1f}s)")

            if not r_smart.passed:
                print(f"           Smart AI failures: {r_smart.failures}")
            if not r_mcp.passed:
                print(f"           MCP failures: {r_mcp.failures}")
            print()

            if i < len(TEST_CASES) - 1:
                await asyncio.sleep(INTER_TEST_DELAY_S)

    # ── Summary table ──────────────────────────────────────────────────────
    smart_pass = sum(1 for r in smart_results if r.passed)
    mcp_pass = sum(1 for r in mcp_results if r.passed)
    smart_avg = sum(r.duration_s for r in smart_results) / len(smart_results)
    mcp_avg = sum(r.duration_s for r in mcp_results) / len(mcp_results)

    print(f"\n{'='*80}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*80}")
    print()
    print(f"  {'Metric':<30} {'Smart AI (5.2)':>18} {'MongoDB MCP':>18}")
    print(f"  {'-'*30} {'-'*18} {'-'*18}")
    print(f"  {'Tests Passed':<30} {f'{smart_pass}/{len(TEST_CASES)}':>18} {f'{mcp_pass}/{len(TEST_CASES)}':>18}")
    print(f"  {'Accuracy %':<30} {f'{smart_pass/len(TEST_CASES)*100:.0f}%':>18} {f'{mcp_pass/len(TEST_CASES)*100:.0f}%':>18}")
    print(f"  {'Avg Response Time':<30} {f'{smart_avg:.1f}s':>18} {f'{mcp_avg:.1f}s':>18}")
    print()

    # Category breakdown
    categories = sorted(set(tc.category for tc in TEST_CASES if tc.category))
    if categories:
        print(f"  {'Category':<20} {'Smart AI':>12} {'MCP':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        for cat in categories:
            cat_indices = [i for i, tc in enumerate(TEST_CASES) if tc.category == cat]
            s_cat = sum(1 for i in cat_indices if smart_results[i].passed)
            m_cat = sum(1 for i in cat_indices if mcp_results[i].passed)
            total = len(cat_indices)
            print(f"  {cat:<20} {f'{s_cat}/{total}':>12} {f'{m_cat}/{total}':>12}")
        print()

    # Head-to-head
    smart_only = []
    mcp_only = []
    both_fail = []
    for i, tc in enumerate(TEST_CASES):
        s, m = smart_results[i], mcp_results[i]
        if s.passed and not m.passed:
            smart_only.append(tc.name)
        elif m.passed and not s.passed:
            mcp_only.append(tc.name)
        elif not s.passed and not m.passed:
            both_fail.append(tc.name)

    if smart_only:
        print(f"  Smart AI wins ({len(smart_only)}):")
        for n in smart_only:
            print(f"    - {n}")
        print()
    if mcp_only:
        print(f"  MCP wins ({len(mcp_only)}):")
        for n in mcp_only:
            print(f"    - {n}")
        print()
    if both_fail:
        print(f"  Both failed ({len(both_fail)}):")
        for n in both_fail:
            print(f"    - {n}")
        print()

    # Detailed per-query comparison
    print(f"\n  {'#':<4} {'Test Name':<30} {'Smart AI':>10} {'MCP':>10} {'Time S':>8} {'Time M':>8}")
    print(f"  {'-'*4} {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for i, tc in enumerate(TEST_CASES):
        s, m = smart_results[i], mcp_results[i]
        s_st = "PASS" if s.passed else "FAIL"
        m_st = "PASS" if m.passed else "FAIL"
        print(f"  {i+1:<4} {tc.name:<30} {s_st:>10} {m_st:>10} {s.duration_s:>7.1f}s {m.duration_s:>7.1f}s")

    print()

    # Save full results
    out_path = "tests/mode_comparison_results.json"
    comparison_data = {
        "summary": {
            "smart_ai_passed": smart_pass,
            "mcp_passed": mcp_pass,
            "total_tests": len(TEST_CASES),
            "smart_ai_accuracy": round(smart_pass / len(TEST_CASES) * 100, 1),
            "mcp_accuracy": round(mcp_pass / len(TEST_CASES) * 100, 1),
            "smart_ai_avg_time": round(smart_avg, 2),
            "mcp_avg_time": round(mcp_avg, 2),
        },
        "per_query": [
            {
                "name": tc.name,
                "query": tc.query,
                "category": tc.category,
                "smart_ai": {
                    "passed": smart_results[i].passed,
                    "failures": smart_results[i].failures,
                    "duration_s": round(smart_results[i].duration_s, 2),
                    "route_source": smart_results[i].route_source,
                    "routed_to": smart_results[i].routed_to,
                    "response": smart_results[i].response_text,
                },
                "mcp_mode": {
                    "passed": mcp_results[i].passed,
                    "failures": mcp_results[i].failures,
                    "duration_s": round(mcp_results[i].duration_s, 2),
                    "route_source": mcp_results[i].route_source,
                    "routed_to": mcp_results[i].routed_to,
                    "response": mcp_results[i].response_text,
                },
            }
            for i, tc in enumerate(TEST_CASES)
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved to {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
