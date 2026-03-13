"""
Complex query comparison: Smart AI (GPT 5.2) vs MongoDB MCP mode.

Focuses on domain-heavy queries, multi-collection joins, aggregations,
and AP-police-specific terminology where domain rules matter most.

Run with:
    python tests/test_complex_comparison.py
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
    min_matches: int = 1
    category: str = ""


TEST_CASES: List[TestCase] = [

    # ── Domain-specific AP police terminology ────────────────────────────
    TestCase(
        name="ranges_in_ap",
        query="How many Ranges are there in AP?",
        expect_any=["Range", "range", "unit", "Unit"],
        category="Domain terminology",
    ),
    TestCase(
        name="sp_of_guntur",
        query="Who is the SP of Guntur district?",
        expect_any=["Guntur", "SP", "Superintendent", "Jindal", "Malika"],
        category="Domain terminology",
    ),
    TestCase(
        name="dpo_units",
        query="List all DPO units in the system",
        expect_any=["DPO", "Guntur", "Chittoor", "unit", "Unit"],
        category="Domain terminology",
    ),
    TestCase(
        name="sdpo_in_chittoor",
        query="Who are the SDPOs in Chittoor?",
        expect_any=["SDPO", "Chittoor", "DSP", "Deputy"],
        category="Domain terminology",
    ),
    TestCase(
        name="ars_in_system",
        query="List all Additional SPs in the system",
        expect_any=["Additional", "Addl", "ASP", "SP"],
        category="Domain terminology",
    ),

    # ── Multi-collection joins ───────────────────────────────────────────
    TestCase(
        name="ci_in_guntur",
        query="List all Circle Inspectors posted in Guntur district with their names",
        expect_any=["Circle Inspector", "CI", "Guntur", "Brahmam", "Ashok", "Srinivas"],
        category="Multi-join",
    ),
    TestCase(
        name="dsp_with_units",
        query="Show all DSPs with their current posting units and districts",
        expect_any=["DSP", "Deputy Superintendent", "unit", "Unit", "district", "District"],
        category="Multi-join",
    ),
    TestCase(
        name="hc_count_per_district",
        query="How many Head Constables are there in each district?",
        expect_any=["Head Constable", "HC", "Guntur", "Chittoor"],
        category="Multi-join",
    ),
    TestCase(
        name="person_by_name_detail",
        query="Find all details of Nakka Venkata Srinivasarao",
        expect_any=["Nakka", "Srinivasarao", "Venkata"],
        expect_none=["I wasn't able"],
        category="Multi-join",
    ),

    # ── Aggregation / comparative ────────────────────────────────────────
    TestCase(
        name="rank_distribution_chittoor",
        query="What is the rank-wise distribution of officers in Chittoor district?",
        expect_any=["Chittoor", "Constable", "PC", "Inspector", "SI", "rank", "Rank", "distribution"],
        category="Aggregation",
    ),
    TestCase(
        name="largest_unit_by_strength",
        query="Which unit has the most personnel posted?",
        expect_any=["unit", "Unit", "personnel", "officer", "most", "highest"],
        category="Aggregation",
    ),
    TestCase(
        name="constable_count_top_districts",
        query="Which 3 districts have the most Police Constables?",
        expect_any=["Constable", "PC", "Guntur", "district", "District"],
        category="Aggregation",
    ),
    TestCase(
        name="department_distribution",
        query="How many personnel are in each department? Group by department name",
        expect_any=["Law & Order", "department", "Department", "count", "Count", "Wing", "personnel", "group"],
        category="Aggregation",
    ),
    TestCase(
        name="rank_per_district_matrix",
        query="Show how many SIs and CIs are in Guntur and Chittoor districts",
        expect_any=["Guntur", "Chittoor"],
        category="Aggregation",
    ),

    # ── Unit hierarchy & geography ───────────────────────────────────────
    TestCase(
        name="units_in_visakhapatnam",
        query="List all police stations in Visakhapatnam district",
        expect_any=["Visakhapatnam", "PS", "station", "Station", "unit", "Unit"],
        category="Unit hierarchy",
    ),
    TestCase(
        name="unit_personnel_count",
        query="How many officers are posted at the Guntur Traffic PS unit?",
        expect_any=["Guntur", "Traffic", "officer", "personnel", "posted", "no record", "No record", "matched", "not found"],
        expect_none=["3,266", "3266"],
        category="Unit hierarchy",
    ),

    # ── Edge cases ───────────────────────────────────────────────────────
    TestCase(
        name="misspelled_district",
        query="Show officers in Chitoor district",
        expect_any=["Chittoor", "Chitoor", "officer", "personnel", "did you mean", "not found", "No district", "no record", "No record", "posted", "district"],
        category="Edge case",
    ),
    TestCase(
        name="no_objectids_in_response",
        query="Give me full details of the SP of Guntur including unit and district",
        expect_any=["Guntur", "SP", "Superintendent"],
        expect_none=["693915bb87e2a550106d4fbf", "697a60defddfcfc418c8f6288"],
        category="Edge case",
    ),
    TestCase(
        name="empty_district_query",
        query="List all officers in Polavaram district",
        expect_any=["Polavaram", "officer", "personnel", "no ", "No "],
        category="Edge case",
    ),

    # ── Speed test: simple lookups that should be fast ───────────────────
    TestCase(
        name="fast_badge_lookup",
        query="Who has badge number 3001?",
        expect_any=["3001", "Malika", "Garg"],
        category="Speed test",
    ),
    TestCase(
        name="fast_phone_lookup",
        query="What is the mobile number of Vakul Jindal?",
        expect_any=["mobile", "Mobile", "phone", "Phone", "8688831300", "Vakul"],
        category="Speed test",
    ),
    TestCase(
        name="fast_district_count",
        query="How many districts are there?",
        expect_any=["district", "District", "26", "27", "28"],
        category="Speed test",
    ),
    TestCase(
        name="fast_total_personnel",
        query="Total number of officers in the system?",
        expect_any=["3,266", "3266", "total", "personnel", "officer"],
        category="Speed test",
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


async def run_one(
    client: httpx.AsyncClient,
    tc: TestCase,
    mode: str,
    session_id: str,
) -> SingleResult:
    failures: List[str] = []
    t0 = time.time()
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
        body: Dict[str, Any] = resp.json()
        response_text = str(body.get("response") or body.get("answer") or "")
        route_source = body.get("route_source", "")
        routed_to = body.get("routed_to", "")

        if resp.status_code >= 500:
            failures.append(f"HTTP {resp.status_code}")
        if tc.expect_nonempty and not response_text.strip():
            failures.append("Empty response")

        resp_lower = response_text.lower()
        matches = sum(1 for token in tc.expect_any if token.lower() in resp_lower)
        if matches < tc.min_matches:
            failures.append(f"Missing keywords ({matches}/{tc.min_matches})")
        for token in tc.expect_none:
            if token.lower() in resp_lower:
                failures.append(f"Forbidden: {token!r}")

    except httpx.TimeoutException:
        failures.append(f"Timeout ({TIMEOUT}s)")
        response_text = "TIMEOUT"
    except Exception as exc:
        failures.append(f"Error: {exc}")

    return SingleResult(
        mode=mode, name=tc.name, query=tc.query,
        passed=len(failures) == 0, failures=failures,
        response_text=response_text, duration_s=time.time() - t0,
        route_source=route_source, routed_to=routed_to,
    )


async def main():
    n = len(TEST_CASES)
    print(f"\n{'='*80}")
    print(f"  Complex Query Comparison: Smart AI (5.2) vs MongoDB MCP")
    print(f"  Target: {BASE_URL}  |  Tests: {n}")
    print(f"{'='*80}")

    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{BASE_URL}/docs", timeout=5)
        except Exception:
            print(f"  ERROR: Cannot reach server at {BASE_URL}.")
            return

        try:
            m = (await client.get(f"{BASE_URL}/api/v1/llm-model", timeout=5)).json()
            active = m.get("active", "?")
            label = m.get("profiles", {}).get(active, {}).get("label", "?")
            print(f"  Active LLM: {label} ({active})\n")
        except Exception:
            print()

        smart_results: List[SingleResult] = []
        mcp_results: List[SingleResult] = []

        for i, tc in enumerate(TEST_CASES):
            print(f"  [{i+1:2d}/{n}] {tc.query[:65]}")

            r_s = await run_one(client, tc, "smart_ai", f"cx_s_{i}")
            smart_results.append(r_s)
            await asyncio.sleep(INTER_TEST_DELAY_S)

            r_m = await run_one(client, tc, "mcp_mode", f"cx_m_{i}")
            mcp_results.append(r_m)

            s_tag = "PASS" if r_s.passed else "FAIL"
            m_tag = "PASS" if r_m.passed else "FAIL"
            print(f"         Smart AI: {s_tag} ({r_s.duration_s:.1f}s) | MCP: {m_tag} ({r_m.duration_s:.1f}s)")
            if not r_s.passed:
                print(f"           Smart AI: {r_s.failures}")
            if not r_m.passed:
                print(f"           MCP:      {r_m.failures}")
            print()
            if i < n - 1:
                await asyncio.sleep(INTER_TEST_DELAY_S)

    # ── Summary ────────────────────────────────────────────────────────────
    sp = sum(1 for r in smart_results if r.passed)
    mp = sum(1 for r in mcp_results if r.passed)
    sa = sum(r.duration_s for r in smart_results) / n
    ma = sum(r.duration_s for r in mcp_results) / n
    st = sum(r.duration_s for r in smart_results)
    mt = sum(r.duration_s for r in mcp_results)

    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(f"  {'Metric':<30} {'Smart AI (5.2)':>18} {'MongoDB MCP':>18}")
    print(f"  {'-'*30} {'-'*18} {'-'*18}")
    print(f"  {'Tests Passed':<30} {f'{sp}/{n}':>18} {f'{mp}/{n}':>18}")
    print(f"  {'Accuracy %':<30} {f'{sp/n*100:.0f}%':>18} {f'{mp/n*100:.0f}%':>18}")
    print(f"  {'Avg Response Time':<30} {f'{sa:.1f}s':>18} {f'{ma:.1f}s':>18}")
    print(f"  {'Total Time':<30} {f'{st:.0f}s':>18} {f'{mt:.0f}s':>18}")
    print()

    # Category breakdown
    categories = sorted(set(tc.category for tc in TEST_CASES if tc.category))
    print(f"  {'Category':<22} {'Smart AI':>12} {'MCP':>12}  {'Avg Time S':>10} {'Avg Time M':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*12}  {'-'*10} {'-'*10}")
    for cat in categories:
        idx = [i for i, tc in enumerate(TEST_CASES) if tc.category == cat]
        sc = sum(1 for i in idx if smart_results[i].passed)
        mc = sum(1 for i in idx if mcp_results[i].passed)
        sat = sum(smart_results[i].duration_s for i in idx) / len(idx)
        mat = sum(mcp_results[i].duration_s for i in idx) / len(idx)
        tot = len(idx)
        print(f"  {cat:<22} {f'{sc}/{tot}':>12} {f'{mc}/{tot}':>12}  {f'{sat:.1f}s':>10} {f'{mat:.1f}s':>10}")
    print()

    # Speed test detail
    speed_idx = [i for i, tc in enumerate(TEST_CASES) if tc.category == "Speed test"]
    if speed_idx:
        print(f"  SPEED TEST DETAIL (simple lookups):")
        for i in speed_idx:
            tc = TEST_CASES[i]
            s, m = smart_results[i], mcp_results[i]
            faster = "Smart AI" if s.duration_s < m.duration_s else "MCP"
            diff = abs(s.duration_s - m.duration_s)
            print(f"    {tc.name:<25} Smart: {s.duration_s:>5.1f}s  MCP: {m.duration_s:>5.1f}s  ({faster} by {diff:.1f}s)")
        print()

    # Head-to-head
    smart_only = [TEST_CASES[i].name for i in range(n) if smart_results[i].passed and not mcp_results[i].passed]
    mcp_only = [TEST_CASES[i].name for i in range(n) if mcp_results[i].passed and not smart_results[i].passed]
    both_fail = [TEST_CASES[i].name for i in range(n) if not smart_results[i].passed and not mcp_results[i].passed]

    if smart_only:
        print(f"  Smart AI ONLY wins ({len(smart_only)}):")
        for nm in smart_only:
            print(f"    - {nm}")
        print()
    if mcp_only:
        print(f"  MCP ONLY wins ({len(mcp_only)}):")
        for nm in mcp_only:
            print(f"    - {nm}")
        print()
    if both_fail:
        print(f"  BOTH failed ({len(both_fail)}):")
        for nm in both_fail:
            print(f"    - {nm}")
        print()

    # Full table
    print(f"\n  {'#':<4} {'Test Name':<28} {'Cat':<18} {'Smart':>6} {'MCP':>6} {'T(S)':>7} {'T(M)':>7}")
    print(f"  {'-'*4} {'-'*28} {'-'*18} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for i, tc in enumerate(TEST_CASES):
        s, m = smart_results[i], mcp_results[i]
        print(f"  {i+1:<4} {tc.name:<28} {tc.category:<18} {'PASS' if s.passed else 'FAIL':>6} {'PASS' if m.passed else 'FAIL':>6} {s.duration_s:>6.1f}s {m.duration_s:>6.1f}s")
    print()

    # Save
    out_path = "tests/complex_comparison_results.json"
    data = {
        "summary": {
            "smart_ai_passed": sp, "mcp_passed": mp, "total": n,
            "smart_ai_accuracy": round(sp / n * 100, 1),
            "mcp_accuracy": round(mp / n * 100, 1),
            "smart_ai_avg_time": round(sa, 2),
            "mcp_avg_time": round(ma, 2),
        },
        "per_query": [
            {
                "name": TEST_CASES[i].name, "query": TEST_CASES[i].query,
                "category": TEST_CASES[i].category,
                "smart_ai": {
                    "passed": smart_results[i].passed, "failures": smart_results[i].failures,
                    "duration_s": round(smart_results[i].duration_s, 2),
                    "route_source": smart_results[i].route_source,
                    "response": smart_results[i].response_text,
                },
                "mcp_mode": {
                    "passed": mcp_results[i].passed, "failures": mcp_results[i].failures,
                    "duration_s": round(mcp_results[i].duration_s, 2),
                    "route_source": mcp_results[i].route_source,
                    "response": mcp_results[i].response_text,
                },
            }
            for i in range(n)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
