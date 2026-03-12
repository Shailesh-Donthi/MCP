"""
Compare LLM model accuracy and response time across test cases.
Runs the full test suite once per model profile and prints a side-by-side report.

Usage:
    python tests/compare_models.py
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


# Import test cases from both test files
from test_complex_queries_live import TEST_CASES as COMPLEX_CASES
from test_dynamic_query_live import TEST_CASES as DYNAMIC_CASES

# Combine both suites — dynamic cases need min_matches defaulted
ALL_CASES = []
for tc in COMPLEX_CASES:
    ALL_CASES.append(tc)
for tc in DYNAMIC_CASES:
    ALL_CASES.append(TestCase(
        name=f"dyn_{tc.name}",
        query=tc.query,
        expect_any=tc.expect_any,
        expect_none=tc.expect_none,
        expect_nonempty=tc.expect_nonempty,
        expect_success=tc.expect_success,
        min_matches=1,
    ))


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_s: float
    failures: List[str]
    route_source: Optional[str] = None
    routed_to: Optional[str] = None
    response_preview: str = ""


async def switch_model(client: httpx.AsyncClient, profile: str) -> str:
    """Switch the active LLM model and return the active label."""
    resp = await client.post(
        f"{BASE_URL}/api/v1/llm-model",
        json={"profile": profile},
        timeout=10,
    )
    data = resp.json()
    active = data.get("active", profile)
    label = data.get("profiles", {}).get(active, {}).get("label", active)
    return label


async def run_single_test(client: httpx.AsyncClient, tc) -> TestResult:
    failures: List[str] = []
    t0 = time.time()
    route_source = None
    routed_to = None
    response_text = ""

    try:
        resp = await client.post(
            f"{BASE_URL}/api/v1/ask",
            json={"query": tc.query, "output_format": "text"},
            timeout=TIMEOUT,
        )
        body: Dict[str, Any] = resp.json()
        response_text = str(body.get("response") or body.get("answer") or "")
        route_source = body.get("route_source")
        routed_to = body.get("routed_to")

        if tc.expect_success and resp.status_code >= 500:
            failures.append(f"HTTP {resp.status_code}")

        if tc.expect_nonempty and not response_text.strip():
            failures.append("Empty response")

        resp_lower = response_text.lower()
        matches = sum(1 for token in tc.expect_any if token.lower() in resp_lower)
        if matches < tc.min_matches:
            failures.append(f"Missing expected tokens ({matches}/{tc.min_matches})")

        for token in tc.expect_none:
            if token.lower() in resp_lower:
                failures.append(f"Forbidden: {token!r}")

    except httpx.TimeoutException:
        failures.append("TIMEOUT")
    except Exception as exc:
        failures.append(f"Error: {exc}")

    duration = time.time() - t0
    preview = response_text[:100].replace("\n", " ").encode("ascii", "replace").decode()
    return TestResult(
        name=tc.name,
        passed=len(failures) == 0,
        duration_s=duration,
        failures=failures,
        route_source=route_source,
        routed_to=routed_to,
        response_preview=preview,
    )


async def run_suite(client: httpx.AsyncClient, profile: str, label: str) -> List[TestResult]:
    print(f"\n{'='*72}")
    print(f"  Running {len(ALL_CASES)} tests on: {label} (profile={profile})")
    print(f"{'='*72}\n")

    results: List[TestResult] = []
    for i, tc in enumerate(ALL_CASES):
        if i > 0:
            await asyncio.sleep(INTER_TEST_DELAY_S)
        short_q = tc.query[:55]
        print(f"  [{i+1:2d}/{len(ALL_CASES)}] {short_q}...", end="", flush=True)
        r = await run_single_test(client, tc)
        results.append(r)
        tag = "PASS" if r.passed else "FAIL"
        print(f"  {tag}  {r.duration_s:.1f}s")
        if not r.passed:
            for f in r.failures:
                print(f"         -> {f}")

    return results


def print_comparison(label_a: str, results_a: List[TestResult],
                     label_b: str, results_b: List[TestResult]):
    print(f"\n{'='*90}")
    print(f"  MODEL COMPARISON: {label_a}  vs  {label_b}")
    print(f"{'='*90}\n")

    # Header
    name_w = 28
    print(f"  {'Test':<{name_w}} | {'':^16} {label_a:^16} | {'':^16} {label_b:^16}")
    print(f"  {'':-<{name_w}} | {'Status':^16} {'Time':^16} | {'Status':^16} {'Time':^16}")
    print(f"  {'-'*name_w}-+-{'-'*16}-{'-'*16}-+-{'-'*16}-{'-'*16}")

    for ra, rb in zip(results_a, results_b):
        sa = "PASS" if ra.passed else "FAIL"
        sb = "PASS" if rb.passed else "FAIL"
        ta = f"{ra.duration_s:.1f}s"
        tb = f"{rb.duration_s:.1f}s"
        name = ra.name[:name_w]
        print(f"  {name:<{name_w}} | {sa:^16} {ta:^16} | {sb:^16} {tb:^16}")

    # Summary stats
    pass_a = sum(1 for r in results_a if r.passed)
    pass_b = sum(1 for r in results_b if r.passed)
    total = len(results_a)
    avg_a = sum(r.duration_s for r in results_a) / total if total else 0
    avg_b = sum(r.duration_s for r in results_b) / total if total else 0
    total_a = sum(r.duration_s for r in results_a)
    total_b = sum(r.duration_s for r in results_b)
    fast_a = sum(1 for r in results_a if r.duration_s < 5)
    fast_b = sum(1 for r in results_b if r.duration_s < 5)
    slow_a = sum(1 for r in results_a if r.duration_s >= 20)
    slow_b = sum(1 for r in results_b if r.duration_s >= 20)

    print(f"\n  {'='*90}")
    print(f"  {'METRIC':<30} | {label_a:^20} | {label_b:^20}")
    print(f"  {'-'*30}-+-{'-'*20}-+-{'-'*20}")
    print(f"  {'Pass rate':<30} | {pass_a}/{total} ({100*pass_a/total:.0f}%){'':<8} | {pass_b}/{total} ({100*pass_b/total:.0f}%)")
    print(f"  {'Avg response time':<30} | {avg_a:.1f}s{'':<14} | {avg_b:.1f}s")
    print(f"  {'Total time':<30} | {total_a:.1f}s{'':<13} | {total_b:.1f}s")
    print(f"  {'Fast queries (<5s)':<30} | {fast_a:<20} | {fast_b}")
    print(f"  {'Slow queries (>20s)':<30} | {slow_a:<20} | {slow_b}")
    print(f"  {'='*90}\n")

    # Differences
    only_a_pass = [ra.name for ra, rb in zip(results_a, results_b) if ra.passed and not rb.passed]
    only_b_pass = [ra.name for ra, rb in zip(results_a, results_b) if not ra.passed and rb.passed]
    if only_a_pass:
        print(f"  Tests passing ONLY on {label_a}: {only_a_pass}")
    if only_b_pass:
        print(f"  Tests passing ONLY on {label_b}: {only_b_pass}")
    if not only_a_pass and not only_b_pass:
        print(f"  Both models pass/fail the same tests.")
    print()

    # Speed comparison
    faster_a = sum(1 for ra, rb in zip(results_a, results_b) if ra.duration_s < rb.duration_s)
    faster_b = total - faster_a
    print(f"  Speed: {label_a} faster on {faster_a}/{total} queries, {label_b} faster on {faster_b}/{total}")
    speedup = total_a / total_b if total_b > 0 else 0
    if speedup > 1:
        print(f"  Overall: {label_b} is {speedup:.1f}x faster in total time")
    else:
        print(f"  Overall: {label_a} is {1/speedup:.1f}x faster in total time")
    print()


async def main():
    async with httpx.AsyncClient() as client:
        # Verify server is up
        try:
            await client.get(f"{BASE_URL}/health", timeout=5)
        except Exception:
            print(f"ERROR: Cannot reach server at {BASE_URL}")
            return

        # Get available profiles
        model_resp = await client.get(f"{BASE_URL}/api/v1/llm-model", timeout=5)
        profiles = model_resp.json().get("profiles", {})
        print(f"Available models: {json.dumps(profiles, indent=2)}")

        # Run on default model (Azure gpt-oss-120b)
        label_a = await switch_model(client, "default")
        results_a = await run_suite(client, "default", label_a)

        # Run on GPT 5.2
        label_b = await switch_model(client, "gpt5")
        results_b = await run_suite(client, "gpt5", label_b)

        # Switch back to default
        await switch_model(client, "default")

        # Print comparison
        print_comparison(label_a, results_a, label_b, results_b)

        # Save results
        out = {
            "models": {
                "a": {"profile": "default", "label": label_a},
                "b": {"profile": "gpt5", "label": label_b},
            },
            "results_a": [
                {"name": r.name, "passed": r.passed, "duration_s": round(r.duration_s, 2),
                 "failures": r.failures, "route_source": r.route_source}
                for r in results_a
            ],
            "results_b": [
                {"name": r.name, "passed": r.passed, "duration_s": round(r.duration_s, 2),
                 "failures": r.failures, "route_source": r.route_source}
                for r in results_b
            ],
        }
        import os
        out_path = os.path.join(os.path.dirname(__file__), "model_comparison_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Full results saved to {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
