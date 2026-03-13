"""Test edge cases #4-#12 for routing_rules.repair_route."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from mcp.router.routing_rules import repair_route

PASS = 0
FAIL = 0

def check(label, query, expected_tool, expected_arg_subset=None):
    global PASS, FAIL
    tool, args = repair_route(query, "search_personnel", {})
    ok = tool == expected_tool
    if expected_arg_subset:
        for k, v in expected_arg_subset.items():
            if args.get(k) != v:
                ok = False
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  {status}: {label}")
        print(f"         query: {query}")
        print(f"         expected: {expected_tool} {expected_arg_subset or ''}")
        print(f"         got:      {tool} {args}")
    else:
        PASS += 1
        print(f"  {status}: {label}")

print("=== Edge Case #4: Subordinate/reporting queries ===")
check("#4a reports to SP Guntur",
      "Who reports to the SP of Guntur?",
      "query_personnel_by_rank",
      {"rank_relation": "below"})
check("#4b subordinates of DSP",
      "Subordinates of DSP in Chittoor",
      "query_personnel_by_rank",
      {"rank_relation": "below"})

print("\n=== Edge Case #5: Temporal/retirement queries ===")
check("#5a retiring this year",
      "Officers retiring this year",
      "dynamic_query")
check("#5b born before 1970",
      "Officers born before 1970 in Guntur",
      "dynamic_query")

print("\n=== Edge Case #6: Transfer source+destination ===")
check("#6a from Guntur to Chittoor",
      "Transfers from Guntur to Chittoor",
      "dynamic_query")
check("#6b single district transfer",
      "Transfers in Guntur last 30 days",
      "query_recent_transfers",
      {"district_name": "Guntur"})

print("\n=== Edge Case #7: Unit type count ===")
check("#7a how many PS in Guntur",
      "How many police stations in Guntur?",
      "list_units_in_district",
      {"unit_type_name": "PS"})
check("#7b count of stations in Guntur",
      "Count of PS in Guntur",
      "list_units_in_district",
      {"unit_type_name": "PS", "district_name": "Guntur"})
check("#7c how many circles",
      "How many circles are there?",
      "dynamic_query")

print("\n=== Edge Case #8: Bare org chart ===")
check("#8a org chart no district",
      "Show me the org chart",
      "dynamic_query")
check("#8b AP police structure",
      "AP police organization structure",
      "dynamic_query")
check("#8c hierarchy of Guntur (existing)",
      "Hierarchy of Guntur district",
      "get_unit_hierarchy",
      {"district_name": "Guntur"})

print("\n=== Edge Case #9: Gender filter ===")
check("#9a female officers in Guntur",
      "Female officers in Guntur",
      "dynamic_query")
check("#9b women SIs",
      "Women SIs in Chittoor",
      "dynamic_query")
check("#9c male constables",
      "Male constables in Guntur",
      "dynamic_query")

print("\n=== Edge Case #10: SDPO-wise distribution ===")
check("#10a SDPO wise distribution",
      "SDPO wise officer distribution in Guntur",
      "dynamic_query")
check("#10b PS wise strength",
      "PS wise strength in Chittoor",
      "dynamic_query")

print("\n=== Edge Case #11: Vacancy (already handled) ===")
check("#11a vacant posts in Guntur",
      "Vacant posts in Guntur",
      "count_vacancies_by_unit_rank",
      {"district_name": "Guntur"})
check("#11b vacancies in Guntur DPO",
      "Vacancies in Guntur DPO",
      "count_vacancies_by_unit_rank")

print("\n=== Edge Case #12: Data quality ===")
check("#12a officers without email",
      "Officers without email in Guntur",
      "dynamic_query")
check("#12b personnel with missing phone",
      "Personnel with missing phone number",
      "dynamic_query")

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
