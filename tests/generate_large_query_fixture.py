import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT_DIR / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from mcp.router import repair_route
from mcp.v1.server_http import route_query_to_tool as route_query_to_tool_v1


QUERIES: list[str] = [
    "list all SI in guntur and chittoor districts and show their contact details",
    "who is the SDPO of kuppam and what is his mobile number",
    "show me units in guntur range and also vacancies in guntur district",
    "how many ranges in AP and list them",
    "show recent transfers in guntur for 15 days and who is current SHO of guntur ps",
    "find all constables in guntur district and which districts do they belong to",
    "show all SIs and for each district give count",
    "where is Addl DGP LO and who heads it",
    "SP of guntur dpo and list all SI under guntur district",
    "list units in guntur range and then who is in charge of arundelpet ps",
    "how many senior officers si and above are in each district and compare guntur vs chittoor",
    "who has designation of SDPO and where is kuppam sdpo",
    "show all SI in guntur district and show for which unit these SI's are attached",
    "show all ASIs in annamayya district and give phone numbers",
    "show all HCs in chittoor district and then list only active personnel",
    "list circle inspectors in guntur district and show their emails",
    "how many personnel are in guntur district and also compare with krishna district",
    "list all units in guntur district and then show hierarchy of guntur district",
    "show hierarchy of chittoor district and list police stations only",
    "find missing village mappings in guntur district and list affected units",
    "which villages are mapped to arundelpet ps and also show count",
    "who was the sho of guntur ps in last 30 days and show contact",
    "show recent transfers for 30 days and group by district",
    "show recent transfers in vijayawada for 7 days and list names",
    "count vacancies in guntur district and show rank wise split",
    "show vacancies in annamayya district and compare with chittoor",
    "list all SI in guntur district and next page and then show details of 2nd one",
    "where is kuppam sdpo and who is in charge there",
    "who is the sp of guntur and show office unit",
    "who is the sp of guntur dpo and what is mobile number",
    "sp of guntur gpo and who was previous officer",
    "who is in charge of guntur traffic ps and give posting history",
    "who has user id 14402876 and show assignment details",
    "is ravi kumar a responsible user and where is he posted",
    "search person ravi kumar and show district and unit",
    "tell me about person a ashok kumar and include contact details",
    "show personnel in guntur ps and list only SI and above",
    "show personnel in dachepalli ps and include user ids",
    "list all units in chittoor district and exclude special wing",
    "show me units in guntur range and then units in vijayawada range",
    "how many ranges in andhra pradesh and list all range units",
    "list all districts and then show unit hierarchy for available districts",
    "available districts and total unit counts for each district",
    "show district wise personnel distribution and compare guntur vs chittoor",
    "show rank wise distribution in guntur district and include totals",
    "how many si are in guntur district and list them",
    "how many constables are in guntur and chittoor districts",
    "show sdpo designation holders and list their districts",
    "who has designation of spdo and show location",
    "show transfer movement in guntur district for last 10 days",
    "show postings and transfers in guntur for 60 days and list affected units",
    "find units without village mapping in chittoor district and show personnel count",
    "where is district police office of guntur and who heads it",
    "where is ig guntur range and show units under it",
    "show hierarchy starting from guntur dpo and include depth 3",
    "show hierarchy of guntur and personnel distribution in same district",
    "list all si in alphabetical order and show district",
    "show all inspectors in guntur district and then compare with prakasam",
    "compare guntur vs chittoor vacancies and show top deficient ranks",
    "show command history for kuppam sdpo in last 90 days",
    "who was in charge of arundelpet ps in last 15 days",
    "show all personnel in guntur district with designation sdpo",
    "show all personnel named kumar in guntur district and list units",
    "find officer with mobile 9000000001 and show current assignment",
    "list all units in guntur district and who is responsible user for each",
    "who is sho of arundelpet ps and list previous two officers",
    "give me si list guntur plus attached unit plus transferred in last 60 days",
    "show all SI in guntur district and which districts do they belong to",
]


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not text:
        return "query"
    return text[:60]


def _build_case(idx: int, query: str) -> dict[str, Any]:
    routed_tool, routed_args = route_query_to_tool_v1(query)
    routed_tool, routed_args = repair_route(query, routed_tool, routed_args)
    return {
        "id": f"case_{idx:03d}_{_slug(query)}",
        "query": query,
        "expected_result": {
            "tool": routed_tool,
            "args_subset": routed_args,
            "absent_keys": [],
            "response_should_contain": [],
            "response_should_not_contain": ["Error code:"],
        },
    }


def main() -> None:
    output_path = Path(__file__).resolve().parent / "query_cases_with_expected_results.json"
    cases = [_build_case(idx + 1, query) for idx, query in enumerate(QUERIES)]
    output_path.write_text(json.dumps(cases, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
