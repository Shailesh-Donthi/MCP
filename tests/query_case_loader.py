import json
from pathlib import Path
from typing import Any


def load_query_cases_with_expected_results() -> list[dict[str, Any]]:
    fixture_path = Path(__file__).resolve().parent / "query_cases_with_expected_results.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("query_cases_with_expected_results.json must contain a top-level list")
    return payload
