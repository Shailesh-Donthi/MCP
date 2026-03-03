"""Output formatting layer for natural-language responses."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

OutputFormat = str


def resolve_output_preferences(
    query: str,
    requested_format: Optional[str] = None,
    allow_download: Optional[bool] = None,
) -> Tuple[OutputFormat, bool]:
    """
    Resolve output format and download preference.

    Priority:
    1. Explicit request payload values.
    2. Natural-language hints in query.
    3. Defaults: text, no download.
    """
    normalized = (requested_format or "auto").strip().lower()
    query_lower = (query or "").lower()

    detected_format = "text"
    asks_json = bool(
        re.search(
            r"\b(?:show|display|format|convert|render|give)\b[\w\s]*\b(?:in|as)\b[\w\s]*\bjson\b",
            query_lower,
        )
        or re.search(r"\bjson\s+format\b", query_lower)
    )
    asks_tree = bool(
        re.search(
            r"\b(?:show|display|format|convert|render|give)\b[\w\s]*\b(?:in|as)\b[\w\s]*\b(?:tree|tree\s+view)\b",
            query_lower,
        )
        or re.search(r"\btree\s+format\b", query_lower)
    )
    asks_text = bool(
        re.search(
            r"\b(?:show|display|format|convert|render|give)\b[\w\s]*\b(?:in|as)\b[\w\s]*\b(?:text|txt|plain\s+text)\b",
            query_lower,
        )
        or re.search(r"\b(?:text|txt|plain\s+text)\s+format\b", query_lower)
    )

    asks_table = bool(
        re.search(
            r"\b(?:show|display|format|convert|render|give)\b[\w\s]*\b(?:in|as)\b[\w\s]*\b(?:table|tabular)\b",
            query_lower,
        )
        or re.search(r"\b(?:table|tabular)\s+format\b", query_lower)
    )
    asks_chart = bool(
        re.search(r"\b(?:chart|graph|plot|visual(?:ize|ise|ization|isation)?|representation)\b", query_lower)
    )
    chart_type = (
        "pie"
        if re.search(r"\bpie\b", query_lower)
        else "line"
        if re.search(r"\bline\b", query_lower)
        else "bar"
    )

    if asks_json:
        detected_format = "json"
    elif asks_table:
        detected_format = "table"
    elif asks_chart:
        detected_format = chart_type
    elif asks_tree:
        detected_format = "tree"
    elif asks_text:
        detected_format = "text"

    if normalized in {"text", "json", "tree", "table", "bar", "line", "pie"}:
        final_format = normalized
    elif normalized == "auto":
        final_format = detected_format
    else:
        final_format = "text"

    if allow_download is None:
        final_download = bool(
            re.search(
                r"\b(download|export|save (?:it|this|output)|file)\b",
                query_lower,
            )
        )
    else:
        final_download = bool(allow_download)

    return final_format, final_download


def _to_tree_lines(
    value: Any,
    name: str,
    prefix: str = "",
    is_last: bool = True,
    depth: int = 0,
    max_depth: int = 6,
    max_items: int = 20,
) -> List[str]:
    connector = "`- " if is_last else "|- "
    lines: List[str] = [f"{prefix}{connector}{name}"]

    if depth >= max_depth:
        lines.append(f"{prefix}{'   ' if is_last else '|  '}`- ...")
        return lines

    child_prefix = prefix + ("   " if is_last else "|  ")

    if isinstance(value, dict):
        items = list(value.items())
        if not items:
            lines[-1] = f"{prefix}{connector}{name}: {{}}"
            return lines
        if len(items) > max_items:
            items = items[:max_items]
            truncated = True
        else:
            truncated = False
        for idx, (k, v) in enumerate(items):
            is_child_last = idx == len(items) - 1 and not truncated
            if isinstance(v, (dict, list)):
                lines.extend(
                    _to_tree_lines(
                        v,
                        str(k),
                        child_prefix,
                        is_child_last,
                        depth + 1,
                        max_depth,
                        max_items,
                    )
                )
            else:
                branch = "`- " if is_child_last else "|- "
                lines.append(f"{child_prefix}{branch}{k}: {v!r}")
        if truncated:
            lines.append(f"{child_prefix}`- ... ({len(value) - max_items} more keys)")
        return lines

    if isinstance(value, list):
        if not value:
            lines[-1] = f"{prefix}{connector}{name}: []"
            return lines
        items = value[:max_items]
        for idx, item in enumerate(items):
            is_child_last = idx == len(items) - 1 and len(value) <= max_items
            item_name = f"[{idx}]"
            if isinstance(item, (dict, list)):
                lines.extend(
                    _to_tree_lines(
                        item,
                        item_name,
                        child_prefix,
                        is_child_last,
                        depth + 1,
                        max_depth,
                        max_items,
                    )
                )
            else:
                branch = "`- " if is_child_last else "|- "
                lines.append(f"{child_prefix}{branch}{item_name}: {item!r}")
        if len(value) > max_items:
            lines.append(f"{child_prefix}`- ... ({len(value) - max_items} more items)")
        return lines

    lines[-1] = f"{prefix}{connector}{name}: {value!r}"
    return lines


def to_tree_text(value: Any, root_name: str = "result") -> str:
    """Render dict/list-like data as an ASCII tree."""
    lines = _to_tree_lines(value, root_name, prefix="", is_last=True)
    return "\n".join(lines)


def _get_base_data(result: Dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return result
    return result.get("data", result)


def _rows_for_table(base: Any, max_rows: int = 50, max_cols: int = 12) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows = base if isinstance(base, list) else [base]
    object_rows = [r for r in rows if isinstance(r, dict)]
    if not object_rows:
        return [], []

    headers: List[str] = []
    for row in object_rows[:max_rows]:
        for key in row.keys():
            if key not in headers:
                headers.append(str(key))
                if len(headers) >= max_cols:
                    break
        if len(headers) >= max_cols:
            break
    return headers, object_rows[:max_rows]


def to_table_text(base: Any) -> str:
    headers, rows = _rows_for_table(base)
    if not headers or not rows:
        # Fall back to JSON when tabular projection is not obvious.
        return json.dumps(base, indent=2, default=str)

    str_rows: List[List[str]] = []
    for row in rows:
        values: List[str] = []
        for h in headers:
            value = row.get(h)
            if isinstance(value, (dict, list)):
                text = json.dumps(value, default=str)
            else:
                text = "" if value is None else str(value)
            values.append(text.replace("\n", " "))
        str_rows.append(values)

    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = min(max(widths[idx], len(cell)), 80)

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(
            (
                (cell[: widths[i] - 3] + "...")
                if len(cell) > widths[i] and widths[i] >= 4
                else (cell[: widths[i]] if len(cell) > widths[i] else cell)
            ).ljust(widths[i])
            for i, cell in enumerate(cells)
        )

    header_line = fmt_row(headers)
    divider = "-+-".join("-" * w for w in widths)
    body_lines = [fmt_row(r) for r in str_rows]
    return "\n".join([header_line, divider, *body_lines])


def _extract_chart_series(base: Any) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    if isinstance(base, dict) and isinstance(base.get("distribution"), list):
        for item in base["distribution"]:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("rankName")
                or item.get("districtName")
                or item.get("unitTypeName")
                or item.get("name")
                or "Unknown"
            )
            value = item.get("count", 0)
            try:
                n = float(value)
            except Exception:
                continue
            if n >= 0:
                series.append({"label": str(label), "value": n})
    elif isinstance(base, list):
        for item in base:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("rankName")
                or item.get("districtName")
                or item.get("unitTypeName")
                or item.get("name")
                or item.get("label")
                or "Unknown"
            )
            value = (
                item.get("count")
                if item.get("count") is not None
                else item.get("value")
                if item.get("value") is not None
                else item.get("personnelCount")
                if item.get("personnelCount") is not None
                else item.get("totalPersonnel")
                if item.get("totalPersonnel") is not None
                else item.get("villageCount")
            )
            try:
                n = float(value) if value is not None else None
            except Exception:
                n = None
            if n is not None and n >= 0:
                series.append({"label": str(label), "value": n})
    elif isinstance(base, dict):
        for key, value in base.items():
            if isinstance(value, (int, float)):
                series.append({"label": str(key), "value": float(value)})
    return series


def to_chart_payload(base: Any, chart_type: str = "bar") -> Dict[str, Any]:
    series = _extract_chart_series(base)
    # Keep payload small and stable.
    series = sorted(series, key=lambda item: item["value"], reverse=True)[:20]
    return {
        "chart_type": chart_type,
        "series": series,
        "total": sum(item["value"] for item in series),
    }


def build_output_payload(
    query: str,
    response_text: str,
    routed_to: Optional[str],
    arguments: Dict[str, Any],
    result: Dict[str, Any],
    requested_format: Optional[str] = None,
    allow_download: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build formatted output payload for API responses."""
    output_format, download_enabled = resolve_output_preferences(
        query=query,
        requested_format=requested_format,
        allow_download=allow_download,
    )

    response_bundle: Dict[str, Any] = {
        "query": query,
        "routed_to": routed_to,
        "arguments": arguments,
        "response": response_text,
        "data": result,
    }

    chart_payload: Optional[Dict[str, Any]] = None
    base_data = _get_base_data(result)

    if output_format == "json":
        rendered = json.dumps(response_bundle, indent=2, default=str)
        content_type = "application/json"
        extension = "json"
    elif output_format == "table":
        rendered = to_table_text(base_data)
        content_type = "text/plain"
        extension = "txt"
    elif output_format in {"bar", "line", "pie"}:
        chart_payload = to_chart_payload(base_data, chart_type=output_format)
        rendered = json.dumps(chart_payload, indent=2, default=str)
        content_type = "application/json"
        extension = "json"
    elif output_format == "tree":
        rendered = to_tree_text(result.get("data", result), root_name="data")
        content_type = "text/plain"
        extension = "txt"
    else:
        rendered = response_text
        content_type = "text/plain"
        extension = "txt"

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_tool = (routed_to or "query_result").replace(" ", "_")
    filename = f"{safe_tool}_{stamp}.{extension}"

    payload = {
        "format": output_format,
        "rendered": rendered,
        "download": {
            "enabled": download_enabled,
            "filename": filename,
            "content_type": content_type,
            "content": rendered if download_enabled else None,
        },
    }
    if chart_payload is not None:
        payload["chart"] = chart_payload
    return payload
