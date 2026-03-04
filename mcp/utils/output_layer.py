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

    if asks_json:
        detected_format = "json"
    elif asks_tree:
        detected_format = "tree"
    elif asks_text:
        detected_format = "text"

    if normalized in {"text", "json", "tree"}:
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

    if output_format == "json":
        rendered = json.dumps(response_bundle, indent=2, default=str)
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

    return {
        "format": output_format,
        "rendered": rendered,
        "download": {
            "enabled": download_enabled,
            "filename": filename,
            "content_type": content_type,
            "content": rendered if download_enabled else None,
        },
    }
