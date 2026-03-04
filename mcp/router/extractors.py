"""Entity extraction helpers for natural language routing."""

from typing import Any, Dict, List, Optional, Tuple
import re


def extract_place_hint(text: str) -> Optional[str]:
    filler_words = {
        "give", "me", "info", "about", "details", "detail", "show", "list",
        "what", "which", "is", "are", "the", "on", "in", "for", "of", "at",
        "by",
        "who", "all",
        "unit", "units", "station", "stations",
        "district", "dist", "dist.", "present", "there", "here", "please",
        "hierarchy", "heirarchy", "structure", "tree", "organization",
        "personnel", "personell", "officers", "officer", "staff", "people", "how", "many",
        "distribution", "distributed", "visual", "representation", "chart", "graph", "plot",
        "rank", "ranks", "wise",
        "police", "ap",
        "sdpo", "spdo", "sho", "head", "incharge", "charge",
        "si", "sis", "asi", "hc", "pc",
        "inspector", "inspectors", "constable", "constables",
        "circle", "sub", "sub-inspector", "subinspector",
    }

    def normalize_place(raw: str) -> Optional[str]:
        words = [w for w in re.split(r"\s+", raw.strip()) if w]
        words = [w for w in words if w.lower() not in filler_words]
        if not words:
            return None
        if len(words) > 3:
            words = words[-2:]
        value = " ".join(words).strip()
        # Reject obvious non-place fragments that can leak in from generic queries
        # like "personell distribution" or visual/chart prompts.
        if re.search(r"\b(personnel|personell|distribution|visual|representation|chart|graph|plot)\b", value, re.IGNORECASE):
            return None
        if value.lower() in {"db", "database", "the db", "the database"}:
            return None
        return value.title()

    district_matches = list(
        re.finditer(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:district|dist\.?)\b", text, re.IGNORECASE)
    )
    if district_matches:
        normalized = normalize_place(district_matches[-1].group(1))
        if normalized:
            return normalized

    for pattern in [r"\b(?:in|for|of|at|on)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\?|$)"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            normalized = normalize_place(match.group(1))
            if normalized:
                return normalized
    return None


def extract_rank_hint(text: str) -> Optional[str]:
    rank_patterns: List[Tuple[str, str]] = [
        (r"\bcircle\s+inspectors?\b", "Circle Inspector"),
        (r"\binspector\s+general\s+of\s+police\b", "Inspector General of Police"),
        (r"\bdeputy\s+superintendent\s+of\s+police\b", "Deputy Superintendent of Police"),
        (r"\bsuperintendent\s+of\s+police\b", "Superintendent of Police"),
        (r"\bassistant\s+sub[-\s]?inspectors?\b", "Assistant SubInspector"),
        (r"\bsub[-\s]?inspectors?\b", "Sub Inspector"),
        (r"\binspector\s+of\s+police\b", "Inspector Of Police"),
        (r"\bpolice\s+constables?\b", "Police Constable"),
        (r"\bhead\s+constables?\b", "Head Constable"),
        (r"\bconstables?\b", "Constable"),
        (r"\binspectors?\b", "Inspector"),
        (r"\bsi\b", "Sub-Inspector"),
        (r"\bsis\b", "Sub-Inspector"),
        (r"\basi\b", "Assistant Sub-Inspector"),
        (r"\bhc\b", "Head Constable"),
        (r"\bpc\b", "Police Constable"),
        (r"\bdysp\b", "Deputy Superintendent of Police"),
        (r"\bdsp\b", "Deputy Superintendent of Police"),
        (r"\bsp\b", "Superintendent of Police"),
    ]
    for pattern, rank in rank_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return rank
    return None


def extract_user_id_hint(text: str) -> Optional[str]:
    for pattern in [
        r"\buser\s*[-_:]?\s*(\d{6,12})\b",
        r"\buser\s*id\s*[:\-]?\s*(\d{6,12})\b",
        r"\buid\s*[:\-]?\s*(\d{6,12})\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def extract_mobile_hint(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = re.findall(r"(?:\+?\d[\d\s\-]{8,16}\d)", text)
    for raw in candidates:
        normalized = re.sub(r"[^\d+]", "", raw)
        digits_only = re.sub(r"[^\d]", "", normalized)
        if len(digits_only) >= 10:
            return normalized
    m = re.search(r"\b\d{10,14}\b", text)
    if m:
        return m.group(0)
    return None


def extract_ordinal_index(text: str) -> Optional[int]:
    if not text:
        return None
    ordinal_words = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    for word, idx in ordinal_words.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            return idx
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_list_reference_index(text: str) -> Optional[int]:
    if not text:
        return None
    for pattern in [
        r"\b(?:about|of|on|for)\s+#?(\d{1,2})\b",
        r"\b(?:item|number|no\.?)\s+#?(\d{1,2})\b",
        r"\b#(\d{1,2})\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if idx >= 1:
                return idx
    return None


def extract_user_id_from_list_item(text: str, index_1_based: int) -> Optional[str]:
    if not text or not index_1_based or index_1_based < 1:
        return None
    pattern = rf"^\s*{index_1_based}\.\s*(?:User\s*)?(\d{{6,12}})\b"
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def extract_person_hint(text: str) -> Optional[str]:
    patterns = [
        r"\bwho\s+is\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\btell\s+me\s+about\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:mobile|phone|contact)\s+(?:number\s+)?(?:of|for)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:email|e-?mail|dob|date\s+of\s+birth|birthday|address|blood\s+group)\s+(?:of|for)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:with\s+(?:the\s+)?name|name\s+is|named)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:all\s+)?person(?:nel)?\s+with\s+(?:the\s+)?name\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:show|get|give|provide)\s+details?\s+(?:of|about)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:give|show|get|provide)\s+me\s+(?:info|details?)\s+(?:on|about)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
        r"\b(?:info|details?)\s+(?:on|about)\s+([A-Za-z][A-Za-z\s\.\-']{1,60})(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip()
            value = re.sub(r"^(?:the|an)\s+", "", value, flags=re.IGNORECASE).strip()
            # Reject role-holder phrases like "SPDO of Kuppam" (not a person name).
            if re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge|responsible\s*user|head)\b", value, re.IGNORECASE):
                continue
            if re.search(r"\bof\s+[A-Za-z]", value, re.IGNORECASE):
                continue
            # Reject place/catalog nouns so generic "info on all districts/units"
            # does not get misclassified as a person-name lookup.
            if value and not re.search(
                r"\b(?:dist(?:ricts?)?|station(?:s)?|ps|unit(?:s)?|village(?:s)?|rank(?:s)?)\b",
                value,
                re.IGNORECASE,
            ):
                return value
    return None


def extract_unit_hint(text: str) -> Optional[str]:
    patterns = [
        r"(?:which\s+villages?.*?(?:mapped|assigned|covered)\s+(?:to|in|for|by)|village\s+(?:coverage|mapping)\s+for)\s+([A-Za-z0-9\.\-\s]+?\s+(?:ps|police station|station|dpo|circle|sdpo|spdo|range|ups))(?:\s*\([^)]*\))?(?:\?|$)",
        r"(?:where\s+is|search\s+unit|find\s+unit|locate\s+unit)\s+([A-Za-z0-9\.\-\s]+?\s+(?:ps|police station|station|dpo|circle|sdpo|spdo|range|ups))(?:\s*\([^)]*\))?(?:\?|$)",
        r"(?:info|details?|about|on|for)\s+([A-Za-z0-9\.\-\s]+?\s+(?:ps|police station|station|dpo|circle|sdpo|spdo|range))(?:\s*\([^)]*\))?(?:\?|$)",
        r"([A-Za-z0-9\.\-\s]+?\s+(?:ps|police station|station|dpo|circle|sdpo|spdo|range))(?:\s*\([^)]*\))?(?:\?|$)",
        r"(?:in|at|under)\s+([A-Za-z0-9\.\-\s]+?\s+(?:ps|police station|station))(?:\s*\([^)]*\))?(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            value = re.sub(r"^(?:on|in|at|for|of)\s+", "", value, flags=re.IGNORECASE).strip()
            value = re.sub(
                r"^(?:which\s+villages?.*?(?:mapped|assigned|covered)\s+(?:to|in|for|by)|village\s+(?:coverage|mapping)\s+for|where\s+is|search\s+unit|find\s+unit|locate\s+unit)\s+",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
            value = re.sub(r"\b(the|a|an)\b", "", value, flags=re.IGNORECASE).strip()
            if value:
                return value
    return None


def is_followup_district_query(text: str) -> bool:
    q = (text or "").lower()
    return bool(
        re.search(r"\b(their|them|they|those|these)\b", q)
        and re.search(r"\bdistricts?\b", q)
        and re.search(r"\b(belong|belongs|belonging|in|of)\b", q)
    )


def district_from_person_record(person: Dict[str, Any]) -> Optional[str]:
    candidates: List[Optional[str]] = [
        person.get("districtName"),
        (person.get("district") or {}).get("name") if isinstance(person.get("district"), dict) else None,
        (person.get("unit") or {}).get("districtName") if isinstance(person.get("unit"), dict) else None,
        person.get("district"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def format_followup_district_response(data: Any) -> Optional[str]:
    if not isinstance(data, list) or not data:
        return None
    district_counts: Dict[str, int] = {}
    for person in data:
        if not isinstance(person, dict):
            continue
        district = district_from_person_record(person)
        if district:
            district_counts[district] = district_counts.get(district, 0) + 1
    if not district_counts:
        return "I couldn't determine district information for these personnel from the available data."
    sorted_items = sorted(district_counts.items(), key=lambda item: item[1], reverse=True)
    lines = ["These personnel belong to the following districts:", ""]
    for idx, (district, count) in enumerate(sorted_items, 1):
        lines.append(f"{idx}. {district} ({count})")
    return "\n".join(lines)
