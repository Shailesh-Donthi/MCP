"""Entity extraction helpers for natural language routing."""

from difflib import get_close_matches
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re


COMMON_QUERY_TYPO_MAP: Dict[str, str] = {
    "personell": "personnel",
    "disctrict": "district",
    "disctricts": "districts",
    "heirarchy": "hierarchy",
    "hierachy": "hierarchy",
    "villlage": "village",
    "whch": "which",
    "spdo": "sdpo",
    # Data alias / common misspelling observed in field usage
    "arunelpet": "arundelpet",
}

RANK_PATTERNS: List[Tuple[str, str]] = [
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

_PLACE_FILLER_WORDS = {
    "give", "me", "info", "about", "details", "detail", "show", "list", "get",
    "what", "which", "is", "are", "the", "on", "in", "for", "of", "at",
    "by", "from", "starting", "start", "between", "across", "both", "same",
    "who", "all",
    "unit", "units", "station", "stations", "wing", "wings",
    "district", "districts", "disctrict", "disctricts", "dist", "dist.", "present", "there", "here", "please",
    "hierarchy", "heirarchy", "structure", "tree", "organization", "organizational",
    "personnel", "personell", "officers", "officer", "staff", "people", "how", "many",
    "distribution", "distributed", "visual", "representation", "chart", "graph", "plot",
    "rank", "ranks", "wise", "ratio", "compare", "comparison", "versus", "vs",
    "police", "ap",
    "sdpo", "spdo", "sho", "head", "incharge", "charge", "dpo", "reporting",
    "ps",
    "work", "working",
    "si", "sis", "asi", "hc", "pc",
    "inspector", "inspectors", "constable", "constables",
    "circle", "sub", "sub-inspector", "subinspector",
    "date", "birth", "earliest", "oldest", "latest",
    "alphabetical", "alphabetic", "ascending", "descending",
    "sorted", "order", "a-z", "z-a", "asc", "desc",
}


def normalize_common_query_typos(text: str) -> str:
    if not text:
        return text
    normalized = str(text)
    for typo, canonical in COMMON_QUERY_TYPO_MAP.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", canonical, normalized, flags=re.IGNORECASE)
    return normalized


def normalize_common_entity_aliases(text: str) -> str:
    """Normalize common domain aliases/misspellings while preserving user wording."""
    return normalize_common_query_typos(text or "")


def fuzzy_best_match(
    candidate: str,
    choices: Iterable[str],
    *,
    cutoff: float = 0.84,
) -> Optional[str]:
    raw = re.sub(r"\s+", " ", str(candidate or "").strip())
    if not raw:
        return None
    normalized_to_original: Dict[str, str] = {}
    normalized_choices: List[str] = []
    for item in choices:
        if not isinstance(item, str):
            continue
        original = item.strip()
        if not original:
            continue
        norm = re.sub(r"[^a-z0-9]+", " ", normalize_common_query_typos(original).lower()).strip()
        if not norm:
            continue
        if norm not in normalized_to_original:
            normalized_to_original[norm] = original
            normalized_choices.append(norm)
    if not normalized_choices:
        return None
    probe = re.sub(r"[^a-z0-9]+", " ", normalize_common_query_typos(raw).lower()).strip()
    if not probe:
        return None
    matches = get_close_matches(probe, normalized_choices, n=1, cutoff=cutoff)
    if not matches:
        return None
    return normalized_to_original.get(matches[0])


def _normalize_place_candidate(raw: str) -> Optional[str]:
    words = [w for w in re.split(r"\s+", str(raw or "").strip()) if w]
    words = [w for w in words if w.lower() not in _PLACE_FILLER_WORDS]
    if not words:
        return None
    if len(words) > 5:
        words = words[-4:]
    value = " ".join(words).strip()
    if not value:
        return None
    if re.search(
        r"\b(personnel|personell|distribution|visual|representation|chart|graph|plot|birth|starting\s+from|alphabetical|alphabetic|ascending|descending|sorted?|order|a-z|z-a|asc|desc)\b",
        value,
        re.IGNORECASE,
    ):
        return None
    if value.lower() in {"db", "database", "the db", "the database", "birth", "starting", "from"}:
        return None
    return value.title()


def extract_place_hints(text: str, *, max_results: int = 4) -> List[str]:
    text = normalize_common_query_typos(text or "")
    if not text:
        return []

    found: List[str] = []

    def add_candidate(raw: str) -> None:
        normalized = _normalize_place_candidate(raw)
        if not normalized:
            return
        if " and " in normalized.lower():
            parts = [p.strip() for p in re.split(r"\band\b", normalized, flags=re.IGNORECASE) if p and p.strip()]
            if len(parts) >= 2:
                for part in parts:
                    add_candidate(part)
                return
        if normalized.lower() in {x.lower() for x in found}:
            return
        found.append(normalized)

    for match in re.finditer(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,5})\s+(?:district|dist\.?)\b", text, re.IGNORECASE):
        add_candidate(match.group(1))
        if len(found) >= max_results:
            return found

    # Rank/role phrasing fallback: "who is the SP of guntur"
    # Keep this narrow so generic "of <name>" person queries are not treated as places.
    role_place_pattern = (
        r"\b(?:who\s+is\s+(?:the\s+)?)?"
        r"(?:superintendent\s+of\s+police|deputy\s+superintendent\s+of\s+police|"
        r"circle\s+inspector|inspector|constable|si|asi|hc|pc|dsp|dysp|sp)"
        r"\s+of\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\s+district)?(?=\s*(?:\?|,|\.|$))"
    )
    for match in re.finditer(role_place_pattern, text, re.IGNORECASE):
        candidate = match.group(1)
        if re.search(
            r"\b(ps|police\s+station|station|sdpo|spdo|dpo|range|circle|wing|unit)\b",
            candidate,
            re.IGNORECASE,
        ):
            continue
        add_candidate(candidate)
        if len(found) >= max_results:
            return found

    connectors = [
        r"\bbetween\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+and\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\s+districts?)?(?=[\s\?\.,]|$)",
        r"\bboth\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+and\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\s+districts?)?(?=[\s\?\.,]|$)",
        r"\b(?:in|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+and\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+districts?\b",
        r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\s+(?:vs\.?|versus)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\s+districts?)?(?=[\s\?\.,]|$)",
    ]
    for pattern in connectors:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            add_candidate(match.group(1))
            add_candidate(match.group(2))
            if len(found) >= max_results:
                return found

    for pattern in [r"\b(?:in|for|at|on|under)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:\?|$|,|\.)"]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            add_candidate(match.group(1))
            if len(found) >= max_results:
                return found

    return found


def extract_place_hint(text: str) -> Optional[str]:
    candidates = extract_place_hints(text, max_results=1)
    return candidates[0] if candidates else None


def extract_rank_hints(text: str) -> List[str]:
    text = normalize_common_query_typos(text or "")
    if not text:
        return []
    hits: List[Tuple[int, str]] = []
    for pattern, rank in RANK_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            hits.append((match.start(), rank))
    if not hits:
        return []
    hits.sort(key=lambda item: item[0])
    ordered: List[str] = []
    for _, rank in hits:
        if rank not in ordered:
            ordered.append(rank)
    return ordered


def extract_rank_hint(text: str) -> Optional[str]:
    hints = extract_rank_hints(text)
    return hints[0] if hints else None


def extract_user_id_hint(text: str) -> Optional[str]:
    text = normalize_common_query_typos(text or "")
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
    text = normalize_common_query_typos(text or "")
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
    text = normalize_common_query_typos(text)
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
    text = normalize_common_query_typos(text)
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
    text = normalize_common_query_typos(text)
    pattern = rf"^\s*{index_1_based}\.\s*(?:User\s*)?(\d{{6,12}})\b"
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def extract_person_hint(text: str) -> Optional[str]:
    text = normalize_common_query_typos(text or "")
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
    text = normalize_common_query_typos(text or "")
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
    q = normalize_common_query_typos(text or "").lower()
    if not (
        re.search(r"\b(their|them|they|those|these)\b", q)
        and re.search(r"\bdistricts?\b", q)
    ):
        return False

    # If user is asking for contact/details/rank retrieval, do not collapse to
    # district-only follow-up formatting.
    if re.search(r"\b(contact|mobile|phone|email|dob|address|rank|designation|details?)\b", q):
        return False

    followup_patterns = [
        r"\b(which|what)\s+districts?\s+do\s+(they|those|these)\s+belong\s+to\b",
        r"\b(they|those|these)\s+belong\s+to\s+(which|what)\s+districts?\b",
        r"\bdistricts?\s+(they|those|these)\s+belong\s+to\b",
        r"\b(which|what)\s+districts?\s+are\s+(they|those|these)\s+from\b",
        r"\b(?:districts?\s+of\s+(them|those|these))\b",
    ]
    return any(re.search(pattern, q) for pattern in followup_patterns)


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
