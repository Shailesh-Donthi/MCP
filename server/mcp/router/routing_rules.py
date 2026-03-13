"""Rule-based route repair and clarification checks."""

import re
from typing import Any, Dict, Optional, Tuple

from mcp.router.extractors import (
    extract_place_hint,
    extract_place_hints,
    extract_unit_hint,
    extract_rank_hint,
    extract_rank_hints,
    extract_user_id_hint,
    extract_mobile_hint,
    extract_person_hint,
    extract_ordinal_index,
    extract_list_reference_index,
    extract_user_id_from_list_item,
    is_followup_district_query,
    format_followup_district_response,
)
from mcp.tools.master_data_tools import infer_master_collection_from_query


def repair_route(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    last_user_query: Optional[str] = None,
    last_assistant_response: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    hierarchy_intent_pattern = r"\b(?:hierarchy|heirarchy|hierachy|structure|tree|organization)\b"

    def _strip_temporal_tail(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return text
        # Trim trailing temporal qualifiers from unit-like phrases:
        # e.g. "Guntur PS in last 15 days" -> "Guntur PS"
        text = re.sub(
            r"\s+(?:in|for|during|within)\s+(?:the\s+)?(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(
            r"\s+(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(r"\s+\b(?:in|for|during|within)\b\s*$", "", text, flags=re.IGNORECASE).strip()
        return text

    def _extract_primary_clause(text: str) -> str:
        raw = re.sub(r"\s+", " ", str(text or "")).strip()
        if not raw:
            return ""
        # Split only when conjunction starts a new intent/action clause.
        parts = re.split(
            r"\s+\b(?:and\s+then|then|and\s+also|also|and)\b\s+(?="
            r"(?:show|list|get|give|provide|find|where|who|what|which|how|compare|count|next|previous|details?|info|"
            r"in[\s-]?charge|heads?|belongs?)\b"
            r")",
            raw,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        return parts[0].strip() if parts else raw

    def _trim_conjoined_tail(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return text
        text = re.sub(
            r"\s+\b(?:and\s+then|then|and\s+also|also|and)\b\s+"
            r"(?:show|list|get|give|provide|find|where|who|what|which|how|compare|count|next|previous|details?|info|"
            r"in[\s-]?charge|heads?|belongs?)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return text

    def _extract_range_place(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"\b(?:in|for|of|under)\s+([A-Za-z][A-Za-z\s]{1,60}?)\s+ranges?\b",
            r"\b([A-Za-z][A-Za-z\s]{1,60}?)\s+ranges?\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = _trim_conjoined_tail(m.group(1))
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                return value
        return None

    def _extract_compare_places(text: str) -> list[str]:
        if not text:
            return []
        places: list[str] = []
        patterns = [
            r"\bcompare\s+([A-Za-z][A-Za-z\s]{1,50}?)\s+(?:vs|versus)\s+([A-Za-z][A-Za-z\s]{1,50})(?:\b|$)",
            r"\b([A-Za-z][A-Za-z\s]{1,50}?)\s+(?:vs|versus)\s+([A-Za-z][A-Za-z\s]{1,50})(?:\b|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            for idx in (1, 2):
                candidate = re.sub(r"\s+", " ", m.group(idx)).strip(" ?.")
                candidate = _trim_conjoined_tail(candidate)
                candidate = _sanitize_district_value(candidate)
                if candidate and candidate not in places:
                    places.append(candidate)
            if places:
                break
        return places

    def _extract_transfer_place(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"\b(?:transfers?|postings?|movement)\s+(?:in|for|of|under)\s+([A-Za-z][A-Za-z\s]{1,60}?)(?:\s+district)?(?:\s+for\s+\d+\s+days?)?(?:\b|$)",
            r"\b(?:in|for|of|under)\s+([A-Za-z][A-Za-z\s]{1,60}?)(?:\s+district)?\s+(?:for|in)\s+\d+\s+days?\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = re.sub(r"\s+", " ", m.group(1)).strip(" ?.")
            value = _trim_conjoined_tail(value)
            value = _sanitize_district_value(value)
            if value:
                return value
        return None

    def _extract_bare_info_subject(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"^\s*(?:info|details?|about)\s+([A-Za-z][A-Za-z0-9\s\.\-']{1,80})(?:\?|$)",
            r"^\s*(?:show|get|give|provide)\s+(?:info|details?)\s+(?:for|on|about)?\s*([A-Za-z][A-Za-z0-9\s\.\-']{1,80})(?:\?|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = re.sub(r"\s+", " ", m.group(1)).strip(" ?.")
            value = _trim_conjoined_tail(value)
            value = re.sub(r"^(?:on|about|for|of)\s+", "", value, flags=re.IGNORECASE).strip()
            if not value:
                continue
            if re.search(r"\b(?:sho|sdpo|spdo|ps|station|unit|mobile|phone|email|dob|address)\b", value, re.IGNORECASE):
                continue
            return value
        return None

    def _canonicalize_role_unit(role: str, raw_target: str) -> Optional[str]:
        def _smart_title(text: str) -> str:
            acronyms = {"PS", "SDPO", "SPDO", "DPO", "GPO", "IGP", "AO", "PC", "SI", "ASI", "HC"}
            words = [w for w in re.split(r"\s+", text or "") if w]
            normalized: list[str] = []
            for word in words:
                plain = re.sub(r"[^A-Za-z]", "", word).upper()
                if plain in acronyms:
                    normalized.append(plain)
                elif re.fullmatch(r"[A-Z]{2,5}", word):
                    normalized.append(word)
                else:
                    normalized.append(word.capitalize())
            return " ".join(normalized).strip()

        target = re.sub(r"\s+", " ", (raw_target or "")).strip(" ?")
        target = _strip_temporal_tail(target)
        target = _trim_conjoined_tail(target)
        if not target:
            return None
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        if not target:
            return None

        # Avoid obviously bad captures like "who is" or whole-sentence fragments.
        if re.search(r"\b(?:who|what|name|of|is)\b", target, re.IGNORECASE) and not re.search(
            r"\b(?:ps|station|sdpo|spdo|sho)\b", target, re.IGNORECASE
        ):
            return None

        if role in {"sdpo", "spdo"}:
            target = re.sub(r"\bSPDO\b", "SDPO", target, flags=re.IGNORECASE)
            if not re.search(r"\bSDPO\b", target, re.IGNORECASE):
                target = f"{target} SDPO"
            target = re.sub(r"\bSDPO\b(?:\s+\bSDPO\b)+", "SDPO", target, flags=re.IGNORECASE)
        elif role == "sho":
            # "SHO of <station>" maps to command history for the unit (PS).
            if not re.search(r"\b(?:ps|police\s+station|station)\b", target, re.IGNORECASE):
                target = f"{target} PS"
            target = re.sub(r"\bPS\b(?:\s+\bPS\b)+", "PS", target, flags=re.IGNORECASE)

        return _smart_title(re.sub(r"\s+", " ", target).strip())

    def _extract_role_unit_candidate(text: str) -> Optional[str]:
        if not text:
            return None
        # Forms: "who is the SDPO of Kuppam", "who is spdo kuppam", "spdo kuppam"
        patterns = [
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)?\s*(?:the\s+)?(sho|sdpo|spdo|in[\s-]?charge)\s+of\s+([A-Za-z0-9\.\-\s]+?)(?:\s+district)?(?:\?|$)",
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)?\s*(?:the\s+)?(sho|sdpo|spdo)\s+([A-Za-z0-9\.\-\s]+?)(?:\s+district)?(?:\?|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            role = (m.group(1) or "").lower().replace(" ", "").replace("-", "")
            role = "incharge" if role.startswith("in") else role
            candidate = _canonicalize_role_unit(role, m.group(2))
            if candidate:
                return candidate
        return None

    def _extract_sp_of_unit_candidate(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)?\s*(?:the\s+)?(?:sp|superintendent\s+of\s+police)\s+of\s+([A-Za-z0-9\.\-\s]+?)(?:\s+district)?(?:\?|$)",
            r"^\s*(?:the\s+)?(?:sp|superintendent\s+of\s+police)\s+([A-Za-z0-9\.\-\s]+?\s+(?:dpo|gpo|sdpo|spdo|ps|police\s+station|station|range|circle|wing|office))(?:\?|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            target = re.sub(r"\s+", " ", (m.group(1) or "")).strip(" ?")
            target = _strip_temporal_tail(target)
            target = _trim_conjoined_tail(target)
            target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
            if not target:
                continue
            if not re.search(
                r"\b(?:dpo|gpo|sdpo|spdo|ps|police\s+station|station|range|circle|wing|office)\b",
                target,
                re.IGNORECASE,
            ):
                continue
            target = re.sub(r"\bSPDO\b", "SDPO", target, flags=re.IGNORECASE)
            target = re.sub(r"\bGPO\b", "DPO", target, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", target).strip()
        return None

    def _extract_where_is_target(text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"^\s*where\s+is\s+([A-Za-z0-9\.\-][A-Za-z0-9\.\-\s]{1,80}?)(?:\?|$)", text, re.IGNORECASE)
        if not m:
            return None
        value = re.sub(r"\s+", " ", m.group(1)).strip(" ?.")
        value = _strip_temporal_tail(value)
        value = _trim_conjoined_tail(value)
        value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE).strip()
        if not value:
            return None
        # Ignore obvious person-attribute asks masquerading as location.
        if re.search(r"\b(email|mobile|phone|contact|dob|address)\b", value, re.IGNORECASE):
            return None
        return value

    def _extract_designation_hint(text: str) -> Optional[str]:
        if not text:
            return None
        canonical_map = {
            "spdo": "SDPO",
            "sdpo": "SDPO",
        }
        patterns = [
            r"\b(?:designation|post|role)\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9\s\.\-']{1,60})(?:\?|$)",
            r"\bwho\s+has\s+(?:the\s+)?(?:designation|post|role)\s+(?:of|for)?\s*([A-Za-z][A-Za-z0-9\s\.\-']{1,60})(?:\?|$)",
            r"\blist\s+(?:all\s+)?([A-Za-z][A-Za-z0-9\s\.\-']{1,40})s?\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = re.sub(r"\s+", " ", m.group(1)).strip(" ?.")
            value = _trim_conjoined_tail(value).lower()
            value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE).strip()
            if not value:
                continue
            # Singularize compact acronym-like tokens (e.g., "spdos" -> "spdo").
            if " " not in value and value.endswith("s") and len(value) <= 10:
                value = value[:-1]
            if re.search(r"\b(district|unit|station|ps)\b", value, re.IGNORECASE):
                continue
            if value in canonical_map:
                return canonical_map[value]
            # Treat acronyms/short forms as uppercase designation labels.
            if re.fullmatch(r"[a-z]{2,8}", value):
                return value.upper()
            return value.title()
        return None

    def _extract_hierarchy_place(text: str) -> Optional[str]:
        if not text:
            return None
        generic_place_words = {"available", "each", "every", "all"}
        patterns = [
            rf"{hierarchy_intent_pattern}\s+(?:of|for|in)\s+([A-Za-z][A-Za-z0-9\s\.\-']{{1,80}})",
            rf"{hierarchy_intent_pattern}\s*[:\-]\s*([A-Za-z][A-Za-z0-9\s\.\-']{{1,80}})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            value = m.group(1)
            value = re.split(r"[,\?\.;]", value, maxsplit=1)[0]
            value = _trim_conjoined_tail(value)
            value = re.sub(r"\b(?:districts?|disctricts?|dist\.?)\b", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\s+", " ", value).strip(" -")
            if not value:
                continue
            if re.search(r"\b(personnel|personell|officers?|staff|people|unit|units)\b", value, re.IGNORECASE):
                continue
            if value.lower() in generic_place_words:
                continue
            return value.title()
        return None

    def _looks_like_bad_unit_hint(value: Optional[str]) -> bool:
        v = (value or "").strip()
        if not v:
            return True
        if re.search(r"^\s*(?:who|what)\s+is\b", v, re.IGNORECASE):
            return True
        if re.search(r"\b(?:sho|sdpo|spdo)\b\s+of\b", v, re.IGNORECASE):
            return True
        if re.search(r"\b(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b", v, re.IGNORECASE):
            return True
        return False

    def _looks_like_sort_phrase(value: Optional[str]) -> bool:
        v = (value or "").strip()
        if not v:
            return False
        return bool(
            re.search(
                r"\b(alphabetical|alphabetic|ascending|descending|sorted?|order|a-z|z-a|asc|desc)\b",
                v,
                re.IGNORECASE,
            )
        )

    def _sanitize_district_value(raw_value: Optional[str]) -> Optional[str]:
        invalid_tokens = {
            "each", "every", "available", "all",
            "senior", "junior", "above", "below",
            "officer", "officers", "personnel", "personell", "staff", "people",
            "unit", "units", "station", "stations",
            "si", "sis", "asi", "asis", "hc", "pc", "dsp", "dysp", "sp",
            "and", "also", "then",
            "them", "those", "these", "this", "that",
            "find", "show", "list", "give", "get", "provide", "tell",
            "count", "compare", "vs", "versus",
            "who", "what", "which", "how",
            "for", "of", "in", "under", "to",
            "many", "total", "number",
            "the", "a", "an",
            "last", "past", "previous", "current", "recent",
            "day", "days", "week", "weeks", "month", "months", "year", "years",
            # Transfer / village / vacancy keywords
            "transfer", "transfers", "posting", "postings", "movement",
            "village", "villages", "mapping", "mappings", "coverage", "unmapped",
            "vacancy", "vacancies", "vacant",
            "missing",
        }

        value = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        value = _trim_conjoined_tail(value)
        value = re.sub(r"\s*(?:districts?|disctricts?|dist\.?)\s*$", "", value, flags=re.IGNORECASE).strip()
        if not value or _looks_like_sort_phrase(value):
            return None
        # Ambiguous multi-place text should be derived from place_hints, not a
        # single district slot.
        if re.search(r"\b(?:and|also|then)\b", value, re.IGNORECASE):
            return None
        if value.lower() in {"db", "database", "the db", "the database"}:
            return None
        tokens = [t for t in re.findall(r"[A-Za-z]+", value) if t]
        if not tokens:
            return None
        filtered = [t for t in tokens if t.lower() not in invalid_tokens]
        if not filtered:
            return None
        if len(filtered) > 5:
            filtered = filtered[-4:]
        normalized = " ".join(filtered).strip()
        if not normalized:
            return None
        if normalized.lower() in {"db", "database", "the db", "the database"}:
            return None
        return normalized.title()

    def _extract_rank_relation(text: str) -> str:
        q = (text or "").lower()
        if re.search(r"\b(at\s*least|at\s*or\s*above|or\s*above|not\s+below)\b", q):
            return "at_or_above"
        if re.search(r"\b(at\s*most|at\s*or\s*below|or\s*below|not\s+above)\b", q):
            return "at_or_below"
        if re.search(r"\b(above|higher\s+than|senior\s+to|greater\s+than)\b", q):
            return "above"
        if re.search(r"\b(below|lower\s+than|junior\s+to|less\s+than)\b", q):
            return "below"
        return "exact"

    query_lower = (query or "").lower()
    tool_aliases = {
        "count_vacancies_by_unit": "count_vacancies_by_unit_rank",
        "count_vacancies": "count_vacancies_by_unit_rank",
        "vacancies_by_unit": "count_vacancies_by_unit_rank",
    }

    primary_query = _extract_primary_clause(query or "")
    primary_query_lower = primary_query.lower()
    tool_name = tool_aliases.get(str(tool_name or "").strip(), tool_name)
    master_collection = infer_master_collection_from_query(query or "")
    asks_master_relations = bool(
        re.search(
            r"\b(interlinked?|relation(ship)?|dependency|dependencies|schema|map\s+between|master\s+data|how\s+.*\s+linked)\b",
            query_lower,
        )
    )
    if master_collection and (
        asks_master_relations
        or re.search(
            r"\b(approval|module|notification|prompt|permissions?|roles?|value\s*set|error|log)\b",
            query_lower,
        )
    ):
        args_master: Dict[str, Any] = {
            "collection": master_collection,
            "include_related": True,
            "include_reverse": True,
        }
        if asks_master_relations:
            args_master["mode"] = "discover"
            args_master["include_integrity"] = True
        return "query_linked_master_data", args_master
    # Only merge previous-turn text for explicit follow-up phrasing. Merging on
    # every new query causes context leakage like "spdo of kuppam" influencing
    # a later standalone query "personell distribution".
    followup_like_query = bool(
        re.search(r"\b(their|them|they|those|these|there|here|that|this|same|also|then)\b", query_lower)
        or extract_ordinal_index(query) is not None
        or extract_list_reference_index(query) is not None
    )
    # Do not merge previous-turn text for markup/tag-like input; otherwise a
    # default route can accidentally inherit a prior district/unit hint and look
    # like a "cached result" contamination.
    is_markup_like_query = bool(re.search(r"<\s*/?\s*[a-z][^>]*>", query or "", re.IGNORECASE))
    if is_markup_like_query or not followup_like_query:
        merged = (query or "").strip()
    else:
        merged = f"{(last_user_query or '').strip()} {query}".strip()
    place_hints = extract_place_hints(query)
    if not place_hints and merged != query:
        place_hints = extract_place_hints(merged)
    place = place_hints[0] if place_hints else (
        extract_place_hint(primary_query or query) or extract_place_hint(query) or extract_place_hint(merged)
    )
    sanitized_place_hints = []
    for raw_place in place_hints:
        normalized_place = _sanitize_district_value(raw_place)
        if normalized_place and normalized_place not in sanitized_place_hints:
            sanitized_place_hints.append(normalized_place)
    place_hints = sanitized_place_hints
    place = _sanitize_district_value(place)
    if place and _looks_like_sort_phrase(place):
        place = None
    hierarchy_place_hint = _sanitize_district_value(_extract_hierarchy_place(query) or _extract_hierarchy_place(merged))
    if not place and hierarchy_place_hint:
        place = hierarchy_place_hint
    if hierarchy_place_hint and hierarchy_place_hint not in place_hints:
        place_hints = [hierarchy_place_hint, *place_hints][:4]
    compare_places = _extract_compare_places(query)
    for candidate in compare_places:
        if candidate not in place_hints:
            place_hints.append(candidate)
    place_hints = place_hints[:4]
    unit_hint = extract_unit_hint(query) or extract_unit_hint(merged)
    rank_hints = extract_rank_hints(query)
    if not rank_hints and merged != query:
        rank_hints = extract_rank_hints(merged)
    rank_in_query = rank_hints[0] if rank_hints else extract_rank_hint(query)
    rank_name = rank_hints[0] if rank_hints else (rank_in_query or extract_rank_hint(merged))
    rank_relation = _extract_rank_relation(query)
    user_id_hint = extract_user_id_hint(query)
    mobile_hint = extract_mobile_hint(query)
    person_hint = extract_person_hint(query)
    designation_hint = _extract_designation_hint(query) or _extract_designation_hint(merged)
    bare_info_subject = _extract_bare_info_subject(query)
    fixed_args = dict(arguments or {})
    existing_district = _sanitize_district_value(str(fixed_args.get("district_name") or ""))
    if existing_district:
        fixed_args["district_name"] = existing_district
    else:
        fixed_args.pop("district_name", None)
    existing_districts = fixed_args.get("district_names")
    if isinstance(existing_districts, list):
        normalized_districts = []
        for item in existing_districts:
            normalized = _sanitize_district_value(str(item or ""))
            if normalized and normalized not in normalized_districts:
                normalized_districts.append(normalized)
        if normalized_districts:
            fixed_args["district_names"] = normalized_districts[:4]
        else:
            fixed_args.pop("district_names", None)
    fixed_tool = tool_name
    prev_assistant_lower = (last_assistant_response or "").lower()

    asks_for_people = bool(re.search(r"\b(who|which|list|show|get|give|tell|provide|names?)\b", query_lower))
    asks_for_rank_members = bool(
        re.search(r"\b(?:who|which|list|show|get)\b.*\b(?:constable|si|asi|hc|pc|inspector|dsp|sp|ips|dysp)s?\b", query_lower)
    ) or bool(rank_name and re.search(r"\b(?:who|which|list|show|get)\b", query_lower))
    asks_followup_details = bool(
        re.search(r"\b(their|them|they|those|these)\b", query_lower)
        and re.search(r"\b(info|information|details?|contact|email|mobile|phone)\b", query_lower)
    )
    asks_followup_district = bool(
        re.search(r"\b(their|them|they|those|these)\b", query_lower)
        and re.search(r"\b(districts?|belong\s+to|belongs\s+to)\b", query_lower)
    )
    asks_attachment_followup = bool(
        re.search(r"\b(their|them|they|those|these)\b", query_lower)
        and re.search(r"\b(unit|attached|attachment|assigned|posted|belongs?)\b", query_lower)
        and re.search(
            r"\b(si|si's|sis|sub[-\s]?inspectors?|asi|asis|hc|pc|constables?|inspectors?|dsp|dysp|sp)\b",
            query_lower,
        )
    )
    asks_ordinal_details = bool(
        re.search(r"\b(info|information|details?|contact|email|mobile|phone)\b", query_lower)
        and extract_ordinal_index(query_lower) is not None
    )
    asks_numbered_followup = bool(
        re.search(r"\b(?:tell|show|get|give|provide)\b", query_lower)
        and re.search(r"\b(?:about|of|on|for|item|number|no\.?|#)\b", query_lower)
    )
    asks_person_attribute = bool(
        re.search(
            r"\b(email|e-?mail|mobile|phone|contact|dob|date of birth|birthday|address|blood group|rank|designation)\b",
            query_lower,
        )
    )
    prev_was_district_list = bool(
        "available districts" in prev_assistant_lower
        or (
            "district" in prev_assistant_lower
            and bool(re.search(r"^\s*\d+\.\s+", last_assistant_response or "", re.MULTILINE))
        )
    )
    asks_rank_detail_listing = bool(
        rank_name
        and place
        and re.search(r"\b(details?|info|information|list|show|get|give|which|who)\b", query_lower)
        and not re.search(r"\b(email|e-?mail|mobile|phone|contact|dob|date of birth|birthday|address|blood group)\b", query_lower)
    )
    asks_unit_leader_name = bool(
        re.search(
            r"\b(?:who\s+is|what\s+is\s+the\s+name\s+of|name\s+of)\b.*\b(?:sho|sdpo|spdo|in[\s-]?charge|responsible\s*user|head)\b",
            primary_query_lower or query_lower,
        )
        or re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge)\b\s+of\b", primary_query_lower or query_lower)
        or re.search(r"^\s*(?:the\s+)?(?:sho|sdpo|spdo)\s+[A-Za-z0-9]", primary_query or query, re.IGNORECASE)
    )
    asks_designation_lookup = bool(
        designation_hint
        and (
            re.search(r"\b(designation|post|role)\b", primary_query_lower or query_lower)
            or re.search(r"\blist\s+(?:all\s+)?(?:sdpo|spdo)s?\b", primary_query_lower or query_lower)
            or re.search(r"\bwho\s+has\b", primary_query_lower or query_lower)
        )
    )
    mentions_personnel_typo_or_synonym = bool(
        re.search(r"\b(personnel|personell|staff|officers?|people)\b", query_lower)
    )
    asks_distribution_query = bool(
        re.search(
            r"\b(?:personnel|personell|officers?|staff|people)\s+"
            r"(?:count|total|strength|distribution|breakdown)\b",
            query_lower,
        )
        or re.search(r"\b(?:count|total|number)\s+(?:of\s+)?(?:personnel|personell|officers?|staff|people)\b", query_lower)
        or re.search(r"\bhow\s+many\s+(?:personnel|personell|officers?|staff|people)\b", query_lower)
    )
    asks_distribution_by_district = bool(
        re.search(
            r"\b(?:personnel|personell|officers?|staff|people)\s+distribution\b.*\bby\s+district\b",
            query_lower,
        )
        or re.search(r"\bdistrict\s+wise\b", query_lower)
    )
    asks_ratio_query = bool(re.search(r"\bratio\b", query_lower))
    asks_comparison_query = bool(re.search(r"\b(compare|comparison|versus|vs\.?)\b", query_lower))
    asks_senior_officer_compare = bool(re.search(r"\bsenior\s+officers?\b", query_lower))
    asks_rank_dob_extreme = bool(
        rank_name
        and re.search(r"\b(earliest|oldest)\b", query_lower)
        and re.search(r"\b(date\s+of\s+birth|dob|birthday|birth)\b", query_lower)
    )
    asks_district_catalog = bool(
        re.search(
            r"\b(?:info|information|details?|about|list|show|get)\b",
            query_lower,
        )
        and re.search(r"\b(?:all\s+)?(?:districts|disctricts)\b", query_lower)
        and not place
        and not unit_hint
        and not rank_name
    )
    has_conjunction_multi_intent = bool(
        re.search(r"\b(?:and|also|then)\b", query_lower)
        and sum(
            1
            for flag in (
                bool(re.search(r"\bunits?\b", query_lower)),
                bool(re.search(r"\bvillages?\b|\bmapping\b|\bcoverage\b", query_lower)),
                bool(re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge|responsible\s*user)\b", query_lower)),
                bool(re.search(r"\bpersonnel|personell|officers?|staff|people|rank\b", query_lower)),
                bool(re.search(r"\btransfers?\b|\bposting\b|\bmovement\b", query_lower)),
            )
        ) >= 2
    )
    sp_unit_candidate = (
        _extract_sp_of_unit_candidate(primary_query or query)
        or _extract_sp_of_unit_candidate(query)
        or _extract_sp_of_unit_candidate(merged)
    )
    where_is_target = _extract_where_is_target(primary_query or query) or _extract_where_is_target(query)

    # "SP of <unit>" should resolve to current command/in-charge for that unit,
    # not rank listing in district.
    if sp_unit_candidate:
        return "get_unit_command_history", {"unit_name": sp_unit_candidate}

    # "Where is <target>" should prefer unit lookup over designation/person routes.
    if where_is_target:
        return "search_unit", {"name": where_is_target}

    # Vacancy queries → count_vacancies_by_unit_rank
    if re.search(r"\b(vacanc(?:y|ies)|vacant\s+posts?)\b", query_lower):
        vacancy_args: Dict[str, Any] = {}
        if place:
            vacancy_args["district_name"] = place
        elif unit_hint:
            vacancy_args["unit_name"] = unit_hint
        return "count_vacancies_by_unit_rank", vacancy_args

    # Rank + 'each district' pattern → query_personnel_by_rank with no district filter
    if (
        rank_name
        and re.search(r"\beach\s+district\b", query_lower)
    ):
        each_args: Dict[str, Any] = {"rank_name": rank_name}
        if rank_relation != "exact":
            each_args["rank_relation"] = rank_relation
        return "query_personnel_by_rank", each_args

    # Transfer queries → query_recent_transfers
    if re.search(r"\b(transfers?|posting|postings|movement)\b", query_lower):
        transfer_args: Dict[str, Any] = {}
        transfer_place = _sanitize_district_value(
            _extract_transfer_place(primary_query or query) or _extract_transfer_place(query) or place
        )
        if transfer_place and transfer_place.lower() not in {"ap", "andhra pradesh"}:
            transfer_args["district_name"] = transfer_place
        days_match = re.search(r"\b(\d+)\s*days?\b", query_lower)
        if days_match:
            transfer_args["days"] = int(days_match.group(1))
        else:
            # Also guard "last 30 days" form
            last_match = re.search(r"\blast\s+(\d+)\s*days?\b", query_lower)
            if last_match:
                transfer_args["days"] = int(last_match.group(1))
            else:
                transfer_args["days"] = 30
        return "query_recent_transfers", transfer_args

    # Village/coverage queries
    if re.search(r"\b(missing\s+village|missing\s+villages?)\b", query_lower):
        village_args: Dict[str, Any] = {}
        if place:
            village_args["district_name"] = place
        return "find_missing_village_mappings", village_args

    if re.search(r"\b(village|villages?|mapped|coverage)\b", query_lower) and re.search(r"\b(ps|station|unit)\b", query_lower):
        unit_candidate = unit_hint or place
        village_cov_args: Dict[str, Any] = {}
        if unit_candidate:
            village_cov_args["unit_name"] = unit_candidate
        return "get_village_coverage", village_cov_args


    if (
        re.search(r"\b(?:list|show|get|find)\b", query_lower)
        and re.search(r"\bunits?\b", query_lower)
        and not re.search(r"\branges?\b", query_lower)
        and place
    ):
        list_units_args: Dict[str, Any] = {"district_name": place}
        return "list_units_in_district", list_units_args

    # "list personnel in <district>" with no rank filter → get_personnel_distribution
    if (
        re.search(r"\b(?:list|show|get|find)\b", query_lower)
        and re.search(r"\b(personnel|personell|officers?|staff|people)\b", query_lower)
        and place
        and not rank_name
        and not unit_hint
    ):
        return "get_personnel_distribution", {"district_name": place, "group_by": "rank"}


    if (
        re.search(r"\bhow\s+many\s+ranges?\b", query_lower)
        or re.search(r"\b(?:count|total|number)\s+(?:of\s+)?ranges?\b", query_lower)
        or re.search(r"\branges?\s+(?:count|total|number)\b", query_lower)
    ):
        range_place = _sanitize_district_value(_extract_range_place(primary_query or query) or _extract_range_place(query))
        district_candidate = range_place or place
        if district_candidate and district_candidate.lower() not in {"ap", "andhra pradesh"}:
            args: Dict[str, Any] = {"unit_type_name": "Range", "district_name": district_candidate}
            return "search_unit", args
        # System-wide Range count — use dynamic_query since search_unit requires a district
        return "dynamic_query", {"intent": "Count all units of type Range in the system"}

    if (
        re.search(r"\b(?:list|show|get|find)\b", query_lower)
        and re.search(r"\bunits?\b", query_lower)
        and re.search(r"\branges?\b", query_lower)
    ):
        range_place = _sanitize_district_value(_extract_range_place(primary_query or query) or _extract_range_place(query))
        district = range_place or place
        if district:
            district = _sanitize_district_value(re.sub(r"\branges?\b", "", district, flags=re.IGNORECASE).strip())
        if district:
            return "list_units_in_district", {"unit_type_name": "Range", "district_name": district}
        # No specific district — use dynamic_query for system-wide Range listing
        return "dynamic_query", {"intent": "List all units of type Range in the system"}

    # Prefer unit hierarchy early for typo variants like "heirarchy of chittoor district"
    # before other generic "details/about" or list-style heuristics can interfere.
    if re.search(hierarchy_intent_pattern, query_lower) and not re.search(
        r"\b(personnel|officers?|staff)\b", query_lower
    ):
        reporting_to = re.search(
            r"\breporting\s+to\s+([A-Za-z0-9\.\-\s]+?\s+(?:dpo|sdpo|spdo|ps|station|circle|range|wing|office))\b",
            query,
            re.IGNORECASE,
        )
        if reporting_to:
            root_name = re.sub(r"\s+", " ", reporting_to.group(1)).strip()
            return "get_unit_hierarchy", {"root_unit_name": root_name}
        hierarchy_place = extract_place_hint(query) or (None if is_markup_like_query else extract_place_hint(merged))
        if hierarchy_place and re.search(r"\b(starting\s+from|district\s+dpo)\b", query_lower):
            return "get_unit_hierarchy", {"root_unit_name": f"{hierarchy_place} DPO"}
        if unit_hint and re.search(r"\b(ps|station|sdpo|spdo|dpo|circle|range|wing|office)\b", unit_hint, re.IGNORECASE):
            return "get_unit_hierarchy", {"root_unit_name": unit_hint}
        if hierarchy_place:
            return "get_unit_hierarchy", {"district_name": hierarchy_place}

    if asks_ratio_query and rank_hints:
        ratio_args: Dict[str, Any] = {"group_by": "rank"}
        if len(place_hints) >= 2:
            ratio_args["district_names"] = place_hints[:4]
        elif place:
            ratio_args["district_name"] = place
        return "get_personnel_distribution", ratio_args

    if (
        (asks_comparison_query or asks_senior_officer_compare)
        and re.search(r"\b(personnel|officers?|staff|rank|distribution|senior)\b", query_lower)
        and len(place_hints) >= 2
    ):
        return "get_personnel_distribution", {"group_by": "rank", "district_names": place_hints[:4]}

    if (
        rank_name
        and len(place_hints) >= 2
        and re.search(r"\b(find|list|show|get|all|contact|mobile|phone|email|details?)\b", query_lower)
    ):
        rank_args: Dict[str, Any] = {"rank_name": rank_name, "district_names": place_hints[:4]}
        if re.search(r"\b(contact|mobile|phone|email|details?)\b", query_lower):
            rank_args["page_size"] = 200
        return "query_personnel_by_rank", rank_args

    if asks_rank_dob_extreme:
        dob_args: Dict[str, Any] = {"rank_name": rank_name, "page_size": 500}
        if len(place_hints) == 1:
            dob_args["district_name"] = place_hints[0]
        return "query_personnel_by_rank", dob_args

    if asks_ordinal_details or asks_numbered_followup:
        # If session-state hint injection already resolved the referenced list item,
        # force a person lookup instead of letting keyword routing drift elsewhere.
        injected_user_id = (fixed_args.get("user_id") or "").strip() if isinstance(fixed_args, dict) else ""
        if injected_user_id:
            return "search_personnel", {"user_id": injected_user_id}
        idx = extract_ordinal_index(query_lower) or extract_list_reference_index(query_lower)
        user_from_list = extract_user_id_from_list_item(last_assistant_response or "", idx or 0)
        if user_from_list:
            return "search_personnel", {"user_id": user_from_list}

    if asks_rank_detail_listing:
        rank_args: Dict[str, Any] = {"rank_name": rank_name}
        if rank_relation != "exact":
            rank_args["rank_relation"] = rank_relation
        if len(place_hints) >= 2:
            rank_args["district_names"] = place_hints[:4]
        elif place:
            rank_args["district_name"] = place
        return "query_personnel_by_rank", rank_args

    if person_hint and asks_person_attribute:
        return "search_personnel", {"name": person_hint}

    if asks_unit_leader_name:
        role_unit = (
            _extract_role_unit_candidate(primary_query or query)
            or _extract_role_unit_candidate(query)
            or _extract_role_unit_candidate(merged)
        )
        existing_unit = (fixed_args.get("unit_name") or "").strip() if isinstance(fixed_args, dict) else ""
        unit_candidate = role_unit
        if not unit_candidate and existing_unit and not _looks_like_bad_unit_hint(existing_unit):
            unit_candidate = existing_unit
        if not unit_candidate and unit_hint and not _looks_like_bad_unit_hint(unit_hint):
            unit_candidate = unit_hint
        if not unit_candidate and place:
            if re.search(r"\b(?:spdo|sdpo)\b", query_lower):
                unit_candidate = f"{place} SDPO"
            elif re.search(r"\bsho\b", query_lower):
                unit_candidate = f"{place} PS"
            else:
                unit_candidate = place
        if unit_candidate:
            return "get_unit_command_history", {"unit_name": unit_candidate}

    # Designation requests like "who has the designation of SPDO" should not be
    # forced into rank lookup; use personnel search with designation filter.
    if asks_designation_lookup and not asks_unit_leader_name and not person_hint:
        return "search_personnel", {"designation_name": designation_hint}

    if asks_attachment_followup:
        prev_rank = extract_rank_hint(last_user_query or "") or extract_rank_hint(last_assistant_response or "") or rank_name
        prev_place = extract_place_hint(last_user_query or "") or extract_place_hint(last_assistant_response or "") or place
        if prev_rank:
            args_followup: Dict[str, Any] = {"rank_name": prev_rank}
            if prev_place:
                args_followup["district_name"] = prev_place
            return "query_personnel_by_rank", args_followup

    if asks_followup_details:
        explicit_current_target = bool(
            rank_name
            or place
            or place_hints
            or unit_hint
            or user_id_hint
            or person_hint
        )
        if explicit_current_target:
            # Don't override explicit rank/place/unit queries that happen to
            # contain pronouns like "their".
            pass
        else:
            prev_rank = extract_rank_hint(last_user_query or "")
            prev_place = extract_place_hint(last_user_query or "")
            prev_user_id = extract_user_id_hint(last_user_query or "")
            if prev_user_id:
                return "search_personnel", {"user_id": prev_user_id}
            if prev_rank:
                args_followup: Dict[str, Any] = {"rank_name": prev_rank}
                if prev_place:
                    args_followup["district_name"] = prev_place
                return "query_personnel_by_rank", args_followup

    if asks_followup_district:
        prev_rank = extract_rank_hint(last_user_query or "") or rank_name
        prev_place = extract_place_hint(last_user_query or "") or place
        if prev_rank:
            args_followup: Dict[str, Any] = {"rank_name": prev_rank}
            if prev_place:
                args_followup["district_name"] = prev_place
            return "query_personnel_by_rank", args_followup

    if re.search(r"\bdistrict\s*[- ]?\s*wise\b", query_lower) and (
        rank_name or re.search(r"\b(si|asi|hc|pc|constable|inspector|dsp|sp|dysp)\b", query_lower)
    ):
        fixed_args = {"group_by": "district"}
        if place:
            fixed_args["district_name"] = place
        return "get_personnel_distribution", fixed_args

    if asks_distribution_query:
        dist_args: Dict[str, Any] = {"group_by": "district" if asks_distribution_by_district else "rank"}
        if place:
            dist_args["district_name"] = place
        elif unit_hint:
            dist_args["unit_name"] = unit_hint
        return "get_personnel_distribution", dist_args

    # Catalog/list intent like "info on all districts" should not drift into
    # person search even if generic extractors capture "all districts".
    if asks_district_catalog:
        return "list_districts", {}

    if user_id_hint:
        return "search_personnel", {"user_id": user_id_hint}

    if mobile_hint and re.search(r"\b(number|mobile|phone|contact|who['’]?s)\b", query_lower):
        return "search_personnel", {"mobile": mobile_hint}

    if (
        prev_was_district_list
        and bare_info_subject
        and re.search(r"\b(info|details?|about)\b", query_lower)
        and not unit_hint
        and not rank_name
        and not asks_person_attribute
    ):
        return "list_units_in_district", {"district_name": bare_info_subject}

    if (
        re.search(r"\b(info|details?|about)\b", query_lower)
        and re.search(r"\bdist(?:rict)?\.?\b", query_lower)
        and place
        and not unit_hint
        and not rank_name
    ):
        return "list_units_in_district", {"district_name": place}

    # If generic "info/details/about <x>" matches both person and place extractors
    # with the same token (e.g., "info on kuppam"), prefer place/unit routing.
    if (
        re.search(r"\b(info|details?|about)\b", query_lower)
        and place
        and person_hint
        and place.strip().lower() == person_hint.strip().lower()
        and not unit_hint
        and not rank_name
        and not asks_person_attribute
    ):
        return "search_unit", {"name": place}

    if re.search(r"\b(info|details?|about)\b", query_lower) and place and not unit_hint and not rank_name and not person_hint:
        return "list_units_in_district", {"district_name": place}

    if re.search(r"\b(info|details?|about)\b", query_lower) and person_hint:
        if not place or person_hint.strip().lower() != place.strip().lower():
            return "search_personnel", {"name": person_hint}

    if re.search(r"\bwho\s+is\b|\bfind\s+(?:person|officer)\b|\bsearch\s+(?:person|officer)\b", query_lower) and person_hint:
        # Prefer direct person lookup over prior-turn rank context.
        if not rank_in_query:
            return "search_personnel", {"name": person_hint}

    if (
        tool_name == "search_personnel"
        and not any((fixed_args or {}).get(k) for k in ("name", "user_id", "badge_no", "mobile", "email"))
        and person_hint
    ):
        return "search_personnel", {"name": person_hint}

    # Queries like "personell in kuppam district" should not hit strict search_personnel validation.
    if (
        tool_name == "search_personnel"
        and mentions_personnel_typo_or_synonym
        and place
        and not any((fixed_args or {}).get(k) for k in ("name", "user_id", "badge_no", "mobile", "email"))
        and not person_hint
    ):
        if rank_name:
            rank_args: Dict[str, Any] = {"rank_name": rank_name}
            if rank_relation != "exact":
                rank_args["rank_relation"] = rank_relation
            if len(place_hints) >= 2:
                rank_args["district_names"] = place_hints[:4]
            else:
                rank_args["district_name"] = place
            return "query_personnel_by_rank", rank_args
        if unit_hint:
            return "query_personnel_by_unit", {"unit_name": unit_hint}
        return "get_personnel_distribution", {"district_name": place, "group_by": "rank"}

    if asks_for_people and asks_for_rank_members:
        fixed_tool = "query_personnel_by_rank"
        fixed_args = {}
        if rank_name or arguments.get("rank_name"):
            fixed_args["rank_name"] = rank_name or arguments.get("rank_name")
        if rank_relation != "exact":
            fixed_args["rank_relation"] = rank_relation
        if len(place_hints) >= 2:
            fixed_args["district_names"] = place_hints[:4]
        elif not fixed_args.get("district_name") and place:
            fixed_args["district_name"] = place
        return fixed_tool, fixed_args

    # Unit hierarchy misspelling support: "heirarchy of chittoor district"
    if re.search(hierarchy_intent_pattern, query_lower) and not re.search(
        r"\b(personnel|officers?|staff)\b", query_lower
    ):
        hierarchy_args: Dict[str, Any] = {}
        if place and re.search(r"\b(starting\s+from|district\s+dpo)\b", query_lower):
            hierarchy_args["root_unit_name"] = f"{place} DPO"
            return "get_unit_hierarchy", hierarchy_args
        reporting_to = re.search(
            r"\breporting\s+to\s+([A-Za-z0-9\.\-\s]+?\s+(?:dpo|sdpo|spdo|ps|station|circle|range|wing|office))\b",
            query,
            re.IGNORECASE,
        )
        if reporting_to:
            hierarchy_args["root_unit_name"] = re.sub(r"\s+", " ", reporting_to.group(1)).strip()
            return "get_unit_hierarchy", hierarchy_args
        if unit_hint and re.search(r"\b(ps|station|sdpo|spdo|dpo|circle|range|wing|office)\b", unit_hint, re.IGNORECASE):
            hierarchy_args["root_unit_name"] = unit_hint
            return "get_unit_hierarchy", hierarchy_args
        if place:
            hierarchy_args["district_name"] = place
            return "get_unit_hierarchy", hierarchy_args

    if tool_name == "get_unit_hierarchy" and not any(
        fixed_args.get(k) for k in ("root_unit_id", "root_unit_name", "district_id", "district_name")
    ):
        if place:
            fixed_args["district_name"] = place
        elif unit_hint:
            fixed_args["root_unit_name"] = unit_hint

    if ("personnel" in query_lower and re.search(hierarchy_intent_pattern, query_lower)):
        fixed_args = {"group_by": "rank"}
        if place:
            fixed_args["district_name"] = place
        return "get_personnel_distribution", fixed_args

    if re.search(r"\b(available\s+ranks?|different\s+ranks?|list\s+(?:all\s+)?ranks?|ranks?\s+of\s+all|rank\s+distribution|rank\s+wise)\b", query_lower):
        fixed_args = {"group_by": "rank"}
        if place:
            fixed_args["district_name"] = place
        return "get_personnel_distribution", fixed_args

    if re.search(r"\b(info|details?|about)\b", query_lower) and unit_hint:
        return "search_unit", {"name": unit_hint}

    if tool_name in {"search_unit", "get_village_coverage"} and unit_hint:
        if tool_name == "search_unit":
            return "search_unit", {"name": unit_hint}
        return "get_village_coverage", {"unit_name": unit_hint}

    if tool_name == "query_personnel_by_unit" and not any(fixed_args.get(k) for k in ("unit_id", "unit_name")):
        if unit_hint:
            fixed_args["unit_name"] = unit_hint
        elif place:
            # If user explicitly asks for leader/role holder, prefer command history over distribution.
            if asks_unit_leader_name:
                if re.search(r"\b(?:spdo|sdpo)\b", query_lower):
                    return "get_unit_command_history", {"unit_name": f"{place} SDPO"}
                return "get_unit_command_history", {"unit_name": place}
            return "get_personnel_distribution", {"district_name": place, "group_by": "rank"}

    if tool_name == "query_personnel_by_rank":
        if _looks_like_sort_phrase(str(fixed_args.get("district_name") or "")):
            fixed_args.pop("district_name", None)
        if not fixed_args.get("rank_name") and rank_name:
            fixed_args["rank_name"] = rank_name
        if rank_relation != "exact" and not fixed_args.get("rank_relation"):
            fixed_args["rank_relation"] = rank_relation
        if not fixed_args.get("district_names") and len(place_hints) >= 2:
            fixed_args["district_names"] = place_hints[:4]
        if not fixed_args.get("district_name") and not fixed_args.get("district_names") and place:
            fixed_args["district_name"] = place

    if tool_name == "list_units_in_district":
        current_district = (fixed_args.get("district_name") or "").strip().lower()
        invalid = {"", "district", "the district", "db", "database", "the db", "the database"}
        if current_district in invalid and place:
            fixed_args["district_name"] = place

    if tool_name == "query_recent_transfers":
        transfer_place = _extract_transfer_place(primary_query or query) or _extract_transfer_place(query) or place
        if not fixed_args.get("district_name") and transfer_place and transfer_place.lower() not in {"ap", "andhra pradesh"}:
            fixed_args["district_name"] = transfer_place
        days_raw = fixed_args.get("days")
        try:
            days_int = int(days_raw) if days_raw is not None else None
        except Exception:
            days_int = None
        if days_int is None or days_int <= 0:
            days_match = re.search(r"\b(\d+)\s*days?\b", query_lower)
            fixed_args["days"] = int(days_match.group(1)) if days_match else 30

    if re.search(r"\b(there|here|that place|that unit|this unit)\b", query_lower):
        prev_unit = extract_unit_hint(last_user_query or "")
        prev_place = extract_place_hint(last_user_query or "")
        wants_personnel = bool(re.search(r"\b(list|show|get)\b.*\b(personnel|staff|officers?|people)\b", query_lower))
        if wants_personnel and prev_unit:
            return "query_personnel_by_unit", {"unit_name": prev_unit}
        if wants_personnel and prev_place:
            return "get_personnel_distribution", {"district_name": prev_place, "group_by": "rank"}
        if prev_unit and fixed_tool in {"search_personnel", "get_personnel_distribution"}:
            return "search_unit", {"name": prev_unit}

    if tool_name == "get_personnel_distribution" and not fixed_args.get("district_name") and not fixed_args.get("district_names") and place:
        fixed_args["district_name"] = place
        fixed_args.setdefault("group_by", "rank")
    elif tool_name == "get_personnel_distribution":
        district_value = str(fixed_args.get("district_name") or "").strip()
        if district_value and re.search(
            r"\b(personnel|personell|distribution|visual|representation|chart|graph|plot)\b",
            district_value,
            re.IGNORECASE,
        ):
            fixed_args.pop("district_name", None)
        fixed_args.setdefault("group_by", "rank")

    if re.search(
        r"(?:list|show|get|all)\s+(?:districts?|disctricts?)\b|(?:which|what)\s+(?:districts?|disctricts?)\s+(?:are\s+)?(?:available|in\s+the\s+db|in\s+database)\b|(?:districts?|disctricts?)\s+(?:available|in\s+the\s+db|in\s+database)\b|(?:info|information|details?|about)\s+(?:on\s+)?(?:all\s+)?(?:districts|disctricts)\b",
        query,
        re.IGNORECASE,
    ):
        return "list_districts", {}

    return fixed_tool, fixed_args
