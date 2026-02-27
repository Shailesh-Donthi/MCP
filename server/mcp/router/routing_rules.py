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


def repair_route(
    query: str,
    tool_name: str,
    arguments: Dict[str, Any],
    last_user_query: Optional[str] = None,
    last_assistant_response: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
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
            value = re.sub(r"^(?:on|about|for|of)\s+", "", value, flags=re.IGNORECASE).strip()
            if not value:
                continue
            if re.search(r"\b(?:sho|sdpo|spdo|ps|station|unit|mobile|phone|email|dob|address)\b", value, re.IGNORECASE):
                continue
            return value
        return None

    def _canonicalize_role_unit(role: str, raw_target: str) -> Optional[str]:
        target = re.sub(r"\s+", " ", (raw_target or "")).strip(" ?")
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

        return re.sub(r"\s+", " ", target).strip()

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

    def _looks_like_bad_unit_hint(value: Optional[str]) -> bool:
        v = (value or "").strip()
        if not v:
            return True
        if re.search(r"^\s*(?:who|what)\s+is\b", v, re.IGNORECASE):
            return True
        if re.search(r"\b(?:sho|sdpo|spdo)\b\s+of\b", v, re.IGNORECASE):
            return True
        return False

    query_lower = (query or "").lower()
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
    place = place_hints[0] if place_hints else (extract_place_hint(query) or extract_place_hint(merged))
    unit_hint = extract_unit_hint(query) or extract_unit_hint(merged)
    rank_hints = extract_rank_hints(query)
    if not rank_hints and merged != query:
        rank_hints = extract_rank_hints(merged)
    rank_in_query = rank_hints[0] if rank_hints else extract_rank_hint(query)
    rank_name = rank_hints[0] if rank_hints else (rank_in_query or extract_rank_hint(merged))
    user_id_hint = extract_user_id_hint(query)
    mobile_hint = extract_mobile_hint(query)
    person_hint = extract_person_hint(query)
    bare_info_subject = _extract_bare_info_subject(query)
    fixed_args = dict(arguments or {})
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
            query_lower,
        )
        or re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge)\b\s+of\b", query_lower)
        or re.search(r"^\s*(?:the\s+)?(?:sho|sdpo|spdo)\s+[A-Za-z0-9]", query, re.IGNORECASE)
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

    # Prefer unit hierarchy early for typo variants like "heirarchy of chittoor district"
    # before other generic "details/about" or list-style heuristics can interfere.
    if re.search(r"\b(?:hierarchy|heirarchy|structure|tree|organization)\b", query_lower) and not re.search(
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
        if len(place_hints) >= 2:
            rank_args["district_names"] = place_hints[:4]
        elif place:
            rank_args["district_name"] = place
        return "query_personnel_by_rank", rank_args

    if person_hint and asks_person_attribute:
        return "search_personnel", {"name": person_hint}

    if asks_unit_leader_name:
        role_unit = _extract_role_unit_candidate(query) or _extract_role_unit_candidate(merged)
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

    if asks_followup_details:
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
        prev_place = extract_place_hint(last_user_query or "")
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
        if len(place_hints) >= 2:
            fixed_args["district_names"] = place_hints[:4]
        elif not fixed_args.get("district_name") and place:
            fixed_args["district_name"] = place
        return fixed_tool, fixed_args

    # Unit hierarchy misspelling support: "heirarchy of chittoor district"
    if re.search(r"\b(?:hierarchy|heirarchy|structure|tree|organization)\b", query_lower) and not re.search(
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

    if ("personnel" in query_lower and ("hierarchy" in query_lower or "heirarchy" in query_lower)):
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
        if not fixed_args.get("rank_name") and rank_name:
            fixed_args["rank_name"] = rank_name
        if not fixed_args.get("district_names") and len(place_hints) >= 2:
            fixed_args["district_names"] = place_hints[:4]
        if not fixed_args.get("district_name") and not fixed_args.get("district_names") and place:
            fixed_args["district_name"] = place

    if tool_name == "list_units_in_district":
        current_district = (fixed_args.get("district_name") or "").strip().lower()
        invalid = {"", "district", "the district", "db", "database", "the db", "the database"}
        if current_district in invalid and place:
            fixed_args["district_name"] = place

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


def needs_clarification(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    # Markup / tag-like input should not be routed to data tools.
    if "<" in q or ">" in q:
        return True
    if re.search(r"<\s*/?\s*[a-z][^>]*>", q):
        return True
    vague_exact = {
        "info", "information", "details", "tell me", "show me",
        "what information", "what can you tell me",
        "personnel", "personell", "officers", "officer", "staff", "people",
        "who is there", "who is here", "there", "here",
    }
    if q in vague_exact:
        return True
    if re.search(r"^\s*who\s+is\s+(?:there|here|that|this)\s*$", q):
        return True
    # Explicit analytical requests with concrete targets should route directly.
    if re.search(r"\b(compare|comparison|ratio|versus|vs\.?)\b", q):
        place_hints = extract_place_hints(q)
        has_rank = bool(extract_rank_hints(q)) or bool(
            re.search(r"\b(si|asi|hc|pc|constable|inspector|circle\s+inspector|dsp|dysp|sp)\b", q)
        )
        has_subject = bool(re.search(r"\b(personnel|officers?|staff|rank|distribution|hierarchy|wing)\b", q))
        mentions_all_districts = bool(re.search(r"\b(across\s+all\s+districts?|all\s+districts?|across\s+districts?)\b", q))
        if has_subject and (has_rank or len(place_hints) >= 2 or mentions_all_districts):
            return False
    if (
        bool(extract_rank_hints(q))
        and len(extract_place_hints(q)) >= 2
        and re.search(r"\b(find|list|show|get|contact|details?)\b", q)
    ):
        return False
    if (
        bool(extract_rank_hint(q))
        and re.search(r"\b(earliest|oldest)\b", q)
        and re.search(r"\b(date\s+of\s+birth|dob|birthday|birth)\b", q)
    ):
        return False
    # Minimal/compound prompts with multiple intents should be clarified explicitly.
    if re.search(r"\b(?:and|also|then)\b", q):
        place_hints = extract_place_hints(q)
        is_explicit_compare = bool(
            re.search(r"\b(compare|comparison|ratio|versus|vs\.?)\b", q)
            and (
                len(place_hints) >= 2
                or re.search(r"\b(across\s+all\s+districts?|all\s+districts?)\b", q)
            )
        )
        intent_hits = sum(
            1
            for flag in (
                bool(re.search(r"\bunits?\b", q)),
                bool(re.search(r"\bvillages?\b|\bmapping\b|\bcoverage\b", q)),
                bool(re.search(r"\b(?:sho|sdpo|spdo|in[\s-]?charge|responsible\s*user)\b", q)),
                bool(re.search(r"\bpersonnel|personell|officers?|staff|people|rank\b", q)),
                bool(re.search(r"\btransfers?\b|\bposting\b|\bmovement\b", q)),
            )
        )
        if intent_hits >= 2 and not is_explicit_compare:
            return True
    if re.search(r"\b(about|on|for|of)\s*$", q):
        return True
    if re.search(
        r"^(?:what|which|who|tell me|give me|show me).*\b(?:info|information|details)\b.*\b(?:about|on|for)\s*$",
        q,
    ):
        return True
    if re.search(
        r"^(?:what\s+(?:info|information)\s+can\s+you\s+(?:give|provide|tell)\s+me|what\s+can\s+you\s+(?:give|provide|tell)\s+me)(?:\s+about)?\s*$",
        q,
    ):
        return True
    # Ambiguous "details of <rank> in <place>" style requests are better clarified or
    # normalized later; treat them as clarifiable to avoid person-name misreads.
    if re.search(
        r"\bdetails?\s+of\s+(?:all\s+)?(?:si|asi|hc|pc|constable|inspector|circle\s+inspector|dsp|dysp|sp)\b.*\bin\s+[A-Za-z]",
        q,
    ):
        return True
    return False
