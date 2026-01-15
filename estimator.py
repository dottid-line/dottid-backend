# estimator.py

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _majority_condition(imgs, default_label, min_room_conf=0.70, min_cond_conf=0.70):
    """
    Pick the majority condition label among images, with confidence gating.
    Fallback to default_label if nothing qualifies.
    """
    counts = {}
    for r in imgs:
        if r.get("room_conf", 0.0) < min_room_conf:
            continue
        if r.get("condition_conf", 0.0) < min_cond_conf:
            continue
        c = r.get("condition")
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1

    if not counts:
        return default_label

    # majority vote
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _sqft_cost_from_tier(sqft, tier_label):
    """
    Compute interior sqft cost from the property-level tier.

    tier_label in:
      - "fullyupdated"
      - "solidcondition"
      - "needsrehab"   (medium rehab)
      - "fullrehab"    (full gut)
    """
    sqft = _safe_float(sqft, 0.0)
    if sqft <= 0:
        return 0.0

    # Bands:
    # Fully Updated: 5x up to 1500, 2.5x above
    # Solid (Light Rehab): 10x up to 1500, 5x above
    # Medium Rehab: 15x up to 1500, 10x above
    # Full Gut: 15x across all sqft
    if tier_label == "fullyupdated":
        first = min(sqft, 1500.0)
        rest = max(0.0, sqft - 1500.0)
        return first * 5.0 + rest * 2.5

    if tier_label == "solidcondition":
        first = min(sqft, 1500.0)
        rest = max(0.0, sqft - 1500.0)
        return first * 10.0 + rest * 5.0

    if tier_label == "needsrehab":
        first = min(sqft, 1500.0)
        rest = max(0.0, sqft - 1500.0)
        return first * 15.0 + rest * 10.0

    # fullrehab or anything else → treat as full gut
    return sqft * 15.0


def estimate_rehab(subject, image_results):
    """
    Rehab estimator using SUBJECT + image_results and the new condition/tier rules.
    """

    # ------------------------------------------------------------------
    # pull subject fields (support BOTH backend keys and Shopify string keys)
    # ------------------------------------------------------------------
    def _to_float(v, default=0.0):
        try:
            if v is None:
                return default
            if isinstance(v, str):
                v = v.replace(",", "").strip()
            return float(v)
        except Exception:
            return default

    def _to_int(v, default=0):
        try:
            return int(round(_to_float(v, default)))
        except Exception:
            return default

    def _parse_yes_no(v):
        if v is None:
            return None
        t = str(v).lower().strip()
        if t in ("yes", "true", "y", "1"):
            return True
        if t in ("no", "false", "n", "0"):
            return False
        return None

    def _parse_foundation(v):
        """
        Backend expects: 'none' | 'minor_issues' | 'major_issues'
        Shopify sends: 'None' | 'Minor Rehab' | 'Major Rehab'
        """
        if v is None:
            return None
        t = str(v).lower().strip()
        if "major" in t:
            return "major_issues"
        if "minor" in t:
            return "minor_issues"
        if "none" in t:
            return "none"
        return None

    # sqft: accept multiple common keys
    sqft = subject.get("sqft", None)
    if sqft is None:
        sqft = subject.get("sqftNum", None)
    sqft = _to_float(sqft, 0.0)

    # roof/hvac: prefer backend booleans, fallback to Shopify Yes/No strings
    roof_needed = subject.get("roof_needed", None)
    hvac_needed = subject.get("hvac_needed", None)

    if roof_needed is None:
        roof_needed = _parse_yes_no(subject.get("roof"))
    if hvac_needed is None:
        hvac_needed = _parse_yes_no(subject.get("hvac"))

    roof_needed = bool(roof_needed) if roof_needed is not None else False
    hvac_needed = bool(hvac_needed) if hvac_needed is not None else False

    # foundation: prefer backend normalized, fallback to Shopify labels
    foundation_issues = subject.get("foundation_issues", None)
    if foundation_issues is None:
        foundation_issues = _parse_foundation(subject.get("foundation"))
    foundation_issues = (
        foundation_issues
        if foundation_issues in ("none", "minor_issues", "major_issues")
        else "none"
    )

    # units/baths: accept multiple common keys
    units = subject.get("units", 1)
    baths = subject.get("baths", None)
    if baths is None:
        baths = subject.get("bathsNum", 1)

    units = _to_int(units, 1)
    baths = _to_float(baths, 1.0)

    # ------------------------------------------------------------------
    # FILTER OUT INVALID IMAGES
    # ------------------------------------------------------------------
    image_results = [r for r in (image_results or []) if r.get("valid", True)]
    total_imgs = len(image_results)

    # Partition images by room type
    kitchen_imgs = []
    bath_imgs = []
    other_imgs = []

    for r in image_results:
        rt = r.get("room_type", "")
        if rt == "kitchen":
            kitchen_imgs.append(r)
        elif rt == "bathroom":
            bath_imgs.append(r)
        else:
            other_imgs.append(r)

    # ------------------------------------------------------------------
    # HELPER COUNTS (confidence-gated)
    # ------------------------------------------------------------------
    MIN_CONF = 0.70
    MIN_IMAGES_FOR_PERCENT = 5  # for 40% fullrehab rule

    def _strong(img):
        return img.get("room_conf", 0.0) >= MIN_CONF and img.get(
            "condition_conf", 0.0
        ) >= MIN_CONF

    strong_kitchens = [r for r in kitchen_imgs if _strong(r)]
    strong_baths = [r for r in bath_imgs if _strong(r)]
    strong_others = [r for r in other_imgs if _strong(r)]

    others_full = [r for r in strong_others if r.get("condition") == "fullrehab"]
    others_needs = [r for r in strong_others if r.get("condition") == "needsrehab"]
    others_solid_or_better = [
        r
        for r in strong_others
        if r.get("condition") in ("solidcondition", "fullyupdated")
    ]

    total_other_strong = len(strong_others)
    others_full_count = len(others_full)
    others_needs_count = len(others_needs)

    # kitchen/bath condition presence (image-based)
    has_k_full = any(r.get("condition") == "fullrehab" for r in strong_kitchens)
    has_b_full = any(r.get("condition") == "fullrehab" for r in strong_baths)
    has_k_needs = any(r.get("condition") == "needsrehab" for r in strong_kitchens)
    has_b_needs = any(r.get("condition") == "needsrehab" for r in strong_baths)

    # fullyupdated ratio (all images, not just strong)
    if total_imgs > 0:
        fully_count = sum(
            1 for r in image_results if r.get("condition") == "fullyupdated"
        )
        fully_ratio_all = fully_count / float(total_imgs)
    else:
        fully_ratio_all = 0.0

    # ------------------------------------------------------------------
    # KITCHEN / BATH FINAL CONDITIONS (image-majority, else user-condition-based)
    # ------------------------------------------------------------------
    WEB_CONDITION_TO_TIER = {
        "fully_updated": "fullyupdated",
        "solid_condition": "solidcondition",
        "needs_rehab": "needsrehab",
        "full_rehab": "fullrehab",
    }

    subject_condition_raw = subject.get("condition", None)
    subject_condition_key = (
        str(subject_condition_raw).strip().lower()
        if subject_condition_raw is not None
        else ""
    )
    subject_condition_label = WEB_CONDITION_TO_TIER.get(subject_condition_key, "needsrehab")

    # ------------------------------------------------------------------
    # ✅ NEW RULE: if there is NO kitchen image at all, do NOT compute off images.
    #             Fall back to the user-selected condition only.
    # ------------------------------------------------------------------
    property_tier = None  # "fullyupdated" | "solidcondition" | "needsrehab" | "fullrehab"
    if len(kitchen_imgs) == 0:
        property_tier = subject_condition_label
        kitchen_final = subject_condition_label
        bath_final = subject_condition_label

        # keep counters consistent for return payload
        total_other_strong = 0
        others_full_count = 0
        others_needs_count = 0
    else:
        # Majority condition from strong images, fallback to subject condition label
        kitchen_final = _majority_condition(
            kitchen_imgs,
            default_label=subject_condition_label,
            min_room_conf=MIN_CONF,
            min_cond_conf=MIN_CONF
        )
        bath_final = _majority_condition(
            bath_imgs,
            default_label=subject_condition_label,
            min_room_conf=MIN_CONF,
            min_cond_conf=MIN_CONF
        )

        # ------------------------------------------------------------------
        # NO-IMAGE CASE: USER CONDITION FALLBACK (never infer fullrehab)
        # ------------------------------------------------------------------
        if total_imgs == 0:
            property_tier = subject_condition_label

        # ------------------------------------------------------------------
        # IMAGE-BASED CLASSIFICATION
        # ------------------------------------------------------------------
        if property_tier is None:
            # FULL GUT TRIGGERS
            # Rule 1: kitchen full + 3 other full
            fullgut_k_plus3 = has_k_full and (others_full_count >= 3)

            # Rule 2: >= 40% of non-kitchen strong images fullrehab, with minimum depth
            full_percent = (
                (others_full_count / float(total_other_strong))
                if total_other_strong >= MIN_IMAGES_FOR_PERCENT
                else 0.0
            )
            fullgut_percent = full_percent >= 0.40

            # Rule 3: kitchen full & bath needs → full rehab (no further checks)
            k_full_b_needs = kitchen_final == "fullrehab" and bath_final == "needsrehab"

            if fullgut_k_plus3 or fullgut_percent or k_full_b_needs:
                property_tier = "fullrehab"

        if property_tier is None:
            # MEDIUM REHAB TRIGGER:
            # - Full Gut didn't fire
            # - kitchen OR bath needs
            # - 3+ other needsrehab images
            medium_trigger = (has_k_needs or has_b_needs) and (others_needs_count >= 3)
            if medium_trigger:
                property_tier = "needsrehab"

        # ------------------------------------------------------------------
        # CONFLICT RULES (only if still not classified)
        # ------------------------------------------------------------------
        has_other_imgs = len(strong_others) > 0

        if property_tier is None:
            # 1) Kitchen solid + bathroom needs rehab
            if kitchen_final == "solidcondition" and bath_final == "needsrehab":
                if has_other_imgs:
                    # If none of the other images are needs/full, treat as solid
                    if others_needs_count == 0 and len(others_full) == 0:
                        property_tier = "solidcondition"
                    else:
                        # Some other rooms show issues → treat as needs rehab
                        property_tier = "needsrehab"
                else:
                    # Only kitchen/bath images → be safe
                    property_tier = "needsrehab"

        if property_tier is None:
            # 2) Kitchen solid + bathroom full rehab → minimum needs rehab
            if kitchen_final == "solidcondition" and bath_final == "fullrehab":
                property_tier = "needsrehab"

        if property_tier is None:
            # 3) Kitchen needs rehab + bathroom fully updated
            if kitchen_final == "needsrehab" and bath_final == "fullyupdated":
                if has_other_imgs:
                    if others_needs_count >= 3:
                        property_tier = "needsrehab"
                    else:
                        # Light issues evidenced → treat as solid
                        property_tier = "solidcondition"
                else:
                    # Only kitchen/bath images → be safe
                    property_tier = "needsrehab"

        if property_tier is None:
            # 4) Kitchen full rehab + bathroom needs rehab → full rehab
            if kitchen_final == "fullrehab" and bath_final == "needsrehab":
                property_tier = "fullrehab"

        # ------------------------------------------------------------------
        # IF STILL UNCLASSIFIED → GENERAL FALLBACK USING K/B + FULLY UPDATED RULE
        # ------------------------------------------------------------------
        if property_tier is None:
            # Any kitchen/bath fullrehab forces at least needsrehab
            if kitchen_final == "fullrehab" or bath_final == "fullrehab":
                # If both fullrehab, treat as full gut; otherwise medium as floor.
                if kitchen_final == "fullrehab" and bath_final == "fullrehab":
                    property_tier = "fullrehab"
                else:
                    property_tier = "needsrehab"
            else:
                # Fully updated rule: K + B updated + >10 images + ≥80% fullyupdated
                if (
                    kitchen_final == "fullyupdated"
                    and bath_final == "fullyupdated"
                    and total_imgs > 10
                    and fully_ratio_all >= 0.80
                ):
                    property_tier = "fullyupdated"
                else:
                    # If K & B at least solid, treat as solid; otherwise needs.
                    if kitchen_final in ("solidcondition", "fullyupdated") and bath_final in (
                        "solidcondition",
                        "fullyupdated",
                    ):
                        property_tier = "solidcondition"
                    else:
                        property_tier = "needsrehab"

    # ------------------------------------------------------------------
    # COST MAPS
    # ------------------------------------------------------------------
    kitchen_cost_map = {
        "fullyupdated": 0,
        "solidcondition": 3000,
        "needsrehab": 12000,
        "fullrehab": 25000,
    }
    bath_cost_map = {
        "fullyupdated": 0,
        "solidcondition": 2000,
        "needsrehab": 7500,
        "fullrehab": 12000,
    }

    kitchen_cost = kitchen_cost_map.get(kitchen_final, 0) * _safe_float(units, 1.0)
    bath_cost = bath_cost_map.get(bath_final, 0) * _safe_float(baths, 1.0)

    roof_cost = 12000 if roof_needed else 0
    hvac_cost = 7500 if hvac_needed else 0

    foundation_cost = 0
    if foundation_issues == "minor_issues":
        foundation_cost = 8000
    elif foundation_issues == "major_issues":
        foundation_cost = 18000

    # ------------------------------------------------------------------
    # SQFT-BASED INTERIOR COST FROM PROPERTY TIER
    # ------------------------------------------------------------------
    sqft_cost = _sqft_cost_from_tier(sqft, property_tier)

    estimate = sqft_cost + kitchen_cost + bath_cost + roof_cost + hvac_cost + foundation_cost
    estimate_str = f"${estimate:,.0f}"

    # For compatibility, expose a "tier" numeric; use the primary band rate.
    if property_tier == "fullyupdated":
        base_mult = 5
    elif property_tier == "solidcondition":
        base_mult = 10
    elif property_tier == "needsrehab":
        base_mult = 15
    else:
        # fullrehab or anything else
        base_mult = 15

    # ------------------------------------------------------------------
    # ✅ OVERRIDE: Fully Updated + >6 VALIDATED INTERIOR IMAGES + ≥80% fullyupdated
    #            + must include at least 1 kitchen image
    #            + no roof/hvac/foundation -> show "<$10,000" and numeric 10000
    # ------------------------------------------------------------------
    INTERIOR_ROOM_TYPES = {
        "kitchen",
        "bathroom",
        "bedroom",
        "living_room",
        "dining_room",
        "family_room",
        "basement",
        "attic",
        "hallway",
        "laundry",
        "office",
        "den",
        "bonus_room",
    }

    interior_valid = [
        r for r in image_results
        if (r.get("room_type") in INTERIOR_ROOM_TYPES)
    ]
    interior_count = len(interior_valid)
    interior_has_kitchen = any(r.get("room_type") == "kitchen" for r in interior_valid)

    if interior_count > 0:
        interior_fully = sum(1 for r in interior_valid if r.get("condition") == "fullyupdated")
        interior_fully_ratio = interior_fully / float(interior_count)
    else:
        interior_fully_ratio = 0.0

    if (
        subject_condition_label == "fullyupdated"
        and interior_count > 9
        and interior_has_kitchen
        and interior_fully_ratio >= 0.80
        and roof_cost == 0
        and hvac_cost == 0
        and foundation_cost == 0
    ):
        property_tier = "fullyupdated"
        base_mult = 5

        # Force fully-updated presentation for this special case
        kitchen_final = "fullyupdated"
        bath_final = "fullyupdated"
        kitchen_cost = 0
        bath_cost = 0

        estimate = 10000
        estimate_str = "<$10,000"

        # Keep sqft_cost consistent with the displayed estimate in this case
        sqft_cost = 10000

    return {
        "tier": base_mult,
        "rate_per_sqft": base_mult,
        "property_tier": property_tier,
        "estimate": estimate,
        "estimate_numeric": estimate,
        "estimate_str": estimate_str,
        "kitchen_condition": kitchen_final,
        "bath_condition": bath_final,
        "kitchen_cost": kitchen_cost,
        "bath_cost": bath_cost,
        "roof_cost": roof_cost,
        "hvac_cost": hvac_cost,
        "foundation_cost": foundation_cost,
        "sqft_cost": sqft_cost,
        "total_images": total_imgs,
        "total_other_strong": total_other_strong,
        "others_full_count": others_full_count,
        "others_needs_count": others_needs_count,
    }
