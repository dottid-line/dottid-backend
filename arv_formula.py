# arv_formula.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import os
from datetime import datetime, timezone

ARV_DEBUG = False
ROUND_TO = 5000

DAYS_6MO = 183
DAYS_12MO = 365

FULLY_UPDATED_MAX = 1.0
TIER_15_MAX = 1.5
TIER_20_MAX = 2.0

SQFT_DING_20 = 0.20
SQFT_DING_25 = 0.25

CONDITION_UPLIFT_PPSF_PER_POINT = 15.0


def _num(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("$", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        v = _num(x, None)
        if v is None:
            return default
        return int(round(v))
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _get_any(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None and str(d.get(k)).strip() != "":
            return d.get(k)
    return None


def _norm_addr(s: Any) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\b(apt|apartment|unit|ste|suite|#)\b.*$", "", s).strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _condition_value(comp: Dict[str, Any]) -> Optional[float]:
    v = _get_any(comp, ["condition_score", "condition", "condition_rating", "cond_score", "final_score"])
    return _num(v, None)


def _unusable_reason(c: Dict[str, Any]) -> Optional[str]:
    sold_price = _num(c.get("sold_price"), None)
    comp_sqft = _num(c.get("comp_sqft"), None)
    dist = _num(c.get("distance_miles"), None)
    cond = _condition_value(c)

    if sold_price is None or sold_price <= 0:
        return "MISSING_SOLD_PRICE"
    if comp_sqft is None or comp_sqft <= 0:
        return "MISSING_COMP_SQFT"
    if dist is None:
        return "MISSING_DISTANCE"
    if cond is None:
        return "MISSING_CONDITION_SCORE"

    skip_reason = _safe_str(c.get("skip_reason"))
    if skip_reason in {"NO_KITCHEN", "INSUFFICIENT_INTERIOR_IMAGES", "INSUFFICIENT_KITCHEN_IMAGES"}:
        return skip_reason

    return None


def _distance_score(miles: float) -> float:
    if miles <= 0.10:
        return 1.00
    if miles <= 0.25:
        return 0.95
    if miles <= 0.50:
        return 0.90
    if miles <= 0.75:
        return 0.85
    if miles <= 1.00:
        return 0.80
    if miles <= 1.25:
        return 0.75
    if miles <= 1.50:
        return 0.70
    return 0.60


def _ratio_weight(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.70
    return max(0.40, 1.0 - abs(a - b) / max(a, b))


def _estimate_with_breakdown(subject: Dict[str, Any], comp: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    sold_price = _num(comp.get("sold_price"), None)
    comp_sqft = _num(comp.get("comp_sqft"), None)
    subj_sqft = _num(subject.get("sqft"), None)

    if sold_price is None or comp_sqft is None or subj_sqft is None or sold_price <= 0 or comp_sqft <= 0 or subj_sqft <= 0:
        return None, {"reason": "MISSING_PRICE_OR_SQFT"}

    base_ppsf = sold_price / comp_sqft

    cond_score = _condition_value(comp)
    if cond_score is None:
        cond_score = 2.0

    uplift_points = max(0.0, cond_score - FULLY_UPDATED_MAX)
    uplift_ppsf = uplift_points * CONDITION_UPLIFT_PPSF_PER_POINT
    ppsf = base_ppsf + uplift_ppsf

    sb = _num(subject.get("beds"), 0.0) or 0.0
    sba = _num(subject.get("baths"), 0.0) or 0.0
    cb = _num(_get_any(comp, ["comp_beds", "beds"]), sb) or sb
    cba = _num(_get_any(comp, ["comp_baths", "baths"]), sba) or sba

    bed_diff = sb - cb
    bath_diff = sba - cba

    bed_adj_pct = max(-0.04, min(0.04, bed_diff * 0.02))
    bath_adj_pct = max(-0.04, min(0.04, bath_diff * 0.02))

    sqft_raw_diff = subj_sqft - comp_sqft
    sqft_adj_pct = 0.0

    if abs(sqft_raw_diff) > 150:
        excess_sqft = abs(sqft_raw_diff) - 150
        sqft_adj_pct = excess_sqft / 100.0 * 0.01
        if sqft_raw_diff < 0:
            sqft_adj_pct *= -1.0
        sqft_adj_pct = max(-0.03, min(0.03, sqft_adj_pct))

    total_pct = 1.0 + bed_adj_pct + bath_adj_pct + sqft_adj_pct

    ppsf_after = ppsf * total_pct
    est = ppsf_after * float(subj_sqft)

    return max(1.0, est), {}


def _round_to(x: float, base: int) -> int:
    if base <= 0:
        return int(round(x))
    return int(round(x / base) * base)


def _normalize_selected_comp(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the selected comps so downstream code (pipeline/UI) can reliably read:
      - zpid
      - detail_url
      - beds, baths, comp_sqft
      - distance_miles (numeric)
    Keeps ALL original keys; only fills missing standardized fields.
    """
    out = dict(c)

    raw = out.get("raw") if isinstance(out.get("raw"), dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    # distance_miles: ensure numeric if possible
    if out.get("distance_miles") is None:
        out["distance_miles"] = _num(out.get("miles") or out.get("miles_exact"), None)
    else:
        out["distance_miles"] = _num(out.get("distance_miles"), None)

    # zpid: prefer explicit, else from raw common keys
    if out.get("zpid") is None or str(out.get("zpid")).strip() == "":
        zpid = _get_any(raw, ["zpid", "id", "zillowId", "zillow_id"])
        if zpid is None:
            zpid = _get_any(out, ["id", "zillowId", "zillow_id"])
        out["zpid"] = zpid

    # detail_url: prefer explicit, else from raw common keys
    if out.get("detail_url") is None or str(out.get("detail_url")).strip() == "":
        detail_url = _get_any(raw, ["detailUrl", "detail_url", "url", "hdpUrl", "hdp_url", "link"])
        if detail_url is None:
            detail_url = _get_any(out, ["detailUrl", "url", "link"])
        out["detail_url"] = detail_url

    # beds/baths: standardize from comp_beds/comp_baths first, then raw fallbacks
    if out.get("beds") is None or str(out.get("beds")).strip() == "":
        beds = _get_any(out, ["comp_beds", "beds"])
        if beds is None:
            beds = _get_any(raw, ["beds", "bedrooms"])
        out["beds"] = beds

    if out.get("baths") is None or str(out.get("baths")).strip() == "":
        baths = _get_any(out, ["comp_baths", "baths"])
        if baths is None:
            baths = _get_any(raw, ["baths", "bathrooms"])
        out["baths"] = baths

    # comp_sqft: ensure present/standard; fallback to raw
    if out.get("comp_sqft") is None or str(out.get("comp_sqft")).strip() == "":
        sqft = _get_any(out, ["comp_sqft"])
        if sqft is None:
            sqft = _get_any(raw, ["area", "livingArea", "sqft", "living_area"])
        out["comp_sqft"] = sqft

    return out


def compute_arv(subject: Dict[str, Any], comps: List[Dict[str, Any]], total_comps_returned=None) -> Dict[str, Any]:
    _ = total_comps_returned  # safely ignore if passed

    total = len(comps or [])

    subj_addr = _safe_str(subject.get("address"))
    subj_sqft = _num(subject.get("sqft"), None)
    subj_beds = _num(subject.get("beds"), None)
    subj_baths = _num(subject.get("baths"), None)

    if not subj_addr or subj_sqft is None or subj_sqft <= 0 or subj_beds is None or subj_baths is None:
        return {
            "status": "fail",
            "message": "SUBJECT_MISSING_REQUIRED_FIELDS",
            "arv": None,
            "selected_comps": [],
        }

    if total < 2:
        return {
            "status": "fail",
            "message": "NOT_ENOUGH_COMPS",
            "arv": None,
            "selected_comps": [],
        }

    usable = []
    for c in comps or []:
        if not isinstance(c, dict):
            continue

        reason = _unusable_reason(c)
        if reason:
            continue

        usable.append(dict(c))

    if len(usable) < 2:
        return {
            "status": "fail",
            "message": "NOT_ENOUGH_USABLE_COMPS",
            "arv": None,
            "selected_comps": [],
        }

    # CHANGE: apply condition-score thresholds that depend on the scenario
    # - 2-comp scenario: each comp must have condition_score >= 1.5
    # - 3-comp scenario (top-3 eligibility): comp must have condition_score >= 1.7
    if len(usable) == 2:
        usable = [c for c in usable if (_condition_value(c) is not None and float(_condition_value(c)) >= 1.5)]
        if len(usable) < 2:
            return {
                "status": "fail",
                "message": "NOT_ENOUGH_USABLE_COMPS",
                "arv": None,
                "selected_comps": [],
            }
    else:
        usable = [c for c in usable if (_condition_value(c) is not None and float(_condition_value(c)) >= 1.7)]
        if len(usable) < 2:
            return {
                "status": "fail",
                "message": "NOT_ENOUGH_USABLE_COMPS",
                "arv": None,
                "selected_comps": [],
            }

    usable_sorted = sorted(usable, key=lambda c: (_num(c.get("distance_miles"), 9e9),))

    # CHANGE: select up to 3 comps from the post-threshold usable pool
    selected = usable_sorted[:3]

    # CHANGE: if exactly 2 comps are used, reject if sold_price spread is > 20%
    if len(selected) == 2:
        p1 = _num(selected[0].get("sold_price"), None)
        p2 = _num(selected[1].get("sold_price"), None)
        if p1 is None or p2 is None or p1 <= 0 or p2 <= 0:
            return {
                "status": "fail",
                "message": "NOT_ENOUGH_USABLE_COMPS",
                "arv": None,
                "selected_comps": [],
            }
        hi = max(p1, p2)
        lo = min(p1, p2)
        if lo <= 0:
            return {
                "status": "fail",
                "message": "NOT_ENOUGH_USABLE_COMPS",
                "arv": None,
                "selected_comps": [],
            }
        spread_pct = (hi - lo) / lo
        if spread_pct > 0.20:
            return {
                "status": "fail",
                "message": "NOT_ENOUGH_USABLE_COMPS",
                "arv": None,
                "selected_comps": [],
            }

    estimates = []
    weights = []

    for c in selected:
        est, _dbg = _estimate_with_breakdown(subject, c)
        if est is None:
            continue

        miles = _num(c.get("distance_miles"), 9e9) or 9e9
        pick_score = _distance_score(miles)

        cond_score = _condition_value(c)
        if cond_score is None:
            cond_score = 2.0

        condition_weight = (1.0 / (1.0 + cond_score)) ** 2
        final_w = max(0.0001, pick_score * condition_weight)

        estimates.append(est)
        weights.append(final_w)

    if len(estimates) < 1:
        return {
            "status": "fail",
            "message": "NO_ESTIMATES_POSSIBLE",
            "arv": None,
            "selected_comps": [],
        }

    num = sum(est * w for est, w in zip(estimates, weights))
    den = sum(weights)

    if den <= 0:
        return {
            "status": "fail",
            "message": "BAD_WEIGHT_SUM",
            "arv": None,
            "selected_comps": [],
        }

    arv_raw = num / den

    arv_floor = min(estimates)
    arv_ceiling = max(estimates)
    arv_clamped = max(arv_floor, min(arv_raw, arv_ceiling))
    arv_rounded = _round_to(arv_clamped, ROUND_TO)

    selected_out = [_normalize_selected_comp(c) for c in selected]

    return {
        "status": "ok",
        "message": "ARV computed",
        "arv": int(arv_rounded),
        "selected_comps": selected_out,
    }
