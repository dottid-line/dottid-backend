# comp_ranker.py

import json
import math
import os
import urllib.parse
import re
from typing import Any, Dict, List, Tuple, Optional

import requests

MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoiZG90dGlkbGluZSIsImEiOiJjbWoxenNoejgwMWxmM2ZxMmI2bHBtcGFvIn0.cFLHiAGxdfHJlT-9cQW6lg",
).strip()

MAPBOX_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places/{}.json"

MAX_COMPS = 15
MIN_WEIGHT_CUTOFF = 0.35

PRIORITY_MAX_MILES = 0.5
PRIORITY_MAX_BED_DIFF = 1
PRIORITY_MAX_BATH_DIFF = 1
PRIORITY_MAX_SQFT_DIFF = 150

ENABLE_COMP_GEOCODE_FALLBACK = os.environ.get("ENABLE_COMP_GEOCODE_FALLBACK", "1").strip() != "0"
MAX_COMP_GEOCODE_FALLBACKS = int(os.environ.get("MAX_COMP_GEOCODE_FALLBACKS", "60"))

def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _parse_street_name(full_addr: str) -> str:
    s = _safe_str(full_addr).lower()
    if not s:
        return ""
    s = s.split(",")[0].strip()
    if not s:
        return ""
    parts = s.split()
    if len(parts) >= 2 and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts).strip()

def _norm_addr_piece(s: str) -> str:
    s = _safe_str(s).lower().strip()
    if not s:
        return ""
    s = re.sub(r"\b(apt|apartment|unit|ste|suite|#)\b.*$", "", s).strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _addr_line(addr: str) -> str:
    return _safe_str(addr).split(",")[0].strip()

def _addr_tail(addr: str) -> str:
    parts = [p.strip() for p in _safe_str(addr).split(",")]
    if len(parts) <= 1:
        return ""
    return ", ".join(parts[1:]).strip()

def _extract_zip(s: str) -> str:
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", _safe_str(s))
    return m.group(1) if m else ""

def _extract_state(s: str) -> str:
    tokens = re.split(r"[\s,]+", _safe_str(s).upper())
    for t in tokens:
        if len(t) == 2 and t.isalpha():
            return t
    return ""

def _is_subject_comp(subject_address: str, comp_address: str) -> bool:
    if not subject_address or not comp_address:
        return False
    subj_line = _norm_addr_piece(_addr_line(subject_address))
    comp_line = _norm_addr_piece(_addr_line(comp_address))
    if not subj_line or not comp_line:
        return False
    if subj_line != comp_line:
        return False
    subj_zip = _extract_zip(subject_address)
    comp_zip = _extract_zip(comp_address)
    if subj_zip and comp_zip:
        return subj_zip == comp_zip
    subj_state = _extract_state(_addr_tail(subject_address))
    comp_state = _extract_state(_addr_tail(comp_address))
    if subj_state and comp_state:
        return subj_state == comp_state
    return True

def extract_comp_address(c: Dict[str, Any]) -> str:
    addr = _first_nonempty(
        c.get("address"),
        c.get("streetAddress"),
        c.get("street_address"),
        c.get("addressStreet"),
        c.get("address_street"),
    )
    city = _first_nonempty(c.get("city"), c.get("addressCity"), c.get("address_city"))
    state = _first_nonempty(c.get("state"), c.get("addressState"), c.get("address_state"))
    zipcode = _first_nonempty(c.get("zipcode"), c.get("zip"), c.get("addressZipcode"), c.get("address_zip"))
    if isinstance(addr, str) and addr.strip():
        if city or state or zipcode:
            parts = [addr.strip()]
            tail = " ".join([str(x).strip() for x in [city, state, zipcode] if x is not None and str(x).strip()])
            if tail:
                parts.append(tail)
            return ", ".join(parts)
        return addr.strip()
    return ""

def extract_comp_beds(c: Dict[str, Any]) -> float:
    return _num(_first_nonempty(c.get("beds"), c.get("bedrooms"), c.get("bed"), c.get("bds")), 0.0)

def extract_comp_baths(c: Dict[str, Any]) -> float:
    return _num(_first_nonempty(c.get("baths"), c.get("bathrooms"), c.get("bath"), c.get("bas")), 0.0)

def extract_comp_area(c: Dict[str, Any]) -> float:
    return _num(_first_nonempty(c.get("area"), c.get("livingArea"), c.get("sqft"), c.get("living_area")), 0.0)

def extract_comp_price(c: Dict[str, Any]) -> Any:
    return _first_nonempty(c.get("unformattedPrice"), c.get("price"), c.get("soldPrice"), c.get("sold_price"))

def geocode_address(address: str) -> Tuple[float, float]:
    if not MAPBOX_TOKEN:
        raise RuntimeError("MAPBOX_TOKEN missing.")
    url = MAPBOX_GEOCODE_URL.format(urllib.parse.quote(address))
    r = requests.get(url, params={"access_token": MAPBOX_TOKEN, "limit": 1}, timeout=15)
    r.raise_for_status()
    data = r.json()
    feats = data.get("features") or []
    if not feats:
        raise ValueError(f"Cannot geocode {address}")
    lon, lat = feats[0]["center"]
    return float(lat), float(lon)

_comp_geocode_cache: Dict[str, Tuple[float, float]] = {}
_comp_geocode_fallbacks_used = 0

def geocode_comp_cached(address: str) -> Optional[Tuple[float, float]]:
    global _comp_geocode_fallbacks_used
    if not ENABLE_COMP_GEOCODE_FALLBACK:
        return None
    a = _safe_str(address)
    if not a:
        return None
    if a in _comp_geocode_cache:
        return _comp_geocode_cache[a]
    if _comp_geocode_fallbacks_used >= MAX_COMP_GEOCODE_FALLBACKS:
        return None
    try:
        lat, lon = geocode_address(a)
        _comp_geocode_cache[a] = (lat, lon)
        _comp_geocode_fallbacks_used += 1
        return (lat, lon)
    except Exception:
        return None

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def distance_weight(miles: float) -> float:
    if miles <= 0.25:
        return 1.0
    if miles <= 0.50:
        return 0.90
    if miles <= 0.75:
        return 0.80
    if miles <= 1.00:
        return 0.70
    if miles <= 1.25:
        return 0.60
    if miles <= 1.50:
        return 0.50
    return 0.40

def ratio_weight(a: float, b: float) -> float:
    a = float(a or 0)
    b = float(b or 0)
    if a <= 0 or b <= 0:
        return 0.70
    return max(0.40, 1.0 - abs(a - b) / max(a, b))

def extract_latlon(comp: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    ll = comp.get("latLong") or {}
    lat = ll.get("latitude", comp.get("latitude"))
    lon = ll.get("longitude", comp.get("longitude"))
    lat_f = _num(lat, 0.0)
    lon_f = _num(lon, 0.0)
    if lat_f == 0.0 and lon_f == 0.0:
        return None
    return lat_f, lon_f

def is_priority_comp(sub_beds, sub_baths, sub_area, comp_beds, comp_baths, comp_area, miles):
    if miles is None or miles > PRIORITY_MAX_MILES:
        return False
    if sub_beds <= 0 or sub_baths <= 0 or sub_area <= 0:
        return False
    if comp_beds <= 0 or comp_baths <= 0 or comp_area <= 0:
        return False
    if abs(comp_beds - sub_beds) > PRIORITY_MAX_BED_DIFF:
        return False
    if abs(comp_baths - sub_baths) > PRIORITY_MAX_BATH_DIFF:
        return False
    if abs(comp_area - sub_area) > PRIORITY_MAX_SQFT_DIFF:
        return False
    return True

def _sort_key_distance_first(x: Dict[str, Any]) -> Tuple[float, float]:
    return (_num(x.get("miles_exact"), 9e9), -_num(x.get("weight"), 0.0))

def _dedupe_key(x: Dict[str, Any]) -> str:
    raw = x.get("raw") or {}
    zpid = raw.get("zpid") or raw.get("id") or raw.get("zillowId") or raw.get("zillow_id")
    if zpid is not None:
        return f"zpid:{zpid}"
    return f"addr:{(x.get('address') or '').lower()}"

def _print_top_15(ranked: List[Dict[str, Any]]) -> None:
    print("\n=== TOP 15 COMPS (SAME-STREET LOCK + NORMAL RULES) ===\n")
    if not ranked:
        print("No comps.\n")
        return
    for i, c in enumerate(ranked[:MAX_COMPS], 1):
        raw = c.get("raw") or {}
        tags = []
        if c.get("same_street_forced"):
            tags.append("SAME_STREET_FORCED")
        if c.get("priority"):
            tags.append("PRIORITY")
        tag_str = (" [" + ", ".join(tags) + "]") if tags else ""
        print(
            f"{i:>2}. {c.get('address')} | {c.get('miles')} mi | "
            f"beds={raw.get('beds') or raw.get('bedrooms')} "
            f"baths={raw.get('baths') or raw.get('bathrooms')} "
            f"sqft={raw.get('area') or raw.get('livingArea') or raw.get('sqft')} | "
            f"weight={c.get('weight')}{tag_str}"
        )
    print("")

def rank_comps(subject: Dict[str, Any], comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    address = _safe_str(subject.get("address"))
    if not address:
        raise ValueError("Subject missing address")

    sub_beds = _num(subject.get("beds"), 0)
    sub_baths = _num(subject.get("baths"), 0)
    sub_area = _num(subject.get("sqft") or subject.get("area"), 0)

    if sub_beds <= 0 or sub_baths <= 0 or sub_area <= 0:
        raise ValueError("Subject must include beds, baths, sqft")

    subject_street = _parse_street_name(address)
    if not subject_street:
        raise ValueError("Could not parse subject street")

    print("\nGeocoding subject property...")
    subject_lat, subject_lon = geocode_address(address)

    scored: List[Dict[str, Any]] = []

    for c in comps:
        comp_address = extract_comp_address(c)
        if not comp_address:
            continue
        if _is_subject_comp(address, comp_address):
            continue

        comp_price = extract_comp_price(c)
        comp_beds = extract_comp_beds(c)
        comp_baths = extract_comp_baths(c)
        comp_area = extract_comp_area(c)

        ll = extract_latlon(c)
        if ll is None:
            ll = geocode_comp_cached(comp_address)
        if ll is None:
            continue

        lat, lon = ll
        miles_exact = haversine_miles(subject_lat, subject_lon, lat, lon)

        w_dist = distance_weight(miles_exact)
        w_sqft = ratio_weight(sub_area, comp_area)
        w_beds = ratio_weight(sub_beds, comp_beds)
        w_baths = ratio_weight(sub_baths, comp_baths)

        weight = round(float(w_dist * w_sqft * w_beds * w_baths), 3)

        priority = is_priority_comp(
            sub_beds=sub_beds,
            sub_baths=sub_baths,
            sub_area=sub_area,
            comp_beds=comp_beds,
            comp_baths=comp_baths,
            comp_area=comp_area,
            miles=miles_exact,
        )

        comp_street = _parse_street_name(comp_address)
        same_street = (comp_street != "" and comp_street == subject_street)

        scored.append(
            {
                "address": comp_address,
                "price": comp_price,
                "miles": round(float(miles_exact), 2),
                "miles_exact": float(miles_exact),
                "weight": weight,
                "priority": priority,
                "same_street": same_street,
                "same_street_forced": False,
                "raw": c,
            }
        )

    same_street_comps = [x for x in scored if x.get("same_street")]
    other_comps = [x for x in scored if not x.get("same_street")]

    same_street_comps.sort(key=lambda x: (_num(x.get("miles_exact"), 9e9), -_num(x.get("weight"), 0.0)))

    chosen: List[Dict[str, Any]] = []
    seen = set()

    for x in same_street_comps:
        k = _dedupe_key(x)
        if k in seen:
            continue
        x["same_street_forced"] = True
        chosen.append(x)
        seen.add(k)
        if len(chosen) >= MAX_COMPS:
            _print_top_15(chosen)
            return chosen[:MAX_COMPS]

    priority_comps = [x for x in other_comps if x.get("priority")]
    non_priority = [x for x in other_comps if not x.get("priority")]

    priority_comps.sort(key=_sort_key_distance_first)
    non_priority.sort(key=_sort_key_distance_first)

    for x in priority_comps:
        k = _dedupe_key(x)
        if k in seen:
            continue
        chosen.append(x)
        seen.add(k)
        if len(chosen) >= MAX_COMPS:
            _print_top_15(chosen)
            return chosen[:MAX_COMPS]

    for x in non_priority:
        if _num(x.get("weight"), 0.0) < MIN_WEIGHT_CUTOFF:
            continue
        k = _dedupe_key(x)
        if k in seen:
            continue
        chosen.append(x)
        seen.add(k)
        if len(chosen) >= MAX_COMPS:
            _print_top_15(chosen)
            return chosen[:MAX_COMPS]

    for x in non_priority:
        k = _dedupe_key(x)
        if k in seen:
            continue
        chosen.append(x)
        seen.add(k)
        if len(chosen) >= MAX_COMPS:
            break

    _print_top_15(chosen)
    return chosen[:MAX_COMPS]
