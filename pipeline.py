# pipeline.py
import os
import time
import threading
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import re
import inspect
import json
import urllib.parse
from collections import Counter
from typing import Any
import base64

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent

SEARCH_ACTOR_ID = "api-empire~zillow-search-scraper"
DETAIL_ACTOR_ID = "maxcopell~zillow-detail-scraper"

# ✅ IMPORTANT: do NOT hardcode tokens in code (GitHub blocks pushes)
# Set APIFY_TOKEN in your environment (Render → Environment Variables)
DEFAULT_APIFY_TOKEN = os.environ.get("APIFY_TOKEN", "").strip()

os.environ.setdefault(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoiZG90dGlkbGluZSIsImEiOiJjbWoxenNoejgwMWxmM2ZxMmI2bHBtcGFvIn0.cFLHiAGxdfHJlT-9cQW6lg",
)

MAX_COMPS_TO_SCORE = 15
SIMILARITY_THRESHOLD = 0.35

# Median-outlier rule:
OUTLIER_MEDIAN_FLOOR_RATIO = 0.60
OUTLIER_MEDIAN_CEIL_RATIO = 1.40
OUTLIER_MIN_COMPS_FOR_MEDIAN = 4  # apply outlier filter only if >=4 comps eligible for median

# Backfill rule thresholds (used only if we have < 15 comps after similarity threshold)
FILL_PRIORITY_MAX_MILES = 1.0
FILL_PRIORITY_MAX_BED_DIFF = 1
FILL_PRIORITY_MAX_BATH_DIFF = 1
FILL_PRIORITY_MAX_SQFT_DIFF = 150

# Trigger 12-month top-up ONLY if 6-month comps < 12
TRIGGER_12MO_IF_6MO_LESS_THAN = 12
MIN_COMPS_AFTER_12MO_BEFORE_RELAX = 6

MAX_IMAGES_TO_DOWNLOAD = 70
MIN_REQUIRED_IMAGES = 3  # HARD GUARD: if < this, do not score condition (skip as insufficient images)
TARGET_IMAGE_WIDTH = 384
MIN_IMAGE_WIDTH = 320

APIFY_POLL_SECONDS = 3
APIFY_TIMEOUT_SEC = 300

# ---- DOWNLOAD PERFORMANCE (SAFE CAPS) ----
def _env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, "")).strip())
        return v
    except Exception:
        return default


GLOBAL_DOWNLOAD_WORKERS = _env_int("DOWNLOAD_WORKERS", 8)
GLOBAL_MAX_INFLIGHT = _env_int("MAX_INFLIGHT", 4)
IMG_MAX_RETRIES = _env_int("IMG_MAX_RETRIES", 4)

GLOBAL_DOWNLOAD_WORKERS = max(8, min(GLOBAL_DOWNLOAD_WORKERS, 256))
GLOBAL_MAX_INFLIGHT = max(4, min(GLOBAL_MAX_INFLIGHT, 64))
IMG_MAX_RETRIES = max(1, min(IMG_MAX_RETRIES, 10))

IMG_CONNECT_TIMEOUT_SEC = 6
IMG_READ_TIMEOUT_SEC = 20
IMG_BACKOFF_BASE_SEC = 0.7

# How many comps to process in parallel at the “per-comp” level (detail->download->score)
COMP_WORKERS = 3

DEBUG_DOWNLOAD = os.environ.get("DEBUG_DOWNLOAD", "0").strip() == "1"

_thread_local = threading.local()
_inflight_sem = threading.Semaphore(GLOBAL_MAX_INFLIGHT)
_dl_lock = threading.Lock()
_dl_status = Counter()
_dl_ctypes = Counter()
_DOWNLOAD_POOL: ThreadPoolExecutor | None = None


# ============================================================
# Requests sessions
# ============================================================
def get_thread_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is not None:
        return s
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/jpeg,image/png,image/*;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    _thread_local.session = s
    return s


def apify_token() -> str:
    return os.environ.get("APIFY_TOKEN", DEFAULT_APIFY_TOKEN)


def create_apify_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": "DottidAI/1.0 (+https://dottid.ai)",
            "Connection": "keep-alive",
            "Accept": "application/json",
        }
    )
    return s


def apify_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params=None,
    json=None,
    timeout=(10, 120),
):
    last_err = None
    for attempt in range(6):
        try:
            return session.request(method, url, params=params, json=json, timeout=timeout)
        except Exception as e:
            last_err = e
            time.sleep((0.7 * (2**attempt)) + random.uniform(0, 0.25))
    raise last_err  # type: ignore[misc]


def apify_post_run(session: requests.Session, actor_id: str, payload: dict) -> str:
    url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
    r = apify_request(session, "POST", url, params={"token": apify_token()}, json=payload, timeout=(10, 180))
    r.raise_for_status()
    return r.json()["data"]["id"]


def apify_wait_run(session: requests.Session, run_id: str) -> None:
    status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    while True:
        r = apify_request(session, "GET", status_url, params={"token": apify_token()}, timeout=(10, 60))
        r.raise_for_status()
        status = r.json()["data"]["status"]
        if status == "SUCCEEDED":
            return
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Apify run ended with status={status}")
        time.sleep(APIFY_POLL_SECONDS)


def apify_get_run_dataset_items(session: requests.Session, run_id: str):
    url = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items"
    r = apify_request(session, "GET", url, params={"token": apify_token(), "clean": "true"}, timeout=(10, 180))
    r.raise_for_status()
    return r.json()


def apify_run_sync_get_dataset_items(session: requests.Session, actor_id: str, payload: dict, timeout_sec: int = 300):
    url = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    r = apify_request(
        session,
        "POST",
        url,
        params={"token": apify_token(), "clean": "true"},
        json=payload,
        timeout=(10, timeout_sec + 30),
    )
    r.raise_for_status()
    return r.json()


def _run_search_scrape(apify_session: requests.Session, zillow_url: str) -> list[dict]:
    payload = {"searchUrls": [{"url": zillow_url}]}
    run_id = apify_post_run(apify_session, SEARCH_ACTOR_ID, payload)
    apify_wait_run(apify_session, run_id)
    items = apify_get_run_dataset_items(apify_session, run_id)
    return items if isinstance(items, list) else []


# ============================================================
# Dedupe / similarity
# ============================================================
def _dedupe_comps(comps: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in comps:
        if not isinstance(c, dict):
            continue
        raw = c.get("raw") or {}
        zpid = raw.get("zpid") or raw.get("id") or c.get("zpid") or c.get("id")
        if zpid is not None:
            key = f"zpid:{zpid}"
        else:
            u = raw.get("detailUrl") or raw.get("detail_url") or raw.get("url") or c.get("url")
            addr = c.get("address") or raw.get("address")
            if isinstance(u, str) and u.startswith("http"):
                key = f"url:{u.strip()}"
            elif isinstance(addr, str) and addr.strip():
                key = f"addr:{addr.strip().lower()}"
            else:
                continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _get_similarity_value(comp: dict) -> float | None:
    for k in ("similarity", "similarity_score", "sim", "sim_score", "match_score", "score", "weight"):
        v = comp.get(k)
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            pass
    return None


def _counts_toward_target(comp: dict, threshold: float) -> bool:
    s = _get_similarity_value(comp)
    if s is None:
        return True
    return s >= threshold


def _apply_similarity_threshold(ranked: list[dict], threshold: float) -> list[dict]:
    kept = []
    for c in ranked:
        if _counts_toward_target(c, threshold):
            kept.append(c)
    return kept


# ============================================================
# FIX: Money parsing + best-price selection
# ============================================================
def _parse_money_to_int(x: Any) -> int | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            return int(round(float(x)))
        except Exception:
            return None

    s = str(x).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip().upper()

    mult = 1
    if s.endswith("K"):
        mult = 1000
        s = s[:-1].strip()
    elif s.endswith("M"):
        mult = 1_000_000
        s = s[:-1].strip()
    elif s.endswith("B"):
        mult = 1_000_000_000
        s = s[:-1].strip()

    try:
        return int(round(float(s) * mult))
    except Exception:
        return None


def _best_price_int(*vals: Any, min_dollars: int = 1000) -> int | None:
    for v in vals:
        p = _parse_money_to_int(v)
        if p is None:
            continue
        if p >= min_dollars:
            return p
    return None


def _extract_price_from_detail(item: dict) -> int | None:
    if not isinstance(item, dict):
        return None

    p = _best_price_int(
        item.get("soldPrice"),
        item.get("sold_price"),
        item.get("unformattedPrice"),
        item.get("price"),
        item.get("formattedPrice"),
    )
    if p is not None:
        return p

    hdp = item.get("hdpData") or {}
    home = hdp.get("homeInfo") or {}
    p2 = _best_price_int(
        home.get("soldPrice"),
        home.get("price"),
        home.get("unformattedPrice"),
        home.get("formattedPrice"),
    )
    if p2 is not None:
        return p2

    p3 = _best_price_int(item.get("formattedPrice"), home.get("formattedPrice"))
    return p3


def _comp_best_price_from_search(comp: dict) -> int | None:
    raw = comp.get("raw") or {}
    return _best_price_int(
        comp.get("sold_price"),
        comp.get("soldPrice"),
        comp.get("price"),
        comp.get("unformattedPrice"),
        comp.get("formattedPrice"),
        raw.get("soldPrice"),
        raw.get("unformattedPrice"),
        raw.get("price"),
        raw.get("formattedPrice"),
        min_dollars=1000,
    )


def _median_int(values: list[int]) -> int | None:
    vals = [int(v) for v in values if isinstance(v, int) and v > 0]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return int(round((vals[mid - 1] + vals[mid]) / 2))


def _apply_median_outlier_filter_fill_15(
    ranked_countable: list[dict],
    ranked_all: list[dict],
    *,
    sim_threshold: float,
    target: int,
    floor_ratio: float,
    ceiling_ratio: float,
    min_comps_for_median: int,
) -> tuple[list[dict], dict]:
    prices = []
    for c in ranked_countable:
        s = _get_similarity_value(c)
        if s is None or s < sim_threshold:
            continue
        p = _comp_best_price_from_search(c)
        if p is not None and p > 0:
            prices.append(p)

    dbg = {
        "outlier_filter_applied": False,
        "median_price": None,
        "floor_ratio": floor_ratio,
        "ceiling_ratio": ceiling_ratio,
        "floor_price": None,
        "ceiling_price": None,
        "removed_due_to_outlier": 0,
        "eligible_prices_for_median": len(prices),
    }

    # Not enough comps with price+sim for a meaningful median -> no outlier filter
    if len(prices) < min_comps_for_median:
        selected = ranked_countable[:target]
        if len(selected) < target:
            seen = set(_rank_key(x) for x in selected)
            for x in ranked_all:
                k = _rank_key(x)
                if k in seen:
                    continue
                selected.append(x)
                seen.add(k)
                if len(selected) >= target:
                    break
        return selected, dbg

    median_price = _median_int(prices)
    if median_price is None or median_price <= 0:
        selected = ranked_countable[:target]
        if len(selected) < target:
            seen = set(_rank_key(x) for x in selected)
            for x in ranked_all:
                k = _rank_key(x)
                if k in seen:
                    continue
                selected.append(x)
                seen.add(k)
                if len(selected) >= target:
                    break
        return selected, dbg

    floor_price = int(round(float(median_price) * float(floor_ratio)))
    ceiling_price = int(round(float(median_price) * float(ceiling_ratio)))

    dbg["outlier_filter_applied"] = True
    dbg["median_price"] = median_price
    dbg["floor_price"] = floor_price
    dbg["ceiling_price"] = ceiling_price

    selected: list[dict] = []
    removed = 0

    for c in ranked_countable:
        if len(selected) >= target:
            break

        s = _get_similarity_value(c)
        p = _comp_best_price_from_search(c)

        # Remove comps below 60% of median OR above 140% of median
        if (
            s is not None
            and s >= sim_threshold
            and p is not None
            and p > 0
            and (p < floor_price or p > ceiling_price)
        ):
            removed += 1
            continue

        selected.append(c)

    dbg["removed_due_to_outlier"] = removed

    if len(selected) < target:
        seen = set(_rank_key(x) for x in selected)
        for x in ranked_all:
            k = _rank_key(x)
            if k in seen:
                continue
            selected.append(x)
            seen.add(k)
            if len(selected) >= target:
                break

    return selected, dbg


# ============================================================
# Backfill helpers
# ============================================================
def _rank_key(x: dict) -> str:
    raw = x.get("raw") or {}
    zpid = raw.get("zpid") or raw.get("id") or x.get("zpid") or x.get("id")
    if zpid is not None:
        return f"zpid:{zpid}"
    addr = x.get("address") or raw.get("address") or ""
    return f"addr:{str(addr).strip().lower()}"


def _get_raw_num(raw: dict, keys: list[str]) -> float | None:
    for k in keys:
        v = raw.get(k)
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _is_fill_priority(subject_beds: float, subject_baths: float, subject_sqft: float, comp: dict) -> bool:
    miles = comp.get("miles_exact")
    if miles is None:
        miles = comp.get("miles")
    try:
        miles = float(miles)
    except Exception:
        return False
    if miles > FILL_PRIORITY_MAX_MILES:
        return False

    raw = comp.get("raw") or {}
    cb = _get_raw_num(raw, ["beds", "bedrooms"]) or _get_raw_num(comp, ["beds", "bedrooms", "comp_beds"])
    cba = _get_raw_num(raw, ["baths", "bathrooms"]) or _get_raw_num(comp, ["baths", "bathrooms", "comp_baths"])
    ca = _get_raw_num(raw, ["area", "livingArea", "sqft"]) or _get_raw_num(comp, ["area", "livingArea", "sqft", "comp_sqft"])
    if cb is None or cba is None or ca is None:
        return False

    if abs(cb - subject_beds) > FILL_PRIORITY_MAX_BED_DIFF:
        return False
    if abs(cba - subject_baths) > FILL_PRIORITY_MAX_BATH_DIFF:
        return False
    if abs(ca - subject_sqft) > FILL_PRIORITY_MAX_SQFT_DIFF:
        return False
    return True


# ============================================================
# Zillow URL month helpers
# ============================================================
def _make_12_month_url(zillow_url_6mo: str) -> str:
    u = zillow_url_6mo
    u2 = re.sub(r"(?i)(soldInLast|soldinlast|soldInLastMonths|soldinlastmonths)=\d+", r"\g<1>=12", u)
    if u2 != u:
        return u2
    u3 = re.sub(r"(?i)(doz%22%3A%22)6m(%22)", r"\g<1>12m\2", u)
    if u3 != u:
        return u3
    u4 = re.sub(r'(?i)("doz"\s*:\s*")6m(")', r'\g<1>12m\2', u)
    if u4 != u:
        return u4
    return u


def _generate_zillow_url_with_months(generate_zillow_url_fn, address, beds, baths, sqft, year, prop_type, months: int) -> str:
    try:
        sig = inspect.signature(generate_zillow_url_fn)
        params = sig.parameters
        for name in ("sold_months", "sold_window_months", "months", "months_back", "soldMonths"):
            if name in params:
                return generate_zillow_url_fn(address, beds, baths, sqft, year, prop_type, **{name: months})

        url_default = generate_zillow_url_fn(address, beds, baths, sqft, year, prop_type)
        return _make_12_month_url(url_default) if months == 12 else url_default
    except Exception:
        url_default = generate_zillow_url_fn(address, beds, baths, sqft, year, prop_type)
        return _make_12_month_url(url_default) if months == 12 else url_default


def _decode_search_query_state(zillow_url: str) -> dict | None:
    try:
        parsed = urllib.parse.urlparse(zillow_url)
        q = urllib.parse.parse_qs(parsed.query)
        s = q.get("searchQueryState", [None])[0]
        if not s:
            return None
        decoded = urllib.parse.unquote(s)
        return json.loads(decoded)
    except Exception:
        return None


def _encode_search_query_state(base_url: str, state: dict) -> str:
    parsed = urllib.parse.urlparse(base_url)
    q = urllib.parse.parse_qs(parsed.query)
    s = json.dumps(state, separators=(",", ":"))
    q["searchQueryState"] = [urllib.parse.quote(s)]

    # FIX: skip any list values to avoid TypeError when joining
    pairs: list[str] = []
    for k, v in q.items():
        if not v:
            continue
        first_val = v[0]
        if isinstance(first_val, list):
            # Drop list-valued query params in relaxed mode
            continue
        pairs.append(f"{k}={first_val}")

    new_query = "&".join(pairs)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def _build_relaxed_12mo_url(zillow_url_12: str, subject_year: int, subject_sqft: int) -> str:
    state = _decode_search_query_state(zillow_url_12)
    if not isinstance(state, dict):
        return zillow_url_12
    fs = state.get("filterState")
    if not isinstance(fs, dict):
        return zillow_url_12

    doz = fs.get("doz")
    if isinstance(doz, dict) and "value" in doz:
        fs["doz"]["value"] = "12m"

    fs["built"] = {"min": int(subject_year - 25), "max": int(subject_year + 15)}

    if isinstance(fs.get("beds"), dict):
        try:
            bmin = fs["beds"].get("min")
            bmax = fs["beds"].get("max")
            if bmin is not None:
                fs["beds"]["min"] = max(0, int(bmin) - 1)
            if bmax is not None:
                fs["beds"]["max"] = int(bmax) + 1
        except Exception:
            pass

    if isinstance(fs.get("baths"), dict):
        try:
            bn = fs["baths"].get("min")
            bx = fs["baths"].get("max")
            if bn is not None:
                fs["baths"]["min"] = max(0, float(bn) - 1.0)
            if bx is not None:
                fs["baths"]["max"] = float(bx) + 1.0
        except Exception:
            pass

    fs["sqft"] = {"min": max(0, int(subject_sqft * 0.75)), "max": int(subject_sqft * 1.25)}

    state["filterState"] = fs
    return _encode_search_query_state(zillow_url_12, state)


# ============================================================
# Photo URL extraction
# ============================================================
def _norm_url(u: str) -> str:
    return u.strip() if isinstance(u, str) else ""


def _try_int(x):
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _rewrite_url_width(url: str, target_w: int) -> str:
    if not isinstance(url, str) or not url.startswith("http"):
        return url
    target_w = max(MIN_IMAGE_WIDTH, int(target_w))
    u2 = re.sub(r"([?&](?:width|w)=)\d+", rf"\g<1>{target_w}", url, flags=re.IGNORECASE)
    return u2 if u2 != url else url


def _collect_url_candidates(obj):
    if obj is None:
        return
    if isinstance(obj, str):
        if obj.startswith("http"):
            yield (obj, None)
        return
    if isinstance(obj, dict):
        for k in ("url", "imageUrl", "imageURL", "src", "href"):
            u = obj.get(k)
            if isinstance(u, str) and u.startswith("http"):
                w = _try_int(obj.get("width") or obj.get("w"))
                yield (u, w)
        for k in ("mixedSources", "sources", "srcSet", "srcset", "imageSources"):
            v = obj.get(k)
            if isinstance(v, list):
                for item in v:
                    yield from _collect_url_candidates(item)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _collect_url_candidates(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _collect_url_candidates(item)
        return


def _choose_best_small_url(candidates, target_w: int) -> str | None:
    cands = []
    for u, w in candidates:
        u = _norm_url(u)
        if not u.startswith("http"):
            continue
        cands.append((u, w))
    if not cands:
        return None

    with_w = [(u, w) for (u, w) in cands if isinstance(w, int) and w > 0]
    if with_w:
        ge = sorted([(u, w) for (u, w) in with_w if w >= target_w], key=lambda x: x[1])
        if ge:
            return ge[0][0]
        return sorted(with_w, key=lambda x: x[1], reverse=True)[0][0]

    return _rewrite_url_width(cands[0][0], target_w)


def extract_photo_urls(detail_item: dict) -> list[str]:
    urls, seen = [], set()
    photo_lists = []
    if isinstance(detail_item.get("photos"), list):
        photo_lists.append(detail_item.get("photos"))
    if isinstance(detail_item.get("responsivePhotos"), list):
        photo_lists.append(detail_item.get("responsivePhotos"))
    for plist in photo_lists:
        for p in plist:
            cand = list(_collect_url_candidates(p))
            best = _choose_best_small_url(cand, TARGET_IMAGE_WIDTH)
            # STRICT ZILLOW WHITELIST: only keep real property photos
            if not best or not isinstance(best, str) or not best.startswith("http"):
                continue
            bl = best.lower()
            if "photos.zillowstatic.com/fp/" not in bl:
                continue
            if best not in seen:
                seen.add(best)
                urls.append(best)
    return urls


def find_photo_urls_anywhere(detail_item: dict) -> list[str]:
    urls, seen = [], set()

    def ok(u: str) -> bool:
        if not isinstance(u, str) or not u.startswith("http"):
            return False
        ul = u.lower()
        # STRICT ZILLOW WHITELIST: only real listing photos
        if "photos.zillowstatic.com/fp/" not in ul:
            return False
        return True

    for u, _w in _collect_url_candidates(detail_item):
        if ok(u) and u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def build_detail_index(detail_items: list[dict]):
    by_zpid, by_url = {}, {}
    for item in detail_items:
        if not isinstance(item, dict):
            continue
        zpid = item.get("zpid") or item.get("id") or item.get("zillowId") or item.get("zillow_id")
        if zpid is not None:
            by_zpid[str(zpid)] = item
        for k in ("url", "detailUrl", "detail_url", "listingUrl", "zillowUrl"):
            u = item.get(k)
            if isinstance(u, str) and u.startswith("http"):
                by_url[_norm_url(u)] = item
    return by_zpid, by_url


# ============================================================
# SUBJECT thumbnail helper (match comps thumbnail behavior)
# ============================================================
def _norm_addr_key(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace(",", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_house_number(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = re.match(r"^\s*(\d+)", s.strip())
    return m.group(1) if m else ""


def _find_subject_candidate_in_comps(subject_address: str, comps_all: list[dict]) -> dict | None:
    """
    Finds a likely subject property hit from the search-scrape results.
    This avoids adding a new Apify run by reusing comps search results.
    """
    subj_key = _norm_addr_key(subject_address)
    subj_num = _extract_house_number(subject_address)

    best = None
    best_score = -1

    for c in comps_all or []:
        if not isinstance(c, dict):
            continue
        raw = c.get("raw") or {}

        addr = c.get("address") or raw.get("address") or ""
        addr_key = _norm_addr_key(str(addr))

        # Must at least share the same leading house number if we have one
        if subj_num and _extract_house_number(str(addr)) != subj_num:
            continue

        score = 0
        if addr_key == subj_key and addr_key:
            score += 100  # exact match
        else:
            # Partial match heuristic
            if subj_key and addr_key and subj_key in addr_key:
                score += 40
            if subj_num:
                score += 10

        detail_url = raw.get("detailUrl") or raw.get("detail_url") or raw.get("url") or c.get("url")
        if not (isinstance(detail_url, str) and detail_url.startswith("http")):
            continue

        if score > best_score:
            best_score = score
            best = c

    return best


def _thumbnail_url_from_detail_item(detail_item: dict) -> str | None:
    if not isinstance(detail_item, dict):
        return None
    urls = extract_photo_urls(detail_item)
    if not urls:
        urls = find_photo_urls_anywhere(detail_item)
    urls = [u for u in urls if isinstance(u, str) and "photos.zillowstatic.com/fp/" in u.lower()]
    return urls[0] if urls else None


# ============================================================
# Option B subject lookup: search-scrape the SUBJECT address to get its zpid/detailUrl
# ============================================================
def _build_subject_lookup_url(address: str) -> str:
    # Zillow supports /homes/<address>_rb/ patterns; this is the simplest robust build.
    a = (address or "").strip()
    a = re.sub(r"\s+", " ", a)
    a = a.replace("#", " ")
    a = a.replace("/", " ")
    a = re.sub(r"\s+", " ", a).strip()
    slug = a.replace(",", "")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"https://www.zillow.com/homes/{slug}_rb/"


def _pick_best_subject_from_search_items(subject_address: str, items: list[dict]) -> dict | None:
    subj_key = _norm_addr_key(subject_address)
    subj_num = _extract_house_number(subject_address)

    best = None
    best_score = -1

    for it in items or []:
        if not isinstance(it, dict):
            continue

        raw_addr = it.get("address") or ""
        try:
            hdp = it.get("hdpData") or {}
            home = hdp.get("homeInfo") or {}
            if not raw_addr:
                raw_addr = home.get("streetAddress") or home.get("address") or ""
        except Exception:
            pass

        addr_key = _norm_addr_key(str(raw_addr))
        if subj_num and _extract_house_number(str(raw_addr)) != subj_num:
            continue

        score = 0
        if addr_key and addr_key == subj_key:
            score += 100
        else:
            if subj_key and addr_key and subj_key in addr_key:
                score += 40
            if subj_num:
                score += 10

        zpid = it.get("zpid") or it.get("id") or it.get("zillowId") or it.get("zillow_id")
        if zpid is not None:
            score += 5

        detail_url = it.get("detailUrl") or it.get("detail_url") or it.get("url") or it.get("listingUrl") or it.get("zillowUrl")
        if not (isinstance(detail_url, str) and detail_url.startswith("http")):
            # still allow if zpid exists (we can build url)
            if zpid is None:
                continue

        if score > best_score:
            best_score = score
            best = it

    return best


# ============================================================
# Downloader
# ============================================================
def _get_download_pool() -> ThreadPoolExecutor:
    global _DOWNLOAD_POOL
    if _DOWNLOAD_POOL is None:
        _DOWNLOAD_POOL = ThreadPoolExecutor(max_workers=GLOBAL_DOWNLOAD_WORKERS)
    return _DOWNLOAD_POOL


def _url_variants(url: str) -> list[str]:
    out = []
    if not isinstance(url, str) or not url.startswith("http"):
        return out
    out.append(url)

    if re.search(r"\.webp(\?|$)", url, flags=re.IGNORECASE):
        out.append(re.sub(r"\.webp(\?|$)", r".jpg\1", url, flags=re.IGNORECASE))
        out.append(re.sub(r"(?i)(format=)webp", r"\1jpg", url))
        out.append(re.sub(r"(?i)(fm=)webp", r"\1jpg", url))

    if "?" not in url:
        out.append(url + "?format=jpg")
    else:
        if not re.search(r"(?i)format=", url):
            out.append(url + "&format=jpg")

    seen = set()
    final = []
    for u in out:
        if u and u.startswith("http") and u not in seen:
            seen.add(u)
            final.append(u)
    return final


def _fetch_bytes_with_retry(url: str, extra_headers: dict | None = None) -> tuple[bytes | None, int | None, str | None]:
    s = get_thread_session()
    for attempt in range(IMG_MAX_RETRIES):
        try:
            r = s.get(
                url,
                headers=extra_headers,
                timeout=(IMG_CONNECT_TIMEOUT_SEC, IMG_READ_TIMEOUT_SEC),
                allow_redirects=True,
            )
            status = r.status_code
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower() or None

            if DEBUG_DOWNLOAD:
                with _dl_lock:
                    _dl_status[status] += 1
                    if ctype:
                        _dl_ctypes[ctype] += 1

            if status in (429, 403, 503, 520, 521, 522):
                time.sleep(IMG_BACKOFF_BASE_SEC * (2**attempt) + random.uniform(0, 0.25))
                continue

            r.raise_for_status()
            if not r.content:
                time.sleep(IMG_BACKOFF_BASE_SEC * (2**attempt) + random.uniform(0, 0.25))
                continue

            return (r.content, status, ctype)
        except Exception:
            time.sleep(IMG_BACKOFF_BASE_SEC * (2**attempt) + random.uniform(0, 0.25))
    return (None, None, None)


def _fetch_and_decode(url: str) -> Image.Image | None:
    if not isinstance(url, str) or not url.startswith("http"):
        return None

    acquired = _inflight_sem.acquire(timeout=30)
    if not acquired:
        return None
    try:
        headers = {"Referer": "https://www.zillow.com/"}
        for u in _url_variants(url):
            b, _st, _ct = _fetch_bytes_with_retry(u, extra_headers=headers)
            if not b:
                continue
            try:
                img = Image.open(BytesIO(b))
                try:
                    img.draft("RGB", (256, 256))
                except Exception:
                    pass
                return img.convert("RGB")
            except Exception:
                continue
        return None
    finally:
        _inflight_sem.release()


def download_images_to_pil(urls: list[str], max_images: int) -> tuple[list[Image.Image], int]:
    urls = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    urls = urls[:max_images]
    if not urls:
        return ([], 0)

    pool = _get_download_pool()
    futures = [pool.submit(_fetch_and_decode, u) for u in urls]
    images: list[Image.Image] = []
    for f in as_completed(futures):
        img = f.result()
        if img is not None:
            images.append(img)
    return (images, len(images))


# ============================================================
# condition_tester dynamic loader
# ============================================================
def _callable_no_required_args(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            return False
    return True


def _callable_first_arg_images(fn) -> bool:
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params:
            return False
        first = params[0]
        if first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
            return True
        if first.kind == first.VAR_POSITIONAL:
            return True
        return False
    except Exception:
        return False


def _load_condition_functions():
    import importlib

    mod = importlib.import_module("condition_tester")

    get_candidates = ["get_models", "get_model"]
    score_candidates = ["score_pil_images", "score_images_pil", "score_images"]
    get_fallback = ["load_models", "init_models"]
    score_fallback = ["score", "predict", "infer"]

    get_fn = None
    for name in get_candidates:
        fn = getattr(mod, name, None)
        if callable(fn) and _callable_no_required_args(fn):
            get_fn = fn
            break
    if get_fn is None:
        for name in get_fallback:
            fn = getattr(mod, name, None)
            if callable(fn) and _callable_no_required_args(fn):
                get_fn = fn
                break

    score_fn = None
    for name in score_candidates:
        fn = getattr(mod, name, None)
        if callable(fn) and _callable_first_arg_images(fn):
            score_fn = fn
            break
    if score_fn is None:
        for name in score_fallback:
            fn = getattr(mod, name, None)
            if callable(fn) and _callable_first_arg_images(fn):
                score_fn = fn
                break

    if get_fn is None or score_fn is None:
        avail = [n for n in dir(mod) if not n.startswith("_")]
        raise ImportError(
            "condition_tester.py missing required functions.\n"
            f"Need model-loader callable with NO required args (tried: {get_candidates + get_fallback}).\n"
            f"Need scorer callable like score_fn(pil_images) (tried: {score_candidates + score_fallback}).\n"
            f"Available exports: {avail}"
        )

    return get_fn, score_fn


# ============================================================
# Printing
# ============================================================
def _print_ranked_comps_table(ranked: list[dict], title: str):
    print(f"\n{title}\n")
    if not ranked:
        print("(none)\n")
        return

    for i, c in enumerate(ranked, 1):
        raw = c.get("raw") or {}
        addr = c.get("address") or raw.get("address") or "UNKNOWN"

        miles = c.get("miles")
        if miles is None:
            miles = c.get("distance_miles")

        weight = c.get("weight")
        if weight is None:
            weight = c.get("similarity") or c.get("similarity_score") or c.get("score")

        price_int = _best_price_int(
            c.get("sold_price"),
            c.get("price"),
            c.get("unformattedPrice"),
            raw.get("soldPrice"),
            raw.get("unformattedPrice"),
            raw.get("price"),
            raw.get("formattedPrice"),
            c.get("formattedPrice"),
            min_dollars=1000,
        )

        mtxt = f"{float(miles):.2f} mi" if miles is not None else "n/a"
        wtxt = f"{float(weight):.3f}" if weight is not None else "n/a"
        ptxt = f"${price_int:,}" if (price_int is not None and price_int > 0) else ""
        print(f"{i:>2}. {addr} | {mtxt} | weight={wtxt} {ptxt}")
    print("")


# ============================================================
# ARV normalization + compute_arv hook
# ============================================================
def _extract_street_name(address: str) -> str:
    if not isinstance(address, str):
        return ""
    first = address.split(",")[0].strip().lower()
    first = re.sub(r"^\d+\s+", "", first)
    first = re.sub(r"\s+#.*$", "", first)
    first = re.sub(r"\s+apt.*$", "", first)
    first = re.sub(r"\s+unit.*$", "", first)
    return first.strip()


def _normalize_comp_for_arv(comp: dict, subject_street: str) -> dict:
    c = dict(comp)
    raw = c.get("raw") or {}

    c["address"] = c.get("address") or raw.get("address")

    c["sold_price"] = _best_price_int(
        c.get("sold_price"),
        c.get("price"),
        c.get("unformattedPrice"),
        raw.get("soldPrice"),
        raw.get("unformattedPrice"),
        raw.get("price"),
        raw.get("formattedPrice"),
        c.get("formattedPrice"),
        min_dollars=1000,
    )

    c["comp_sqft"] = raw.get("area") or raw.get("livingArea") or raw.get("sqft") or c.get("area") or c.get("sqft")
    c["comp_beds"] = raw.get("beds") or raw.get("bedrooms") or c.get("beds")
    c["comp_baths"] = raw.get("baths") or raw.get("bathrooms") or c.get("baths")

    c["sold_date"] = c.get("sold_date") or raw.get("soldDate") or raw.get("dateSold") or raw.get("sold_date")

    if c.get("distance_miles") is None:
        c["distance_miles"] = c.get("miles") if c.get("miles") is not None else c.get("miles_exact")

    if c.get("condition_score") is None:
        if c.get("final_score") is not None:
            c["condition_score"] = c.get("final_score")

    if "skip_reason" not in c:
        c["skip_reason"] = None

    addr = str(c.get("address") or "")
    c["same_street"] = (_extract_street_name(addr) != "" and _extract_street_name(addr) == subject_street)

    return c


# ============================================================
# ARV selected comps enrichment (UI payload)
# ============================================================
def _build_zillow_url_from_zpid(zpid: Any) -> str | None:
    if zpid is None:
        return None
    try:
        z = str(zpid).strip()
        if not z:
            return None
        if not re.fullmatch(r"\d+", z):
            return None
        return f"https://www.zillow.com/homedetails/{z}_zpid/"
    except Exception:
        return None


def _b64_thumbnail_data_uri_from_pil(img: Image.Image, target_w: int = 240, jpeg_quality: int = 65) -> str | None:
    """
    Compress thumbnail payload for API:
      - downscale to target_w (default 240px wide)
      - JPEG quality default 65
      - optimize + progressive
    """
    try:
        if img is None:
            return None

        # Ensure RGB + no alpha surprises
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if w <= 0 or h <= 0:
            return None

        target_w = max(160, int(target_w))

        # High-quality downscale (Pillow version-safe)
        try:
            resample = Image.Resampling.LANCZOS  # Pillow >= 9
        except Exception:
            resample = Image.LANCZOS  # older Pillow

        img2 = img.copy()
        if w > target_w:
            scale = float(target_w) / float(w)
            target_h = max(1, int(round(h * scale)))
            img2 = img2.resize((target_w, target_h), resample=resample)

        bio = BytesIO()
        img2.save(
            bio,
            format="JPEG",
            quality=int(jpeg_quality),
            optimize=True,
            progressive=True,
        )
        b64 = base64.b64encode(bio.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _enrich_arv_selected_comps(selected_comps: list[dict], by_zpid: dict, by_url: dict) -> list[dict]:
    enriched: list[dict] = []
    for c in selected_comps or []:
        if not isinstance(c, dict):
            continue

        zpid = c.get("zpid") or c.get("id")
        detail_url = c.get("detail_url") or c.get("detailUrl") or c.get("url")
        detail_url_norm = _norm_url(detail_url) if isinstance(detail_url, str) else ""

        zpid_str = str(zpid).strip() if zpid is not None else ""
        detail_item = None
        if zpid_str and zpid_str in by_zpid:
            detail_item = by_zpid.get(zpid_str)
        elif detail_url_norm and detail_url_norm in by_url:
            detail_item = by_url.get(detail_url_norm)

        # Determine zillow_url
        zillow_url = None
        if isinstance(detail_url_norm, str) and detail_url_norm.startswith("http"):
            zillow_url = detail_url_norm
        if not zillow_url:
            zillow_url = _build_zillow_url_from_zpid(zpid)

        # CHANGE: return a short thumbnail URL instead of base64 data uri
        thumbnail_url = None
        if isinstance(detail_item, dict):
            urls = extract_photo_urls(detail_item)
            if not urls:
                urls = find_photo_urls_anywhere(detail_item)
            urls = [u for u in urls if isinstance(u, str) and "photos.zillowstatic.com/fp/" in u.lower()]
            if urls:
                thumbnail_url = urls[0]

        # FIX: include sold_price on enriched comps (pull from selected comp first; fallback to detail scrape)
        raw = c.get("raw") or {}
        sold_price = _best_price_int(
            c.get("sold_price"),
            c.get("soldPrice"),
            c.get("price"),
            c.get("unformattedPrice"),
            c.get("formattedPrice"),
            raw.get("soldPrice"),
            raw.get("unformattedPrice"),
            raw.get("price"),
            raw.get("formattedPrice"),
            min_dollars=1000,
        )
        if sold_price is None and isinstance(detail_item, dict):
            sold_price = _extract_price_from_detail(detail_item)

        # FIX: include address on enriched comps (Shopify/UI)
        comp_address = c.get("address")
        if not comp_address:
            try:
                comp_address = (c.get("raw") or {}).get("address")
            except Exception:
                comp_address = None
        if not comp_address and isinstance(detail_item, dict):
            comp_address = detail_item.get("address")
            if not comp_address:
                try:
                    comp_address = ((detail_item.get("hdpData") or {}).get("homeInfo") or {}).get("streetAddress")
                except Exception:
                    comp_address = None

        distance_miles = c.get("distance_miles")
        beds = c.get("beds") if c.get("beds") is not None else c.get("comp_beds")
        baths = c.get("baths") if c.get("baths") is not None else c.get("comp_baths")
        sqft = c.get("comp_sqft") if c.get("comp_sqft") is not None else c.get("sqft")

        enriched.append(
            {
                "zpid": zpid,
                "zillow_url": zillow_url,
                "thumbnail_data_uri": None,
                "thumbnail_url": thumbnail_url,
                "sold_price": sold_price,
                "address": comp_address,
                "distance_miles": distance_miles,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
            }
        )
    return enriched


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(address: str, beds: float, baths: float, sqft: int, year: int, property_type: str, subject: dict) -> dict:
    from zillow_url_generator import generate_zillow_url_from_subject
    from comp_ranker import rank_comps

    try:
        from arv_formula import compute_arv
    except Exception:
        compute_arv = None  # type: ignore[assignment]

    apify_session = create_apify_session()

    # ===============================================
    # STEP 1 — ZILLOW URLS
    # ===============================================
    print("\nSTEP 1: Generate Zillow URL (6 months)\n")
    zillow_url_6 = generate_zillow_url_from_subject(subject, 6)
    print(zillow_url_6)

    print("\nSTEP 2: Apify search scrape (6 months)\n")
    comps_6 = _run_search_scrape(apify_session, zillow_url_6) or []
    print(f"6-month comps returned: {len(comps_6)}")

    comps_all = list(comps_6)

    ran_12mo = False
    zillow_url_12 = None

    if len(comps_6) < TRIGGER_12MO_IF_6MO_LESS_THAN:
        ran_12mo = True

        print(f"\nSTEP 3: 12-month top-up triggered (6mo comps={len(comps_6)} < {TRIGGER_12MO_IF_6MO_LESS_THAN})\n")

        zillow_url_12 = generate_zillow_url_from_subject(subject, 12)
        print(zillow_url_12)

        comps_12 = _run_search_scrape(apify_session, zillow_url_12) or []
        print(f"12-month comps returned: {len(comps_12)}")

        comps_all.extend(comps_12)

    comps_all = _dedupe_comps(comps_all)
    print(f"Total comps after 6mo(+12mo if run) dedupe: {len(comps_all)}")

    # RELAXED SEARCH PASS
    if ran_12mo and len(comps_all) < MIN_COMPS_AFTER_12MO_BEFORE_RELAX and zillow_url_12:
        print(f"\nSTEP 3B: Relaxed 12-month search triggered (comps after 12mo={len(comps_all)} < {MIN_COMPS_AFTER_12MO_BEFORE_RELAX})\n")

        relaxed_url = _build_relaxed_12mo_url(zillow_url_12, int(year), int(sqft))
        print(relaxed_url)

        comps_relaxed = _run_search_scrape(apify_session, relaxed_url) or []
        print(f"Relaxed 12-month comps returned: {len(comps_relaxed)}")

        comps_all.extend(comps_relaxed)
        comps_all = _dedupe_comps(comps_all)

        print(f"Total comps after relaxed dedupe: {len(comps_all)}")

    if not comps_all:
        print("No comps returned after 6-month + (12-month if triggered) + (relaxed if triggered). Pipeline stopping.\n")
        return {
            "status": "fail",
            "message": "NO_COMPS_AFTER_ALL_SEARCHES",
            "ranked": [],
            "arv": None
        }

    print(f"\nTotal comps collected (after dedupe/top-up): {len(comps_all)}\n")

    # -----------------------------------------------
    # SUBJECT THUMBNAIL: choose a reliable subject detail URL
    # Priority:
    #   1) subject["subject_zillow_url"] / ["subjectZillowUrl"]
    #   2) subject["subject_zpid"] / ["subjectZpid"] -> build url
    #   3) subject lookup via SEARCH_ACTOR on /homes/<address>_rb/ (Option B)
    #   4) fallback heuristic from comps search results (existing behavior)
    # -----------------------------------------------
    subject_detail_url = None
    subject_zpid = None
    try:
        subject_detail_url = (
            subject.get("subject_zillow_url")
            or subject.get("subjectZillowUrl")
            or subject.get("zillow_url")
            or subject.get("zillowUrl")
        )
        if isinstance(subject_detail_url, str):
            subject_detail_url = _norm_url(subject_detail_url)

        subject_zpid = (
            subject.get("subject_zpid")
            or subject.get("subjectZpid")
            or subject.get("zpid")
        )
        if subject_zpid is not None:
            subject_zpid = str(subject_zpid).strip()
    except Exception:
        subject_detail_url = None
        subject_zpid = None

    if not subject_zpid and not (isinstance(subject_detail_url, str) and subject_detail_url.startswith("http")):
        try:
            subj_lookup_url = _build_subject_lookup_url(address)
            subj_items = _run_search_scrape(apify_session, subj_lookup_url) or []
            picked = _pick_best_subject_from_search_items(address, subj_items)
            if isinstance(picked, dict):
                subject_zpid = picked.get("zpid") or picked.get("id") or picked.get("zillowId") or picked.get("zillow_id")
                if subject_zpid is not None:
                    subject_zpid = str(subject_zpid).strip()

                subject_detail_url = (
                    picked.get("detailUrl")
                    or picked.get("detail_url")
                    or picked.get("url")
                    or picked.get("listingUrl")
                    or picked.get("zillowUrl")
                )
                if isinstance(subject_detail_url, str):
                    subject_detail_url = _norm_url(subject_detail_url)
        except Exception:
            pass

    if not (isinstance(subject_detail_url, str) and subject_detail_url.startswith("http")):
        built = _build_zillow_url_from_zpid(subject_zpid) if subject_zpid else None
        if isinstance(built, str) and built.startswith("http"):
            subject_detail_url = built

    if not (isinstance(subject_detail_url, str) and subject_detail_url.startswith("http")):
        try:
            subj_candidate = _find_subject_candidate_in_comps(address, comps_all)
            if isinstance(subj_candidate, dict):
                raw_sc = subj_candidate.get("raw") or {}
                subject_detail_url = raw_sc.get("detailUrl") or raw_sc.get("detail_url") or raw_sc.get("url") or subj_candidate.get("url")
                if isinstance(subject_detail_url, str):
                    subject_detail_url = _norm_url(subject_detail_url)
        except Exception:
            subject_detail_url = None

    # ===============================================
    # STEP 4 — SIMILARITY RANKING
    # ===============================================
    print("\nSTEP 4: Similarity ranking\n")

    subject_for_ranker = {
        "address": address,
        "beds": beds,
        "baths": baths,
        "area": sqft,
        "year": year,
    }

    ranked_all = rank_comps(subject_for_ranker, comps_all)
    ranked_countable = _apply_similarity_threshold(ranked_all, SIMILARITY_THRESHOLD)

    if len(ranked_countable) < MAX_COMPS_TO_SCORE:
        seen = set(_rank_key(x) for x in ranked_countable)
        fill_candidates = []

        for x in ranked_all:
            k = _rank_key(x)
            if k in seen:
                continue

            if _is_fill_priority(float(beds), float(baths), float(sqft), x) and _counts_toward_target(x, SIMILARITY_THRESHOLD):
                fill_candidates.append(x)

        fill_candidates.sort(
            key=lambda x: (
                float(x.get("miles_exact") or x.get("miles") or 9e9),
                -float(x.get("weight") or 0.0),
            )
        )

        for x in fill_candidates:
            ranked_countable.append(x)
            seen.add(_rank_key(x))

            if len(ranked_countable) >= MAX_COMPS_TO_SCORE:
                break

    countable_sim_ge = sum(
        1 for c in ranked_countable
        if (_get_similarity_value(c) or 0) >= SIMILARITY_THRESHOLD
    )

    ranked, outlier_dbg = _apply_median_outlier_filter_fill_15(
        ranked_countable=ranked_countable,
        ranked_all=ranked_all,
        sim_threshold=SIMILARITY_THRESHOLD,
        target=MAX_COMPS_TO_SCORE,
        floor_ratio=OUTLIER_MEDIAN_FLOOR_RATIO,
        ceiling_ratio=OUTLIER_MEDIAN_CEIL_RATIO,
        min_comps_for_median=OUTLIER_MIN_COMPS_FOR_MEDIAN,
    )

    if outlier_dbg.get("outlier_filter_applied"):
        print(
            f"Outlier filter applied: eligible(sim>=0.35 with price)={outlier_dbg.get('eligible_prices_for_median')}, "
            f"median=${int(outlier_dbg.get('median_price') or 0):,}, "
            f"floor({OUTLIER_MEDIAN_FLOOR_RATIO:.2f}x)=${int(outlier_dbg.get('floor_price') or 0):,}, "
            f"ceiling({OUTLIER_MEDIAN_CEIL_RATIO:.2f}x)=${int(outlier_dbg.get('ceiling_price') or 0):,}, "
            f"removed={outlier_dbg.get('removed_due_to_outlier')}"
        )
    else:
        print(
            f"Outlier filter skipped: eligible(sim>=0.35 with price)={outlier_dbg.get('eligible_prices_for_median')} "
            f"(need >= {OUTLIER_MIN_COMPS_FOR_MEDIAN})"
        )
    if not ranked:
        print(f"Final comps kept: 0 (threshold={SIMILARITY_THRESHOLD}, target={MAX_COMPS_TO_SCORE})")
        print("No comps available to proceed. Pipeline stopping before detail/download.\n")
        return {"status": "fail", "message": "NO_COMPS_AFTER_RANKING", "ranked": [], "arv": None}

    print(f"Comps with sim>=0.35 in countable pool: {countable_sim_ge}")
    print(f"Total comps selected for scoring/print: {len(ranked)}/{MAX_COMPS_TO_SCORE}")

    print("\nSTEP 5: Detail scrape for all selected comps (single call)\n")
    detail_urls = []
    for comp in ranked:
        raw = comp.get("raw") or {}
        detail_url = raw.get("detailUrl") or raw.get("detail_url") or raw.get("url")
        if isinstance(detail_url, str) and detail_url.startswith("http"):
            detail_urls.append({"url": detail_url})

    # Include subject detail url (if found) so we can extract subject thumbnail using same detail actor
    if isinstance(subject_detail_url, str) and subject_detail_url.startswith("http"):
        detail_urls.append({"url": subject_detail_url})

    # Dedupe startUrls
    try:
        seen_urls = set()
        deduped = []
        for x in detail_urls:
            u = x.get("url")
            if not isinstance(u, str):
                continue
            un = _norm_url(u)
            if not un or un in seen_urls:
                continue
            seen_urls.add(un)
            deduped.append({"url": un})
        detail_urls = deduped
    except Exception:
        pass

    detail_items = []
    if detail_urls:
        detail_payload = {"startUrls": detail_urls}
        detail_items = apify_run_sync_get_dataset_items(apify_session, DETAIL_ACTOR_ID, detail_payload, timeout_sec=APIFY_TIMEOUT_SEC)
    if not isinstance(detail_items, list):
        detail_items = []

    by_zpid, by_url = build_detail_index(detail_items)

    # Extract subject thumbnail_url (same logic as comps: first fp photo URL)
    subject_thumbnail_url = None
    try:
        di = None

        # First try URL key (normalized)
        if isinstance(subject_detail_url, str) and subject_detail_url:
            key = _norm_url(subject_detail_url)
            di = by_url.get(key)

            # Fallback: try adding/removing trailing slash
            if di is None:
                if key.endswith("/"):
                    di = by_url.get(key[:-1])
                else:
                    di = by_url.get(key + "/")

        # Fallback: try zpid index if we have one
        if di is None and subject_zpid:
            di = by_zpid.get(str(subject_zpid).strip())

        if isinstance(di, dict):
            subject_thumbnail_url = _thumbnail_url_from_detail_item(di)
    except Exception:
        subject_thumbnail_url = None

    for comp in ranked:
        raw = comp.get("raw") or {}
        zpid = raw.get("zpid") or raw.get("id") or comp.get("zpid") or comp.get("id")
        zpid_str = str(zpid) if zpid is not None else ""
        detail_url = raw.get("detailUrl") or raw.get("detail_url") or raw.get("url")
        detail_url = _norm_url(detail_url) if isinstance(detail_url, str) else ""

        detail_item = None
        if zpid_str and zpid_str in by_zpid:
            detail_item = by_zpid.get(zpid_str)
        elif detail_url and detail_url in by_url:
            detail_item = by_url.get(detail_url)

        if isinstance(detail_item, dict):
            p = _extract_price_from_detail(detail_item)
            if p is not None:
                comp["sold_price"] = p
                comp["price"] = p
                try:
                    raw["soldPrice"] = p
                    raw["unformattedPrice"] = p
                except Exception:
                    pass

    _print_ranked_comps_table(ranked, "TOP COMPS CHOSEN (POST THRESHOLD + OUTLIER FILTER + BACKFILL)")

    print("\nLoading models once (warm-up).")
    get_models, score_images = _load_condition_functions()
    get_models()
    print("Models loaded.\n")

    print("STEP 6: Download up to 70 images + condition score (parallel)\n")
    print(f"Downloader: workers={GLOBAL_DOWNLOAD_WORKERS}, inflight={GLOBAL_MAX_INFLIGHT}, retries={IMG_MAX_RETRIES}\n")

    def process_one_comp(comp: dict) -> dict:
        raw = comp.get("raw") or {}
        zpid = raw.get("zpid") or raw.get("id") or comp.get("zpid") or comp.get("id")
        zpid_str = str(zpid) if zpid is not None else ""
        detail_url = raw.get("detailUrl") or raw.get("detail_url") or raw.get("url")
        detail_url = _norm_url(detail_url) if isinstance(detail_url, str) else ""

        detail_item = None
        if zpid_str and zpid_str in by_zpid:
            detail_item = by_zpid.get(zpid_str)
        elif detail_url and detail_url in by_url:
            detail_item = by_url.get(detail_url)

        urls = []
        if isinstance(detail_item, dict):
            urls = extract_photo_urls(detail_item)
            if len(urls) < 5:
                urls = find_photo_urls_anywhere(detail_item)

        # FINAL SAFETY: enforce Zillow fp whitelist on whatever we got
        urls = [u for u in urls if isinstance(u, str) and "photos.zillowstatic.com/fp/" in u.lower()]

        urls = urls[:MAX_IMAGES_TO_DOWNLOAD]
        pil_images, got = download_images_to_pil(urls, MAX_IMAGES_TO_DOWNLOAD)

        # Always log what we found/downloaded (helps diagnose “phantom photo” cases)
        comp["downloaded_images"] = got
        comp["photos_found_urls"] = len(urls)
        comp["photos_sample_urls"] = urls[:3]

        # HARD GUARD: no (or too few) images => do not score condition
        if got < MIN_REQUIRED_IMAGES:
            comp["skip_reason"] = "INSUFFICIENT_INTERIOR_IMAGES"
            return comp

        try:
            result = score_images(pil_images)
            if isinstance(result, dict):
                comp.update(result)
        except Exception as e:
            comp["skip_reason"] = "CONDITION_SCORING_ERROR"
            comp["condition_error"] = str(e)

        return comp

    scored: list[dict] = []
    with ThreadPoolExecutor(max_workers=COMP_WORKERS) as ex:
        futures = [ex.submit(process_one_comp, dict(c)) for c in ranked]
        for f in as_completed(futures):
            scored.append(f.result())

    key_to_comp = {_rank_key(c): c for c in scored}
    scored_ordered = []
    for c in ranked:
        scored_ordered.append(key_to_comp.get(_rank_key(c), c))

    print("\nSTEP 7: Condition scoring results\n")
    for i, c in enumerate(scored_ordered, 1):
        addr = c.get("address") or (c.get("raw") or {}).get("address") or "UNKNOWN"
        skip = c.get("skip_reason")
        cs = c.get("condition_score") if c.get("condition_score") is not None else c.get("final_score")
        lbl = c.get("condition_label") or c.get("label")
        if skip:
            print(f"{i:>2}. {addr} | SKIP={skip} | downloaded_images={c.get('downloaded_images')} | photos_found_urls={c.get('photos_found_urls')}")
        else:
            print(f"{i:>2}. {addr} | condition_score={cs} | label={lbl} | downloaded_images={c.get('downloaded_images')}")
    print("")

    arv_out = None
    if compute_arv is not None:
        print("STEP 8: ARV compute (top-3 selection + final ARV)\n")
        subject_for_arv = {"address": address, "beds": beds, "baths": baths, "sqft": sqft}
        subject_street = _extract_street_name(address)
        comps_for_arv = [_normalize_comp_for_arv(c, subject_street) for c in scored_ordered]
        arv_out = compute_arv(subject_for_arv, comps_for_arv, total_comps_returned=len(comps_all))
        print(f"ARV RESULT: {arv_out.get('status')} | {arv_out.get('message')}")
        if arv_out.get("arv") is not None:
            print(f"ARV: ${int(arv_out.get('arv')):,}")
        print("")

        try:
            selected = arv_out.get("selected_comps") or []
            if isinstance(selected, list) and selected:
                arv_out["selected_comps_enriched"] = _enrich_arv_selected_comps(selected, by_zpid, by_url)
        except Exception:
            pass

    return {
        "status": "ok",
        "ranked": ranked,
        "scored": scored_ordered,
        "arv": arv_out,
        "total_comps_collected": len(comps_all),
        "detail_items_count": len(detail_items) if isinstance(detail_items, list) else 0,
        "countable_comps_toward_target": countable_sim_ge,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "outlier_debug": outlier_dbg,
        "subject_thumbnail_url": subject_thumbnail_url,
        "subject_detail_url": subject_detail_url,
    }


# ============================================================
# Wrapper for unified subject dict (GUI/orchestrator)
# ============================================================
def _combine_address_from_subject(subject: dict) -> str:
    addr = (subject.get("address") or "").strip()
    if addr:
        return addr

    street = (subject.get("street") or "").strip()
    city = (subject.get("city") or "").strip()
    state = (subject.get("state") or "").strip()
    zip_code = (subject.get("zip") or subject.get("postal_code") or "").strip()

    parts = []
    if street:
        parts.append(street)
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if zip_code:
        parts.append(zip_code)

    return ", ".join(parts)


def _normalize_property_type(subject: dict) -> str:
    raw = (
        subject.get("property_type")
        or subject.get("prop_type")
        or ""
    )
    s = raw.strip().lower()

    if s in ("single family", "single-family", "single_family", "single", "sf", "sfr"):
        return "sf"

    if s in ("multi-family", "multi family", "multi_family", "multifamily", "mf", "multi"):
        return "mf"

    if "condo" in s or s == "c":
        return "c"

    if "townhouse" in s or "townhome" in s or s == "th":
        return "th"

    return s


def run_arv_pipeline(subject: dict) -> dict:
    address = _combine_address_from_subject(subject)

    beds_raw = subject.get("beds")
    baths_raw = subject.get("baths")
    sqft_raw = subject.get("sqft")
    year_raw = (
        subject.get("year_built")
        or subject.get("year")
        or subject.get("yearBuilt")
    )

    try:
        beds = float(beds_raw) if beds_raw is not None else 0.0
    except:
        beds = 0.0

    try:
        baths = float(baths_raw) if baths_raw is not None else 0.0
    except:
        baths = 0.0

    try:
        sqft = int(float(sqft_raw)) if sqft_raw is not None else 0
    except:
        sqft = 0

    try:
        year = int(float(year_raw)) if year_raw is not None else 0
    except:
        year = 0

    prop_type = _normalize_property_type(subject)

    return run_pipeline(address, beds, baths, sqft, year, prop_type, subject)


if __name__ == "__main__":
    address = input("Address: ").strip()
    beds = float(input("Bedrooms: ").strip())
    baths = float(input("Bathrooms: ").strip())
    sqft = int(float(input("Square Feet: ").strip()))
    year = int(float(input("Year Built: ").strip()))
    prop_type = input("Property Type (sf / mf / c / th): ").strip().lower()

    subject = {
        "address": address,
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year,
        "property_type": prop_type,
    }

    out = run_pipeline(address, beds, baths, sqft, year, prop_type, subject)
    print(json.dumps(out, indent=2))
