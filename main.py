import os
import sys
import json
import uuid
import tempfile
import traceback
from datetime import datetime
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import re
import time

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import pillow_heif

# ----------------------------
# SUPABASE (EMAIL LEADS)
# ----------------------------
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

class EmailLeadIn(BaseModel):
    email: EmailStr
    source: str = "email_gate"

# ----------------------------
# REDIS (OPTIONAL)
# ----------------------------
try:
    import redis
except Exception:
    redis = None

# Register HEIF/HEIC opener (enables PIL to read iPhone photos)
pillow_heif.register_heif_opener()

# ------------------------------------------------------------------
# PATH FIX (UPDATED FOR DEPLOY FOLDER STRUCTURE)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from orchestrator import run_full_underwrite  # full underwrite (ARV + REHAB)

# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your Shopify domain(s)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# DEBUG PING (DIAGNOSE MOBILE REACHABILITY)
# ------------------------------------------------------------------
@app.get("/debug/ping")
def debug_ping(request: Request):
    return {
        "ok": True,
        "ts": datetime.utcnow().isoformat(),
        "ip": request.client.host if request.client else None,
        "ua": request.headers.get("user-agent"),
        "origin": request.headers.get("origin"),
        "referer": request.headers.get("referer"),
    }

# ------------------------------------------------------------------
# REDIS (Render-safe, OPTIONAL)
# ------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

# Fallback: in-memory job store if REDIS_URL not set OR redis not installed
_inmem_jobs = {}

if REDIS_URL and redis is not None:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
    except Exception:
        redis_client = None
else:
    redis_client = None

JOB_TTL_SECONDS = 60 * 60 * 24  # 24 hours

def save_job(job_id: str, data: dict):
    if redis_client:
        redis_client.set(f"job:{job_id}", json.dumps(data), ex=JOB_TTL_SECONDS)
    else:
        _inmem_jobs[job_id] = data

def load_job(job_id: str) -> Optional[dict]:
    if redis_client:
        raw = redis_client.get(f"job:{job_id}")
        return json.loads(raw) if raw else None
    return _inmem_jobs.get(job_id)

# ------------------------------------------------------------------
# CACHE SETTINGS (OPTIONAL; DISK-BASED)
# ------------------------------------------------------------------
def _env_bool(name: str, default: str = "false") -> bool:
    v = (os.environ.get(name, default) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")

CACHE_ENABLED = _env_bool("CACHE_ENABLED", "false")
CACHE_DIR = (os.environ.get("CACHE_DIR", "/tmp/dottid_cache") or "/tmp/dottid_cache").strip()
try:
    CACHE_TTL_SECONDS = int((os.environ.get("CACHE_TTL_SECONDS", "43200") or "43200").strip())  # default 12 hours
except Exception:
    CACHE_TTL_SECONDS = 43200

def _ensure_cache_dirs():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "comp_images"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "meta"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "locks"), exist_ok=True)
    except Exception:
        pass

def _cleanup_comp_image_cache(ttl_seconds: int) -> int:
    """
    Deletes expired cached comp images (Option B cache), based on per-item metadata.
    Safe to call even if nothing is cached yet.

    Returns: number of cache entries removed
    """
    removed = 0
    try:
        meta_dir = os.path.join(CACHE_DIR, "meta")
        img_dir = os.path.join(CACHE_DIR, "comp_images")
        if not os.path.isdir(meta_dir):
            return 0

        now = time.time()

        for name in os.listdir(meta_dir):
            if not name.lower().endswith(".json"):
                continue

            meta_path = os.path.join(meta_dir, name)
            try:
                raw = open(meta_path, "r", encoding="utf-8").read()
                meta = json.loads(raw) if raw else {}
            except Exception:
                meta = {}

            downloaded_at = None
            try:
                downloaded_at = float(meta.get("downloaded_at")) if meta.get("downloaded_at") is not None else None
            except Exception:
                downloaded_at = None

            per_item_ttl = None
            try:
                per_item_ttl = int(meta.get("ttl_seconds")) if meta.get("ttl_seconds") is not None else None
            except Exception:
                per_item_ttl = None

            if downloaded_at is None:
                # Fallback: use metadata file mtime
                try:
                    downloaded_at = os.path.getmtime(meta_path)
                except Exception:
                    downloaded_at = now

            effective_ttl = per_item_ttl if (per_item_ttl is not None and per_item_ttl > 0) else ttl_seconds
            expired = (now - downloaded_at) > float(effective_ttl)

            if not expired:
                continue

            key = name.rsplit(".", 1)[0]
            # Delete image(s) by prefix match (extension may vary)
            try:
                for img_name in os.listdir(img_dir):
                    if img_name.startswith(key + ".") or img_name == key:
                        try:
                            os.remove(os.path.join(img_dir, img_name))
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                os.remove(meta_path)
            except Exception:
                pass

            removed += 1

    except Exception:
        return removed

    return removed

# ------------------------------------------------------------------
# WORKER POOL (BACKGROUND ONLY)
# ------------------------------------------------------------------
# CHANGE: default to 1 worker to reduce memory pressure on small instances.
try:
    _job_workers = int((os.environ.get("JOB_WORKERS", "1") or "1").strip())
except Exception:
    _job_workers = 1
_job_workers = max(1, min(_job_workers, 6))

executor = ThreadPoolExecutor(max_workers=_job_workers)

def normalize_upload_to_jpeg_bytes(filename: str, raw: bytes) -> Tuple[bytes, str]:
    im = Image.open(BytesIO(raw))
    if im.mode != "RGB":
        im = im.convert("RGB")

    out = BytesIO()
    im.save(out, format="JPEG", quality=92, optimize=True)
    out.seek(0)

    base = (filename or "image").rsplit(".", 1)[0]
    return out.read(), f"{base}.jpg"

def _save_uploaded_images_to_temp(image_blobs: List[Tuple[str, bytes]]) -> dict:
    temp_dir = tempfile.mkdtemp(prefix="dottid_")
    paths = []

    for (fname, raw) in image_blobs or []:
        try:
            jpeg_bytes, out_name = normalize_upload_to_jpeg_bytes(fname, raw)
            out_path = os.path.join(temp_dir, out_name or f"img_{len(paths)+1}.jpg")
            with open(out_path, "wb") as f:
                f.write(jpeg_bytes)
            paths.append(out_path)
        except Exception:
            continue

    return {"temp_dir": temp_dir, "files": paths}

def _extract_arv_value(arv_container):
    v = arv_container
    for _ in range(3):
        if isinstance(v, dict) and "arv" in v:
            v = v.get("arv")
        else:
            break
    return float(v)

# ------------------------------------------------------------------
# Upgrade Zillow thumbnail URLs
# ------------------------------------------------------------------
def upgrade_zillow_thumbnail_url(url: Optional[str], dims: str = "2048_1536") -> Optional[str]:
    if not url or not isinstance(url, str):
        return url

    u = url.strip()
    if not u:
        return u

    # CHANGE: If URL is an uncropped_scaled_within_* variant, force reliable cc_ft thumbnail.
    if "uncropped_scaled_within_" in u:
        base = u.split("-uncropped_scaled_within_")[0]
        return base + "-cc_ft_768.jpg"

    m = re.match(
        r"^(https?://photos\.zillowstatic\.com/fp/[^-]+)-cc_ft_\d+\.(jpg|jpeg|png|webp)$",
        u,
        re.IGNORECASE
    )
    if m:
        # Keep existing cc_ft URLs as-is.
        return u

    return u

# ------------------------------------------------------------------
# SUBJECT BUILD
# ------------------------------------------------------------------
def _build_subject(parsed: dict) -> dict:
    address_full = f"{parsed.get('address','')} {parsed.get('city','')}, {parsed.get('state','')}, {parsed.get('zip','')}".strip()

    return {
        "address": address_full,
        "street": parsed.get("address", ""),
        "city": parsed.get("city", ""),
        "state": parsed.get("state", ""),
        "zip": parsed.get("zip", ""),
        "beds": int(parsed.get("beds", 0) or 0),
        "baths": float(parsed.get("baths", 0) or 0),
        "sqft": int(parsed.get("sqft", 0) or 0),
        "year_built": int(parsed.get("yearBuilt", 0) or 0),
        "property_type": (
            "sf" if "single" in (parsed.get("propertyType","") or "").lower() else
            "mf" if "multi" in (parsed.get("propertyType","") or "").lower() else
            "c"  if "condo" in (parsed.get("propertyType","") or "").lower() else
            "th" if "town" in (parsed.get("propertyType","") or "").lower() else
            "sf"
        ),
        "deal_type": parsed.get("deal_type") or parsed.get("dealType", ""),
        "assignment_fee": parsed.get("assignment_fee") or parsed.get("assignmentFee", ""),
        "units": parsed.get("units", ""),

        # ✅ FIX: pass through user-selected condition so estimator can use it
        "condition": parsed.get("condition", ""),

        "kitchen_updated": parsed.get("kitchen", ""),
        "bath_updated": parsed.get("bath", ""),
        "roof": parsed.get("roof", ""),
        "hvac": parsed.get("hvac", ""),
        "foundation": parsed.get("foundation", ""),
        "kitchen_age": parsed.get("kitchen_age", None),
        "bath_age": parsed.get("bath_age", None),
        "roof_needed": parsed.get("roof_needed", None),
        "hvac_needed": parsed.get("hvac_needed", None),
        "foundation_issues": parsed.get("foundation_issues", None),
    }

# ------------------------------------------------------------------
# PROCESS JOB
# ------------------------------------------------------------------
def process_job(job_id: str, payload: dict):
    job = load_job(job_id)
    if not job:
        return

    def log(msg: str):
        try:
            print(f"[JOB {job_id}] {msg}", flush=True)
        except Exception:
            pass

    job["status"] = "processing"
    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

    t_job_start = time.perf_counter()

    try:
        if CACHE_ENABLED:
            _ensure_cache_dirs()
            try:
                removed = _cleanup_comp_image_cache(CACHE_TTL_SECONDS)
                if removed:
                    log(f"Cache cleanup: removed_expired_comp_images={removed} ttl_seconds={CACHE_TTL_SECONDS}")
            except Exception:
                pass

        subject = payload.get("subject", {}) or {}
        image_blobs = payload.get("image_blobs", []) or []

        log("START process_job()")
        log(f"Input address='{subject.get('address','')}' beds={subject.get('beds')} baths={subject.get('baths')} sqft={subject.get('sqft')} year_built={subject.get('year_built')}")
        log(f"Input deal_type='{subject.get('deal_type','')}' assignment_fee='{subject.get('assignment_fee','')}' condition='{subject.get('condition','')}' units='{subject.get('units','')}'")
        log(f"Images received (raw upload count) = {len(image_blobs)}")

        t_save_start = time.perf_counter()
        img_info = _save_uploaded_images_to_temp(image_blobs)
        t_save_end = time.perf_counter()

        subject["uploaded_image_paths"] = img_info.get("files", [])
        subject["uploaded_image_temp_dir"] = img_info.get("temp_dir")

        log(f"Saved images to temp. temp_dir='{subject.get('uploaded_image_temp_dir')}' valid_saved_files={len(subject.get('uploaded_image_paths') or [])}")
        log(f"TIMING saved_upload_images_seconds={(t_save_end - t_save_start):.3f}")

        log("Calling run_full_underwrite(subject)...")
        t_underwrite_start = time.perf_counter()
        result = run_full_underwrite(subject)
        t_underwrite_end = time.perf_counter()
        log("Returned from run_full_underwrite(subject).")
        log(f"TIMING run_full_underwrite_seconds={(t_underwrite_end - t_underwrite_start):.3f}")

        # ------------------------------------------------------------------
        # CHANGE: if underwriting returns nothing/invalid, treat as NOT_ENOUGH_USABLE_COMPS
        # (prevents blank/failed ARV outcome when searches return 0 comps)
        # ------------------------------------------------------------------
        if not result or not isinstance(result, dict):
            log("UNDERWRITE RESULT INVALID -> treating as NOT_ENOUGH_USABLE_COMPS fallback.")
            rehab_raw = {}
            rehab = 45000
            try:
                rehab_raw = (result or {}).get("rehab", {}) if isinstance(result, dict) else {}
                rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))
            except Exception:
                rehab = 45000

            job["status"] = "complete"
            job["result"] = {
                "subject_address": subject.get("address", ""),
                "arv": "NOT_ENOUGH_USABLE_COMPS",
                "arv_str": "NOT_ENOUGH_USABLE_COMPS",
                "estimated_rehab": rehab,
                "estimated_rehab_str": f"${rehab:,.0f}",
                "max_offer": None,
                "max_offer_str": "",
                "comps": [],
            }
            job["error"] = None
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)

            log("FINAL RESULT (fallback invalid underwriting):")
            try:
                log(json.dumps(job["result"], indent=2))
            except Exception:
                log(str(job["result"]))

            log(f"TIMING total_job_seconds={(time.perf_counter() - t_job_start):.3f}")
            return

        arv_obj = result.get("arv")
        rehab_raw = result.get("rehab", {}) if isinstance(result, dict) else {}
        try:
            rehab_dbg = int(float(rehab_raw.get("estimate_numeric", 45000)))
        except Exception:
            rehab_dbg = 45000
        log(f"Underwrite debug: arv_obj_type={type(arv_obj).__name__} rehab_estimate_numeric={rehab_dbg}")

        # ------------------------------------------------------------------
        # CHANGE: if ARV object missing/invalid, treat as NOT_ENOUGH_USABLE_COMPS
        # ------------------------------------------------------------------
        if not isinstance(arv_obj, dict):
            log("ARV OBJECT INVALID -> treating as NOT_ENOUGH_USABLE_COMPS fallback.")
            try:
                rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))
            except Exception:
                rehab = 45000

            job["status"] = "complete"
            job["result"] = {
                "subject_address": subject.get("address", ""),
                "arv": "NOT_ENOUGH_USABLE_COMPS",
                "arv_str": "NOT_ENOUGH_USABLE_COMPS",
                "estimated_rehab": rehab,
                "estimated_rehab_str": f"${rehab:,.0f}",
                "max_offer": None,
                "max_offer_str": "",
                "comps": [],
            }
            job["error"] = None
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)

            log("FINAL RESULT (fallback invalid ARV object):")
            try:
                log(json.dumps(job["result"], indent=2))
            except Exception:
                log(str(job["result"]))

            log(f"TIMING total_job_seconds={(time.perf_counter() - t_job_start):.3f}")
            return

        # ------------------------------------------------------------------
        # CHANGE: Graceful completion when ARV cannot be computed due to comps
        # ------------------------------------------------------------------
        arv_status = str(arv_obj.get("status") or "").lower().strip()
        arv_msg = str(arv_obj.get("message") or "").upper().strip()
        arv_value_direct = arv_obj.get("arv", None)

        log(f"ARV status='{arv_status}' message='{arv_msg}' arv_value_direct='{arv_value_direct}'")

        if (
            "NOT_ENOUGH_USABLE_COMPS" in arv_msg
            or "NOT_ENOUGH_COMPS" in arv_msg
            or (arv_value_direct is None and "NOT_ENOUGH" in arv_msg)
            or (arv_value_direct is None and arv_status in ["fail", "failed"])
        ):
            log("ARV indicates NOT_ENOUGH_* -> returning graceful NOT_ENOUGH_USABLE_COMPS result.")
            rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))

            job["status"] = "complete"
            job["result"] = {
                "subject_address": subject.get("address", ""),
                "arv": "NOT_ENOUGH_USABLE_COMPS",
                "arv_str": "NOT_ENOUGH_USABLE_COMPS",
                "estimated_rehab": rehab,
                "estimated_rehab_str": f"${rehab:,.0f}",
                "max_offer": None,
                "max_offer_str": "",
                "comps": [],
            }
            job["error"] = None
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)

            log("FINAL RESULT (NOT_ENOUGH_*):")
            try:
                log(json.dumps(job["result"], indent=2))
            except Exception:
                log(str(job["result"]))

            log(f"TIMING total_job_seconds={(time.perf_counter() - t_job_start):.3f}")
            return

        arv = int(_extract_arv_value(arv_obj))
        log(f"Extracted ARV={arv}")

        rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))
        log(f"Extracted Rehab={rehab}")

        deal_type = (subject.get("deal_type") or "").lower().strip()
        assignment_fee = float(subject.get("assignment_fee") or 0)
        log(f"MAO inputs: deal_type='{deal_type}' assignment_fee={assignment_fee}")

        if deal_type == "rental":
            mao = int(arv * 0.85 - rehab)
            log("MAO formula: arv*0.85 - rehab")
        elif deal_type == "flip":
            mao = int(arv * 0.75 - rehab)
            log("MAO formula: arv*0.75 - rehab")
        elif deal_type == "wholesale":
            mao = int(arv * 0.75 - rehab - assignment_fee)
            log("MAO formula: arv*0.75 - rehab - assignment_fee")
        else:
            mao = int(arv * 0.75 - rehab)
            log("MAO formula: arv*0.75 - rehab (default)")

        mao = max(mao, 0)
        log(f"Computed MAO={mao} (clamped >=0)")

        # ----------------------------
        # COMPS: pull from underwriting result (enriched comps)
        # ----------------------------
        comps_out = []

        # Most likely location
        try:
            if isinstance(arv_obj, dict):
                comps_out = arv_obj.get("selected_comps_enriched") or []
        except Exception:
            comps_out = []

        # Fallbacks (in case orchestrator nests it differently)
        if not comps_out:
            try:
                arv2 = (result.get("arv") if isinstance(result, dict) else None)
                if isinstance(arv2, dict):
                    comps_out = arv2.get("selected_comps_enriched") or []
            except Exception:
                comps_out = []

        if not comps_out:
            try:
                if isinstance(arv_obj, dict):
                    comps_out = arv_obj.get("selected_comps") or []
            except Exception:
                comps_out = []

        log(f"Comps selected (raw count)={len(comps_out) if isinstance(comps_out, list) else 0}")

        # Normalize + upgrade thumbnail URLs
        normalized_comps = []
        for c in comps_out if isinstance(comps_out, list) else []:
            if not isinstance(c, dict):
                continue

            thumb = c.get("thumbnail_url") or c.get("thumbnail") or c.get("thumb")
            thumb = upgrade_zillow_thumbnail_url(thumb)

            normalized_comps.append(
                {
                    "zpid": c.get("zpid"),
                    "address": c.get("address"),
                    "sold_price": c.get("sold_price"),
                    "sold_price_str": (f"${int(c.get('sold_price')):,.0f}" if c.get("sold_price") else None),
                    "distance_miles": c.get("distance_miles"),
                    "beds": c.get("beds"),
                    "baths": c.get("baths"),
                    "sqft": c.get("sqft"),
                    "zillow_url": c.get("zillow_url") or c.get("url"),
                    "thumbnail_url": thumb,
                }
            )

        log(f"Comps normalized (count)={len(normalized_comps)}")

        job["status"] = "complete"
        job["result"] = {
            "subject_address": subject.get("address", ""),
            "arv": arv,
            "arv_str": f"${arv:,.0f}",
            "estimated_rehab": rehab,
            "estimated_rehab_str": f"${rehab:,.0f}",
            "max_offer": mao,
            "max_offer_str": f"${mao:,.0f}",
            "comps": normalized_comps,
        }
        job["error"] = None

        log("FINAL RESULT (will be returned by GET /jobs/results/{job_id}):")
        try:
            log(json.dumps(job["result"], indent=2))
        except Exception:
            log(str(job["result"]))

    except Exception as e:
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        try:
            print(f"[JOB {job_id}] FAILED: {str(e)}", flush=True)
        except Exception:
            pass

    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

    try:
        log(f"TIMING total_job_seconds={(time.perf_counter() - t_job_start):.3f}")
    except Exception:
        pass

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.post("/jobs/create")
async def create_job(form_data: Optional[str] = Form(None)):
    parsed = json.loads(form_data) if form_data else {}
    subject = _build_subject(parsed)

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "input": {"address": subject.get("address", ""), "images_received": 0},
        "result": None,
        "error": None,
    }
    save_job(job_id, job)

    executor.submit(process_job, job_id, {"subject": subject, "image_blobs": []})

    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/start")
async def start_job(form_data: Optional[str] = Form(None), images: List[UploadFile] = File([])):
    parsed = json.loads(form_data) if form_data else {}
    subject = _build_subject(parsed)

    image_blobs = []
    for img in images or []:
        raw = await img.read()
        if raw:
            image_blobs.append((img.filename or "image", raw))

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "input": {"address": subject.get("address", ""), "images_received": len(image_blobs)},
        "result": None,
        "error": None,
    }

    save_job(job_id, job)
    executor.submit(process_job, job_id, {"subject": subject, "image_blobs": image_blobs})

    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/status/{job_id}")
def job_status(job_id: str):
    job = load_job(job_id)
    if not job:
        return {"status": "not_found"}

    # CHANGE: include images_received in status to verify whether backend received uploads
    images_received = 0
    try:
        images_received = int((job.get("input") or {}).get("images_received") or 0)
    except Exception:
        images_received = 0

    return {
        "status": job.get("status"),
        "has_result": job.get("status") == "complete",
        "error": job.get("error"),
        "updated_at": job.get("updated_at"),
        "images_received": images_received,
    }

@app.get("/jobs/results/{job_id}")
def job_results(job_id: str):
    job = load_job(job_id)
    if not job:
        return {"error": "not_found"}

    # ------------------------------------------------------------------
    # CHANGE: failed jobs should be terminal (not "not_ready" forever)
    # ------------------------------------------------------------------
    if job.get("status") == "failed":
        return {"error": "failed", "message": job.get("error") or "Unknown error"}

    if job.get("status") != "complete":
        return {"error": "not_ready"}

    return job.get("result")

# ------------------------------------------------------------------
# EMAIL LEADS (OPTION B)
# ------------------------------------------------------------------
@app.post("/leads/email")
def create_email_lead(payload: EmailLeadIn):
    if supabase is None:
        return {"ok": False, "error": "supabase_not_configured"}

    row = {
        "email": str(payload.email).lower().strip(),
        "source": payload.source,
    }

    try:
        supabase.table("email_leads").insert(row).execute()
        return {"ok": True}
    except Exception:
        return {"ok": False, "error": "insert_failed"}
