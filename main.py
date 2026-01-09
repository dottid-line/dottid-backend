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

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import pillow_heif

import redis

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
# REDIS (Render-safe)
# ------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

# Fallback: in-memory job store if REDIS_URL not set (OK for testing, NOT for production)
_inmem_jobs = {}

if REDIS_URL:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
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
# WORKER POOL (BACKGROUND ONLY)
# ------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=6)

def normalize_upload_to_jpeg_bytes(filename: str, raw: bytes) -> Tuple[bytes, str]:
    """
    Converts any readable image (including HEIC/HEIF) into JPEG bytes.
    Returns (jpeg_bytes, output_filename.jpg).
    """
    im = Image.open(BytesIO(raw))
    if im.mode != "RGB":
        im = im.convert("RGB")

    out = BytesIO()
    im.save(out, format="JPEG", quality=92, optimize=True)
    out.seek(0)

    base = (filename or "image").rsplit(".", 1)[0]
    return out.read(), f"{base}.jpg"

def _save_uploaded_images_to_temp(image_blobs: List[Tuple[str, bytes]]) -> dict:
    """
    Runs inside the worker thread.
    Accepts list of (filename, raw_bytes).
    Writes them to a temp folder as JPEG.
    Returns {"temp_dir": str, "files": [paths...]}.
    """
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
# Upgrade Zillow thumbnail URLs (keep as you had it)
# ------------------------------------------------------------------
def upgrade_zillow_thumbnail_url(url: Optional[str], dims: str = "2048_1536") -> Optional[str]:
    if not url or not isinstance(url, str):
        return url

    u = url.strip()
    if not u:
        return u

    if "uncropped_scaled_within_" in u:
        u2 = re.sub(
            r"(uncropped_scaled_within_)\d+_\d+",
            r"\g<1>" + dims,
            u,
            flags=re.IGNORECASE
        )
        return u2

    m = re.match(
        r"^(https?://photos\.zillowstatic\.com/fp/[^-]+)-cc_ft_\d+\.(jpg|jpeg|png|webp)$",
        u,
        re.IGNORECASE
    )
    if m:
        base = m.group(1)
        return base + f"-uncropped_scaled_within_{dims}.jpg"

    return u

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
        # Prefer normalized keys if present
        "deal_type": parsed.get("deal_type") or parsed.get("dealType", ""),
        "assignment_fee": parsed.get("assignment_fee") or parsed.get("assignmentFee", ""),
        "units": parsed.get("units", ""),
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

def process_job(job_id: str, payload: dict):
    job = load_job(job_id)
    if not job:
        return

    job["status"] = "processing"
    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

    try:
        image_blobs = payload.get("image_blobs", [])  # list[(filename, bytes)]
        img_info = _save_uploaded_images_to_temp(image_blobs)

        subject = payload.get("subject", {}) or {}

        # IMPORTANT: make uploaded image paths available to pipeline via subject
        subject["uploaded_image_paths"] = img_info.get("files", [])
        subject["uploaded_image_temp_dir"] = img_info.get("temp_dir")

        job["input"]["image_temp_dir"] = img_info.get("temp_dir")
        job["input"]["image_files_saved"] = len(img_info.get("files", []))
        job["updated_at"] = datetime.utcnow().isoformat()
        save_job(job_id, job)

        result = run_full_underwrite(subject)

        if not result or not isinstance(result, dict):
            job["status"] = "failed"
            job["result"] = None
            job["error"] = "No comps returned for this address. Cannot compute ARV."
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)
            return

        arv_obj = result.get("arv")
        if not isinstance(arv_obj, dict):
            job["status"] = "failed"
            job["result"] = None
            job["error"] = "Pipeline returned invalid ARV object."
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)
            return

        # Handle NOT_ENOUGH_USABLE_COMPS / ARV=None cleanly
        arv_status = (arv_obj.get("status") or "").lower().strip()
        arv_reason = arv_obj.get("reason") or arv_obj.get("error") or "Unable to compute ARV."

        arv_value_candidate = arv_obj.get("arv", None)
        if isinstance(arv_value_candidate, dict):
            arv_value_candidate = arv_value_candidate.get("arv", None)

        if arv_status == "fail" or arv_value_candidate is None:
            job["status"] = "failed"
            job["result"] = None
            job["error"] = arv_reason
            job["updated_at"] = datetime.utcnow().isoformat()
            save_job(job_id, job)
            return

        arv = int(_extract_arv_value(arv_obj))

        rehab_raw = result.get("rehab", None)
        if isinstance(rehab_raw, dict):
            rehab_val = rehab_raw.get("estimate_numeric", None)
            if rehab_val is None:
                rehab_val = rehab_raw.get("estimate", None)
        else:
            rehab_val = rehab_raw

        rehab = int(float(rehab_val)) if rehab_val is not None else 45000

        # Compute MAO based on deal type
        def _safe_float(v, default=0.0):
            try:
                if v is None:
                    return default
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v).strip().replace(",", "").replace("$", "")
                if s == "":
                    return default
                return float(s)
            except Exception:
                return default

        deal_type = (subject.get("deal_type") or "").lower().strip()
        assignment_fee = _safe_float(subject.get("assignment_fee"), 0.0)

        if deal_type == "rental":
            mao_val = (float(arv) * 0.85) - float(rehab)
        elif deal_type == "flip":
            mao_val = (float(arv) * 0.75) - float(rehab)
        elif deal_type == "wholesale":
            mao_val = (float(arv) * 0.75) - float(rehab) - float(assignment_fee)
        else:
            mao_val = (float(arv) * 0.75) - float(rehab)

        mao = int(max(mao_val, 0.0))

        comps = []
        try:
            selected_enriched = arv_obj.get("selected_comps_enriched")
            if isinstance(selected_enriched, list) and selected_enriched:
                for i, c in enumerate(selected_enriched, start=1):
                    if not isinstance(c, dict):
                        continue

                    thumb_url = c.get("thumbnail_url")

                    thumb_url_1x = upgrade_zillow_thumbnail_url(thumb_url, "1344_1008")
                    thumb_url_2x = upgrade_zillow_thumbnail_url(thumb_url, "2048_1536")

                    sold_price = c.get("sold_price")

                    dist_val = round(float(c.get("distance_miles", 0) or 0), 2)
                    beds_val = c.get("beds")
                    baths_val = c.get("baths")
                    sqft_val = c.get("sqft")

                    comps.append({
                        "rank": i,
                        "zpid": c.get("zpid"),
                        "zillow_url": c.get("zillow_url"),
                        "thumbnail_url": thumb_url_1x,
                        "thumbnail_url_2x": thumb_url_2x,
                        "address": c.get("address"),
                        "sold_price": sold_price,
                        "sold_price_str": f"${int(sold_price):,}" if sold_price else None,
                        "distance_miles": dist_val,
                        "distance_miles_str": f"{dist_val:.2f} mi",
                        "beds": beds_val,
                        "baths": baths_val,
                        "sqft": sqft_val,
                        "beds_baths_sqft_str": f"{beds_val} bd / {baths_val} ba / {sqft_val} sqft",
                    })
        except Exception:
            comps = []

        job["status"] = "complete"
        job["result"] = {
            "arv": arv,
            "arv_str": f"${arv:,.0f}",
            "estimated_rehab": rehab,
            "estimated_rehab_str": f"${rehab:,.0f}",
            "max_offer": mao,
            "max_offer_str": f"${mao:,.0f}",
            "comps": comps,
        }
        job["error"] = None

    except Exception as e:
        traceback.print_exc()
        job["status"] = "failed"
        job["result"] = None
        job["error"] = str(e)

    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

# ------------------------------------------------------------------
# Shopify uses this endpoint in your current theme flow
# ------------------------------------------------------------------
@app.post("/jobs/create")
async def create_job(form_data: Optional[str] = Form(None)):
    try:
        parsed = json.loads(form_data) if form_data else {}
    except Exception:
        parsed = {}

    subject = _build_subject(parsed)

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "input": {
            "address": subject.get("address", ""),
            "images_received": 0,
        },
        "result": None,
        "error": None,
    }
    save_job(job_id, job)

    # Start processing even if no images (your UI supports this path)
    executor.submit(process_job, job_id, {"subject": subject, "image_blobs": []})

    return {"job_id": job_id, "status": "queued"}

@app.post("/jobs/start")
async def start_job(
    form_data: Optional[str] = Form(None),
    images: List[UploadFile] = File([]),
):
    try:
        parsed = json.loads(form_data) if form_data else {}
    except Exception:
        parsed = {}

    subject = _build_subject(parsed)

    # IMPORTANT: read bytes while request is still alive
    image_blobs: List[Tuple[str, bytes]] = []
    for img in images or []:
        try:
            img.file.seek(0)
            raw = await img.read()
            if raw:
                image_blobs.append((img.filename or "image", raw))
        except Exception:
            continue

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "input": {
            "address": subject.get("address", ""),
            "images_received": len(image_blobs),
        },
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

    return {
        "status": job.get("status"),
        "has_result": job.get("status") == "complete",
        "error": job.get("error"),
        "updated_at": job.get("updated_at"),
    }

@app.get("/jobs/results/{job_id}")
def job_results(job_id: str):
    job = load_job(job_id)
    if not job:
        return {"error": "not_found"}

    if job.get("status") != "complete":
        return {"error": "not_ready"}

    return job.get("result")
