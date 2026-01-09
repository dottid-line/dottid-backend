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
# WORKER POOL (BACKGROUND ONLY)
# ------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=6)

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

    job["status"] = "processing"
    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

    try:
        image_blobs = payload.get("image_blobs", [])
        img_info = _save_uploaded_images_to_temp(image_blobs)

        subject = payload.get("subject", {}) or {}
        subject["uploaded_image_paths"] = img_info.get("files", [])
        subject["uploaded_image_temp_dir"] = img_info.get("temp_dir")

        result = run_full_underwrite(subject)

        if not result or not isinstance(result, dict):
            job["status"] = "failed"
            job["error"] = "No comps returned."
            save_job(job_id, job)
            return

        arv_obj = result.get("arv")
        if not isinstance(arv_obj, dict):
            job["status"] = "failed"
            job["error"] = "Invalid ARV object."
            save_job(job_id, job)
            return

        arv = int(_extract_arv_value(arv_obj))

        rehab_raw = result.get("rehab", {})
        rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))

        deal_type = (subject.get("deal_type") or "").lower().strip()
        assignment_fee = float(subject.get("assignment_fee") or 0)

        if deal_type == "rental":
            mao = int(arv * 0.85 - rehab)
        elif deal_type == "flip":
            mao = int(arv * 0.75 - rehab)
        elif deal_type == "wholesale":
            mao = int(arv * 0.75 - rehab - assignment_fee)
        else:
            mao = int(arv * 0.75 - rehab)

        mao = max(mao, 0)

        job["status"] = "complete"
        job["result"] = {
            "arv": arv,
            "arv_str": f"${arv:,.0f}",
            "estimated_rehab": rehab,
            "estimated_rehab_str": f"${rehab:,.0f}",
            "max_offer": mao,
            "max_offer_str": f"${mao:,.0f}",
            "comps": [],
        }
        job["error"] = None

    except Exception as e:
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)

    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

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
