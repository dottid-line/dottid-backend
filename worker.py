import os
import json
import time
import traceback
import tempfile
from datetime import datetime
from typing import Optional, List

# Redis
import redis

# AWS S3
import boto3

# Your underwriting orchestrator
from orchestrator import run_full_underwrite


# ----------------------------
# ENV
# ----------------------------
REDIS_URL = (os.environ.get("REDIS_URL", "") or "").strip()
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is required")

JOBS_QUEUE_KEY = (os.environ.get("JOBS_QUEUE_KEY", "") or "").strip() or "jobs:pending"

try:
    WORKER_CONCURRENCY = int((os.environ.get("WORKER_CONCURRENCY", "1") or "1").strip())
except Exception:
    WORKER_CONCURRENCY = 1
WORKER_CONCURRENCY = max(1, min(WORKER_CONCURRENCY, 8))

try:
    JOB_LOCK_TTL_SECONDS = int((os.environ.get("JOB_LOCK_TTL_SECONDS", "900") or "900").strip())
except Exception:
    JOB_LOCK_TTL_SECONDS = 900

# Jobs S3 (same vars you already use)
JOBS_S3_BUCKET = (os.environ.get("JOBS_S3_BUCKET", "") or "").strip()
JOBS_AWS_REGION = (os.environ.get("JOBS_AWS_REGION", "") or "").strip()
JOBS_AWS_ACCESS_KEY_ID = (os.environ.get("JOBS_AWS_ACCESS_KEY_ID", "") or "").strip()
JOBS_AWS_SECRET_ACCESS_KEY = (os.environ.get("JOBS_AWS_SECRET_ACCESS_KEY", "") or "").strip()

if not (JOBS_S3_BUCKET and JOBS_AWS_REGION and JOBS_AWS_ACCESS_KEY_ID and JOBS_AWS_SECRET_ACCESS_KEY):
    raise RuntimeError("Jobs S3 not configured (missing JOBS_* env vars)")

JOB_TTL_SECONDS = 60 * 60 * 24  # should match main.py, but worker can be independent


# ----------------------------
# CLIENTS
# ----------------------------
r = redis.from_url(REDIS_URL, decode_responses=True)
r.ping()

s3 = boto3.client(
    "s3",
    region_name=JOBS_AWS_REGION,
    aws_access_key_id=JOBS_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=JOBS_AWS_SECRET_ACCESS_KEY,
)


# ----------------------------
# HELPERS
# ----------------------------
def log(msg: str):
    try:
        print(f"[WORKER] {msg}", flush=True)
    except Exception:
        pass


def load_job(job_id: str) -> Optional[dict]:
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


def save_job(job_id: str, data: dict):
    r.set(f"job:{job_id}", json.dumps(data), ex=JOB_TTL_SECONDS)


def s3_get_json(key: str) -> dict:
    resp = s3.get_object(Bucket=JOBS_S3_BUCKET, Key=key)
    raw = resp["Body"].read()
    return json.loads(raw.decode("utf-8"))


def s3_put_json(key: str, data: dict):
    body = json.dumps(data).encode("utf-8")
    s3.put_object(
        Bucket=JOBS_S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def acquire_lock(job_id: str) -> bool:
    """
    Simple distributed lock using Redis SET NX EX.
    """
    lock_key = f"joblock:{job_id}"
    try:
        return bool(r.set(lock_key, "1", nx=True, ex=JOB_LOCK_TTL_SECONDS))
    except Exception:
        return False


def release_lock(job_id: str):
    lock_key = f"joblock:{job_id}"
    try:
        r.delete(lock_key)
    except Exception:
        pass


def download_uploads_to_temp(upload_keys: List[str]) -> (str, List[str]):
    temp_dir = tempfile.mkdtemp(prefix="dottid_worker_")
    paths: List[str] = []

    for key in upload_keys or []:
        try:
            resp = s3.get_object(Bucket=JOBS_S3_BUCKET, Key=key)
            b = resp["Body"].read()
            out_path = os.path.join(temp_dir, os.path.basename(key))
            with open(out_path, "wb") as f:
                f.write(b)
            paths.append(out_path)
        except Exception:
            continue

    return temp_dir, paths


def process_one_job(job_id: str):
    job = load_job(job_id)
    if not job:
        log(f"job not found in redis: {job_id}")
        return

    if job.get("status") == "complete":
        log(f"already complete: {job_id}")
        return

    # Move status -> processing
    job["status"] = "processing"
    job["updated_at"] = datetime.utcnow().isoformat()
    save_job(job_id, job)

    try:
        inputs_key = job.get("inputs_s3_key") or f"jobs/inputs/{job_id}.json"
        inputs = s3_get_json(inputs_key)  # contains subject + uploaded_image_s3_keys

        subject = (inputs.get("subject") or {}) if isinstance(inputs, dict) else {}
        upload_keys = (inputs.get("uploaded_image_s3_keys") or []) if isinstance(inputs, dict) else []
        if not isinstance(upload_keys, list):
            upload_keys = []

        log(f"START job={job_id} uploads={len(upload_keys)} address='{subject.get('address','')}'")

        # Download images to temp
        temp_dir, paths = download_uploads_to_temp(upload_keys)
        subject["uploaded_image_paths"] = paths
        subject["uploaded_image_temp_dir"] = temp_dir

        # Run underwriting
        t0 = time.perf_counter()
        result = run_full_underwrite(subject)
        t1 = time.perf_counter()
        log(f"underwrite seconds={(t1 - t0):.3f} job={job_id}")

        # If invalid result, store graceful NOT_ENOUGH_USABLE_COMPS
        if not result or not isinstance(result, dict) or not isinstance(result.get("arv"), dict):
            rehab = 45000
            try:
                rehab_raw = (result or {}).get("rehab", {}) if isinstance(result, dict) else {}
                rehab = int(float(rehab_raw.get("estimate_numeric", 45000)))
            except Exception:
                rehab = 45000

            out_obj = {
                "subject_address": subject.get("address", ""),
                "arv": "NOT_ENOUGH_USABLE_COMPS",
                "arv_str": "NOT_ENOUGH_USABLE_COMPS",
                "estimated_rehab": rehab,
                "estimated_rehab_str": f"${rehab:,.0f}",
                "max_offer": None,
                "max_offer_str": "",
                "comps": [],
            }
        else:
            arv_obj = result.get("arv") or {}
            rehab_raw = result.get("rehab", {}) or {}
            # MAO logic matches your existing main.py
            try:
                arv_val = int(float(arv_obj.get("arv")))
            except Exception:
                arv_val = None

            try:
                rehab_val = int(float(rehab_raw.get("estimate_numeric", 45000)))
            except Exception:
                rehab_val = 45000

            arv_status = str(arv_obj.get("status") or "").lower().strip()
            arv_msg = str(arv_obj.get("message") or "").upper().strip()

            if (
                "NOT_ENOUGH_USABLE_COMPS" in arv_msg
                or "NOT_ENOUGH_COMPS" in arv_msg
                or (arv_val is None and "NOT_ENOUGH" in arv_msg)
                or (arv_val is None and arv_status in ["fail", "failed"])
            ):
                out_obj = {
                    "subject_address": subject.get("address", ""),
                    "arv": "NOT_ENOUGH_USABLE_COMPS",
                    "arv_str": "NOT_ENOUGH_USABLE_COMPS",
                    "estimated_rehab": rehab_val,
                    "estimated_rehab_str": f"${rehab_val:,.0f}",
                    "max_offer": None,
                    "max_offer_str": "",
                    "comps": [],
                }
            else:
                deal_type = (subject.get("deal_type") or "").lower().strip()
                try:
                    assignment_fee = float(subject.get("assignment_fee") or 0)
                except Exception:
                    assignment_fee = 0.0

                if deal_type == "rental":
                    mao = int(arv_val * 0.85 - rehab_val)
                elif deal_type == "flip":
                    mao = int(arv_val * 0.75 - rehab_val)
                elif deal_type == "wholesale":
                    mao = int(arv_val * 0.75 - rehab_val - assignment_fee)
                else:
                    mao = int(arv_val * 0.75 - rehab_val)

                mao = max(mao, 0)

                # comps (same field your UI expects)
                comps_out = []
                try:
                    comps_out = arv_obj.get("selected_comps_enriched") or arv_obj.get("selected_comps") or []
                except Exception:
                    comps_out = []

                out_obj = {
                    "subject_address": subject.get("address", ""),
                    "arv": arv_val,
                    "arv_str": f"${arv_val:,.0f}",
                    "estimated_rehab": rehab_val,
                    "estimated_rehab_str": f"${rehab_val:,.0f}",
                    "max_offer": mao,
                    "max_offer_str": f"${mao:,.0f}",
                    "comps": comps_out if isinstance(comps_out, list) else [],
                }

        outputs_key = job.get("outputs_s3_key") or f"jobs/outputs/{job_id}.json"
        s3_put_json(outputs_key, out_obj)

        job["status"] = "complete"
        job["result"] = None  # keep redis small; source of truth is S3
        job["error"] = None
        job["outputs_s3_key"] = outputs_key
        job["updated_at"] = datetime.utcnow().isoformat()
        save_job(job_id, job)

        log(f"DONE job={job_id} outputs_key={outputs_key}")

    except Exception as e:
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.utcnow().isoformat()
        save_job(job_id, job)
        log(f"FAILED job={job_id} err={str(e)}")


def worker_loop():
    log(f"boot ok. queue={JOBS_QUEUE_KEY} concurrency={WORKER_CONCURRENCY}")

    while True:
        try:
            # Block waiting for one job_id
            item = r.blpop(JOBS_QUEUE_KEY, timeout=20)
            if not item:
                continue

            _, job_id = item
            job_id = (job_id or "").strip()
            if not job_id:
                continue

            if not acquire_lock(job_id):
                # Someone else is doing it; skip
                continue

            try:
                process_one_job(job_id)
            finally:
                release_lock(job_id)

        except Exception:
            traceback.print_exc()
            time.sleep(2)


if __name__ == "__main__":
    # NOTE: This file runs as the Render Background Worker "Start Command".
    worker_loop()
