# download_models.py
import os
from pathlib import Path

import boto3

DEFAULT_BUCKET = os.environ.get("MODEL_S3_BUCKET", "dottid-backend-models").strip()
DEFAULT_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-2")).strip()

MODEL_FILENAMES = [
    "FINAL MVP ROOM TYPE MODEL.pth",
    "FINAL MVP CONDITION MODEL.pth",
    "FINAL MVP VALIDATOR MODEL.pth",
]

def ensure_models_local(model_dir: Path) -> dict:
    """
    Ensures required model files exist locally in model_dir.
    Downloads from S3 if missing.

    Env expected on Render:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - (optional) AWS_SESSION_TOKEN
      - AWS_REGION or AWS_DEFAULT_REGION (you have us-east-2)
      - (optional) MODEL_S3_BUCKET
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=DEFAULT_REGION)

    downloaded = []
    for fname in MODEL_FILENAMES:
        local_path = model_dir / fname
        if local_path.exists() and local_path.stat().st_size > 0:
            continue

        # models sit directly under bucket root (no folder prefix)
        s3.download_file(DEFAULT_BUCKET, fname, str(local_path))
        downloaded.append(fname)

    return {"bucket": DEFAULT_BUCKET, "region": DEFAULT_REGION, "downloaded": downloaded}
