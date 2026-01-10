# download_models.py
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = os.environ.get("MODEL_S3_BUCKET", "dottid-backend-models").strip()
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-2")).strip()

MODEL_FILES = [
    "FINAL MVP ROOM TYPE MODEL.pth",
    "FINAL MVP CONDITION MODEL.pth",
    "FINAL MVP VALIDATOR MODEL.pth",
]

def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def ensure_models_local(model_dir: Path):
    """
    Ensures the 3 .pth files exist in model_dir.
    Downloads from S3 if missing.
    HARD FAILS if download fails or file still missing.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()

    for fname in MODEL_FILES:
        local_path = model_dir / fname
        if local_path.exists() and local_path.stat().st_size > 0:
            continue

        key = fname  # your objects sit directly under the bucket root
        try:
            # Try download
            s3.download_file(BUCKET_NAME, key, str(local_path))
        except ClientError as e:
            raise RuntimeError(
                f"S3 download failed for bucket='{BUCKET_NAME}' key='{key}' "
                f"region='{AWS_REGION}' -> {e}"
            )

        # Verify post-download
        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError(
                f"Downloaded model is missing/empty: {local_path} "
                f"(bucket='{BUCKET_NAME}', key='{key}', region='{AWS_REGION}')"
            )
