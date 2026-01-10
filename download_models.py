# download_models.py
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


REQUIRED_MODEL_FILES = [
    "FINAL MVP ROOM TYPE MODEL.pth",
    "FINAL MVP CONDITION MODEL.pth",
    "FINAL MVP VALIDATOR MODEL.pth",
]


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    v = (v or "").strip()
    return v if v else default


def ensure_models_local(target_dir: Path) -> None:
    """
    Ensures the 3 required .pth model files exist locally at target_dir.
    If missing, downloads them from S3.

    Env vars supported:
      - MODEL_S3_BUCKET (required)
      - AWS_REGION (recommended; or AWS_DEFAULT_REGION)
      - MODEL_S3_PREFIX (optional; if models are inside a folder in the bucket)
    """
    if not isinstance(target_dir, Path):
        target_dir = Path(str(target_dir))

    # Always create directory
    target_dir.mkdir(parents=True, exist_ok=True)

    bucket = _env("MODEL_S3_BUCKET", "")
    if not bucket:
        raise RuntimeError("MODEL_S3_BUCKET is not set in environment.")

    region = _env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-2"))
    prefix = _env("MODEL_S3_PREFIX", "")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3", region_name=region)

    # Download missing files
    for fname in REQUIRED_MODEL_FILES:
        local_path = target_dir / fname
        if local_path.exists() and local_path.stat().st_size > 0:
            continue

        key = f"{prefix}{fname}"

        tmp_path = target_dir / (fname + ".tmp")

        try:
            # download_file writes directly to disk
            s3.download_file(bucket, key, str(tmp_path))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            raise RuntimeError(
                f"S3 download failed for bucket='{bucket}' key='{key}' "
                f"region='{region}' error_code='{code}' message='{msg}'"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"S3 download failed for bucket='{bucket}' key='{key}' region='{region}': {e}"
            ) from e

        # Validate download
        if (not tmp_path.exists()) or tmp_path.stat().st_size == 0:
            raise RuntimeError(
                f"Downloaded file is missing/empty: {tmp_path} (bucket='{bucket}' key='{key}')"
            )

        # Atomic-ish rename
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass
        tmp_path.rename(local_path)

    # Final verification
    missing = []
    for fname in REQUIRED_MODEL_FILES:
        p = target_dir / fname
        if not p.exists() or p.stat().st_size == 0:
            missing.append(str(p))

    if missing:
        raise RuntimeError(f"Models still missing after download: {missing}")
