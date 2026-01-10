import os
from pathlib import Path

import boto3


def download_models():
    """
    Downloads model files from S3 into ./models if they do not exist.
    Runs on server boot.
    """

    bucket = os.environ.get("S3_BUCKET_NAME", "").strip()
    region = os.environ.get("AWS_REGION", "").strip() or "us-east-1"

    if not bucket:
        raise RuntimeError("Missing env var: S3_BUCKET_NAME")

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "FINAL MVP ROOM TYPE MODEL.pth",
        "FINAL MVP CONDITION MODEL.pth",
        "FINAL MVP VALIDATOR MODEL.pth",
    ]

    s3 = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip(),
    )

    for fname in files:
        dest = model_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue  # already present

        s3.download_file(bucket, fname, str(dest))
