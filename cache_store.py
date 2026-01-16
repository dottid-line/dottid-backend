# cache_store.py
import os
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Union

try:
    import boto3  # optional
except Exception:
    boto3 = None


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return default


class CacheStore:
    """
    CacheStore supports:
      - Local filesystem cache (fast)
      - Optional S3 cache (persistent across instances)
    """

    def __init__(
        self,
        local_dir: Union[str, Path] = "/tmp/dottid-cache",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "dottid-cache",
        aws_region: Optional[str] = None,
    ):
        self.local_dir = Path(str(local_dir))
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.s3_bucket = (s3_bucket or "").strip() or None
        self.s3_prefix = s3_prefix.strip("/")

        self.s3 = None
        if self.s3_bucket and boto3 is not None:
            sess = boto3.session.Session(region_name=aws_region) if aws_region else boto3.session.Session()
            self.s3 = sess.client("s3")

    # ----------------------------
    # Keying
    # ----------------------------
    def make_key(self, namespace: str, identity: dict) -> str:
        # canonicalize identity to avoid cache-miss due to spacing / ordering
        canon = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return f"{namespace}/{_sha1(canon)}"

    def _local_path(self, key: str, ext: str) -> Path:
        safe = key.replace("/", "_")
        return self.local_dir / f"{safe}.{ext}"

    def _s3_key(self, key: str, ext: str) -> str:
        return f"{self.s3_prefix}/{key}.{ext}"

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        """
        Atomic write to avoid partial/corrupt cache files under high concurrency.
        Writes to a temp file in the same directory then replaces.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tf:
                tmp_name = tf.name
                tf.write(data)
                tf.flush()
                try:
                    os.fsync(tf.fileno())
                except Exception:
                    pass
            os.replace(tmp_name, str(path))
        except Exception:
            try:
                # best-effort cleanup of temp file if replace failed
                if "tmp_name" in locals() and tmp_name:
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass
            except Exception:
                pass

    def _atomic_write_text(self, path: Path, text: str, encoding: str = "utf-8") -> None:
        self._atomic_write_bytes(path, text.encode(encoding))

    # ----------------------------
    # JSON
    # ----------------------------
    def get_json(self, key: str, ttl_seconds: int) -> Optional[dict]:
        if ttl_seconds is None:
            return None
        try:
            ttl_seconds = int(ttl_seconds)
        except Exception:
            return None
        if ttl_seconds <= 0:
            return None

        now = time.time()
        p = self._local_path(key, "json")

        # 1) local hit
        if p.exists():
            age = now - p.stat().st_mtime
            if age <= ttl_seconds:
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass

        # 2) S3 hit
        if self.s3 and self.s3_bucket:
            try:
                obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self._s3_key(key, "json"))
                body = obj["Body"].read()
                data = json.loads(body.decode("utf-8"))
                # refresh local (atomic)
                try:
                    self._atomic_write_text(p, json.dumps(data), encoding="utf-8")
                except Exception:
                    pass
                return data
            except Exception:
                return None

        return None

    def put_json(self, key: str, data: dict) -> None:
        p = self._local_path(key, "json")
        payload = json.dumps(data, ensure_ascii=False)

        try:
            self._atomic_write_text(p, payload, encoding="utf-8")
        except Exception:
            pass

        if self.s3 and self.s3_bucket:
            try:
                self.s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=self._s3_key(key, "json"),
                    Body=payload.encode("utf-8"),
                    ContentType="application/json",
                )
            except Exception:
                pass

    # ----------------------------
    # BYTES (images)
    # ----------------------------
    def get_bytes(self, key: str, ttl_seconds: int) -> Optional[bytes]:
        if ttl_seconds is None:
            return None
        try:
            ttl_seconds = int(ttl_seconds)
        except Exception:
            return None
        if ttl_seconds <= 0:
            return None

        now = time.time()
        p = self._local_path(key, "bin")

        # 1) local hit
        if p.exists():
            age = now - p.stat().st_mtime
            if age <= ttl_seconds:
                try:
                    return p.read_bytes()
                except Exception:
                    pass

        # 2) S3 hit
        if self.s3 and self.s3_bucket:
            try:
                obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self._s3_key(key, "bin"))
                body = obj["Body"].read()
                # refresh local (atomic)
                try:
                    self._atomic_write_bytes(p, body)
                except Exception:
                    pass
                return body
            except Exception:
                return None

        return None

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._local_path(key, "bin")

        try:
            self._atomic_write_bytes(p, data)
        except Exception:
            pass

        if self.s3 and self.s3_bucket:
            try:
                self.s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=self._s3_key(key, "bin"),
                    Body=data,
                    ContentType="application/octet-stream",
                )
            except Exception:
                pass


# =============================================================================
# ENV-DRIVEN SINGLETON + SIMPLE HELPERS (so pipeline can just import this module)
# =============================================================================

_CACHE_ENABLED = _env_bool("CACHE_ENABLED", default=False)
_CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/dottid-cache").strip() or "/tmp/dottid-cache"
_CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 43200)  # default 12 hours

# Optional S3 envs (ignored unless provided)
_CACHE_S3_BUCKET = (os.environ.get("CACHE_S3_BUCKET", "") or "").strip() or None
_CACHE_S3_PREFIX = (os.environ.get("CACHE_S3_PREFIX", "dottid-cache") or "dottid-cache").strip()
_CACHE_AWS_REGION = (os.environ.get("AWS_REGION", "") or "").strip() or None

_STORE: Optional[CacheStore] = None


def cache_enabled() -> bool:
    return bool(_CACHE_ENABLED)


def cache_ttl_seconds() -> int:
    return int(_CACHE_TTL_SECONDS)


def get_store() -> CacheStore:
    global _STORE
    if _STORE is None:
        _STORE = CacheStore(
            local_dir=_CACHE_DIR,
            s3_bucket=_CACHE_S3_BUCKET,
            s3_prefix=_CACHE_S3_PREFIX,
            aws_region=_CACHE_AWS_REGION,
        )
    return _STORE


def cache_get_json(namespace: str, identity: dict, ttl_seconds: Optional[int] = None) -> Optional[dict]:
    if not cache_enabled():
        return None
    ttl = cache_ttl_seconds() if ttl_seconds is None else int(ttl_seconds)
    if ttl <= 0:
        return None
    store = get_store()
    key = store.make_key(namespace, identity)
    return store.get_json(key, ttl)


def cache_set_json(namespace: str, identity: dict, data: dict) -> None:
    if not cache_enabled():
        return
    store = get_store()
    key = store.make_key(namespace, identity)
    store.put_json(key, data)


def cache_get_bytes(namespace: str, identity: dict, ttl_seconds: Optional[int] = None) -> Optional[bytes]:
    if not cache_enabled():
        return None
    ttl = cache_ttl_seconds() if ttl_seconds is None else int(ttl_seconds)
    if ttl <= 0:
        return None
    store = get_store()
    key = store.make_key(namespace, identity)
    return store.get_bytes(key, ttl)


def cache_set_bytes(namespace: str, identity: dict, data: bytes) -> None:
    if not cache_enabled():
        return
    store = get_store()
    key = store.make_key(namespace, identity)
    store.put_bytes(key, data)
