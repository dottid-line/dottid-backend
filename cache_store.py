# cache_store.py
"""
Simple TTL cache store for Render-like environments.

- Stores JSON blobs on local disk (default: /tmp/dottid_cache)
- TTL enforced per entry (default caller TTL = 12 hours)
- Safe atomic-ish writes (write temp -> replace)
- Key is hashed to a filename (no filesystem issues)

Usage:
    from cache_store import cache_get, cache_set

    v = cache_get("some-key", ttl_seconds=43200)
    if v is None:
        v = {"hello": "world"}
        cache_set("some-key", v)
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Optional


# Default cache directory (Render writable)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/tmp/dottid_cache")).resolve()


def _ensure_dir() -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't create the directory, caching is effectively disabled.
        pass


def _hash_key(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()
    return h


def _path_for_key(key: str) -> Path:
    # file name includes a short prefix for easier debugging
    h = _hash_key(key)
    return CACHE_DIR / f"cache_{h}.json"


def _now() -> float:
    return time.time()


def cache_get(key: str, ttl_seconds: int = 43200) -> Optional[dict]:
    """
    Return cached value dict if present and not expired, else None.
    ttl_seconds is evaluated against the stored 'created_at' timestamp.
    """
    if not isinstance(key, str) or not key.strip():
        return None

    _ensure_dir()
    p = _path_for_key(key)

    try:
        if not p.exists():
            return None

        raw = p.read_text(encoding="utf-8")
        obj = json.loads(raw)

        if not isinstance(obj, dict):
            return None

        created_at = obj.get("created_at")
        if created_at is None:
            return None

        try:
            created_at_f = float(created_at)
        except Exception:
            return None

        age = _now() - created_at_f
        if age < 0:
            # clock weirdness; treat as miss
            return None

        if ttl_seconds is not None:
            try:
                ttl_f = float(ttl_seconds)
            except Exception:
                ttl_f = 43200.0
            if age > ttl_f:
                # expired: best-effort delete
                try:
                    p.unlink(missing_ok=True)  # py3.8+: missing_ok may not exist; handle below
                except TypeError:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                except Exception:
                    pass
                return None

        val = obj.get("value")
        if isinstance(val, dict):
            return val
        # allow returning any JSON-serializable root, but keep API consistent
        return {"_value": val}

    except Exception:
        return None


def cache_set(key: str, value: Any, ttl_seconds: int = 43200) -> bool:
    """
    Store value with current timestamp.
    Returns True if write succeeded, else False.
    """
    if not isinstance(key, str) or not key.strip():
        return False

    _ensure_dir()
    p = _path_for_key(key)

    payload = {
        "created_at": _now(),
        "ttl_seconds": int(ttl_seconds) if isinstance(ttl_seconds, int) else 43200,
        "value": value,
    }

    try:
        # Write to a temp file in same dir, then replace
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp_cache_", suffix=".json", dir=str(CACHE_DIR))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, str(p))
        finally:
            # If replace failed, tmp may still exist
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        return True
    except Exception:
        return False


def cache_delete(key: str) -> bool:
    """Best-effort delete of a cache entry."""
    if not isinstance(key, str) or not key.strip():
        return False
    _ensure_dir()
    p = _path_for_key(key)
    try:
        p.unlink()
        return True
    except Exception:
        return False


def cache_info() -> dict:
    """Basic info for debugging."""
    _ensure_dir()
    try:
        files = list(CACHE_DIR.glob("cache_*.json"))
        return {
            "cache_dir": str(CACHE_DIR),
            "entries": len(files),
        }
    except Exception:
        return {
            "cache_dir": str(CACHE_DIR),
            "entries": None,
        }
