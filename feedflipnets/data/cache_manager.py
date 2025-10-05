"""Dataset caching helpers."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

_CACHE_ROOT = Path(os.environ.get("FEEDFLIP_CACHE_DIR", Path.cwd() / "datasets_cache"))
_OFFLINE = os.environ.get("FEEDFLIP_DATA_OFFLINE", "0") == "1"


class CacheError(RuntimeError):
    """Raised when a resource cannot be fetched."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _offline_response(url: str, offline_path: Optional[Path]) -> Tuple[Path, Dict[str, object]]:
    if offline_path is None:
        raise CacheError("Offline mode requested but no fixture provided.")
    if not offline_path.exists():
        raise CacheError(f"Offline fixture missing: {offline_path}")
    provenance = {
        "mode": "offline",
        "url": url,
        "local_path": str(offline_path),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(0)),
        "checksum": _sha256(offline_path),
    }
    return offline_path, provenance


def _ensure_offline_file(name: str, builder: Callable[[Path], None]) -> Path:
    target = _CACHE_ROOT / "offline" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        builder(target)
    return target


def fetch(
    url: str,
    *,
    checksum: Optional[str] = None,
    filename: Optional[str] = None,
    offline_path: Optional[Path] = None,
    offline_builder: Optional[Callable[[Path], None]] = None,
    retries: int = 3,
) -> Tuple[Path, Dict[str, object]]:
    """Fetch ``url`` into the cache and return the local path and provenance."""

    if offline_path is None and offline_builder is not None:
        if filename is None:
            raise ValueError("filename is required when using offline_builder")
        offline_path = _ensure_offline_file(filename, offline_builder)

    if _OFFLINE:
        return _offline_response(url, offline_path)

    if filename is None:
        filename = os.path.basename(url)
    target = _CACHE_ROOT / filename
    if target.exists():
        provenance = {
            "mode": "cache",
            "url": url,
            "local_path": str(target),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(target.stat().st_mtime)),
            "checksum": _sha256(target),
        }
        if checksum and provenance["checksum"] != checksum:
            target.unlink()
        else:
            return target, provenance

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            import urllib.request

            with urllib.request.urlopen(url) as response, target.open("wb") as handle:
                handle.write(response.read())
            break
        except Exception as exc:  # pragma: no cover - best effort
            error = exc
            time.sleep(min(2 ** attempt, 5))
    else:
        if offline_path is not None and offline_path.exists():
            # Fallback to the shipped fixture when downloads are unavailable.
            return _offline_response(url, offline_path)
        raise CacheError(f"Failed to fetch {url!r}: {error}")

    checksum_value = _sha256(target)
    if checksum and checksum_value != checksum:
        target.unlink(missing_ok=True)
        raise CacheError("Checksum mismatch for fetched resource.")

    provenance = {
        "mode": "download",
        "url": url,
        "local_path": str(target),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checksum": checksum_value,
    }
    return target, provenance

