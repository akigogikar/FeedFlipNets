"""Offline-first cache management for dataset assets."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping

DEFAULT_CACHE_DIR = Path(
    os.environ.get("FFN_CACHE_DIR")
    or os.environ.get("FEEDFLIP_CACHE_DIR")
    or Path.home() / ".cache" / "feedflipnets"
)
MANIFEST_NAME = "manifest.json"


class CacheError(RuntimeError):
    """Raised when resources cannot be fetched or validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CacheManifest:
    """Track cached artefacts and provenance information."""

    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    data: MutableMapping[str, Mapping[str, object]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / MANIFEST_NAME
        if self._path.exists():
            try:
                self.data = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        else:
            self.data = {}

    def record(self, name: str, metadata: Mapping[str, object]) -> None:
        snapshot = dict(metadata)
        snapshot.setdefault(
            "recorded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self.data[name] = snapshot
        self._path.write_text(json.dumps(self.data, indent=2, sort_keys=True))

    def get(self, name: str) -> Mapping[str, object] | None:
        return self.data.get(name)


def fetch(
    name: str,
    url: str,
    *,
    checksum: str | None = None,
    mirrors: Iterable[str] | None = None,
    offline_path: Path | None = None,
    offline_builder: Callable[[Path], None] | None = None,
    filename: str | None = None,
    offline: bool | None = None,
    retries: int = 2,
    manifest: CacheManifest | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, Mapping[str, object]]:
    """Fetch ``url`` into the cache, respecting offline requirements."""

    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    manifest = manifest or CacheManifest(cache_dir)
    offline_env = os.environ.get("FFN_DATA_OFFLINE")
    if offline_env is None:
        offline_env = os.environ.get("FEEDFLIP_DATA_OFFLINE")
    offline_mode = offline if offline is not None else str(offline_env or "1") == "1"

    if offline_mode:
        if offline_path is None:
            raise CacheError(
                f"Offline mode requested for {name!r} but no offline_path provided"
            )
        path = _ensure_offline(offline_path, offline_builder)
        record = _make_record(name, url, path, checksum, mode="offline")
        manifest.record(name, record)
        return path, record

    target = cache_dir / (filename or Path(url).name)
    if target.exists():
        record = _make_record(name, url, target, checksum, mode="cache")
        if checksum and record["checksum"] != checksum:
            target.unlink()
        else:
            manifest.record(name, record)
            return target, record

    sources = [url, *(mirrors or [])]
    last_error: Exception | None = None
    for source in sources:
        for attempt in range(retries + 1):
            try:
                path = _download(source, target)
                record = _make_record(name, source, path, checksum, mode="download")
                manifest.record(name, record)
                return path, record
            except Exception as exc:  # pragma: no cover - best effort
                last_error = exc
                time.sleep(min(2**attempt, 5))
        target.unlink(missing_ok=True)

    if offline_path:
        path = _ensure_offline(offline_path, offline_builder)
        record = _make_record(name, url, path, checksum, mode="offline-fallback")
        manifest.record(name, record)
        return path, record

    raise CacheError(f"Failed to fetch {name!r}: {last_error}")


def _download(url: str, target: Path) -> Path:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        handle.write(response.read())
    return target


def _ensure_offline(path: Path, builder: Callable[[Path], None] | None = None) -> Path:
    if builder and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        builder(path)
    if not path.exists():
        raise CacheError(f"Offline fixture missing: {path}")
    return path


def _make_record(
    name: str,
    url: str,
    path: Path,
    checksum: str | None,
    *,
    mode: str,
) -> Mapping[str, object]:
    digest = _sha256(path)
    if checksum and digest != checksum:
        raise CacheError(f"Checksum mismatch for {name!r}: {digest} != {checksum}")
    return {
        "name": name,
        "url": url,
        "local_path": str(path),
        "checksum": digest,
        "mode": mode,
    }
