"""Tiny on-disk JSON cache and a rate limiter, used for every API call."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def make_key(url: str, params: dict | None = None) -> str:
    """Cache key from URL + query params. Headers (where API keys live) are NOT part of it."""
    return json.dumps({"url": url, "params": sorted((params or {}).items())}, sort_keys=True)


class DiskCache:
    def __init__(self, root: str | Path, ttl_seconds: float | None = None):
        self.root = Path(root)
        self.ttl = ttl_seconds

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: str):
        """Return cached data, or None on a miss / expired entry."""
        p = self._path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if self.ttl is not None and time.time() - obj["ts"] > self.ttl:
            return None
        return obj["data"]

    def set(self, key: str, data) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ts": time.time(), "data": data}), encoding="utf-8")
        os.replace(tmp, p)


class RateLimiter:
    """Guarantees at least `min_interval` seconds between successive calls to wait()."""

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
