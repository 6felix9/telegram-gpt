"""Simple TTL cache for reducing redundant database queries."""
import time
import logging

logger = logging.getLogger(__name__)

MISSING = object()  # Sentinel for cache misses (distinguishes "not cached" from cached None/False)


class TTLCache:
    """In-memory cache with per-key TTL expiration.

    Uses time.monotonic() so expiration is immune to wall-clock adjustments.
    """

    def __init__(self, default_ttl: float = 60.0):
        self._default_ttl = default_ttl
        self._store: dict[str, tuple[object, float]] = {}  # key -> (value, expires_at)

    def get(self, key: str, default=MISSING):
        """Return cached value if present and not expired.

        Args:
            key: Cache key.
            default: Value to return on miss. If not provided, returns MISSING sentinel.

        Returns:
            Cached value, default, or MISSING sentinel on miss.
        """
        entry = self._store.get(key)
        if entry is not None:
            value, expires_at = entry
            if time.monotonic() < expires_at:
                return value
            # Expired — remove lazily
            del self._store[key]
        return default

    def set(self, key: str, value: object, ttl: float | None = None) -> None:
        """Store a value with optional per-key TTL override."""
        ttl = ttl if ttl is not None else self._default_ttl
        self._store[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: str) -> None:
        """Remove a single key from the cache."""
        self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove all keys that start with *prefix*."""
        keys_to_remove = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._store[k]

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
