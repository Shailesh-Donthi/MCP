"""In-memory TTL cache for dynamic query results.

Usage:
    from mcp.core.cache import query_cache

    hit = await query_cache.get("List all SIs")
    if hit is not None:
        return hit  # cached dict

    result = ...  # expensive LLM+DB work
    await query_cache.set("List all SIs", result, ttl=300)

Always active — no Redis or external service required.
Upgrades transparently to Redis when REDIS_URL is configured.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryCache:
    """In-memory TTL cache with optional Redis upgrade.

    Stores results in a dict keyed by normalized query hash.
    Entries expire after ``_DEFAULT_TTL`` seconds.
    Evicts expired entries lazily on access + periodically on set.
    """

    _PREFIX = "mcp:dq:"
    _DEFAULT_TTL = 900        # 15 minutes
    _MAX_ENTRIES = 500        # cap to prevent unbounded growth

    def __init__(self) -> None:
        self._redis: Any = None
        self._mem: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key → (expires_at, value)

    # -- lifecycle -----------------------------------------------------------

    def init(self, redis_client: Any) -> None:
        """Attach a live ``redis.asyncio.Redis`` instance (optional upgrade)."""
        self._redis = redis_client

    def close(self) -> None:
        self._redis = None
        self._mem.clear()

    # -- public API ----------------------------------------------------------

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(query)

        # Try Redis first if available
        if self._redis:
            try:
                import json
                raw = await self._redis.get(key)
                if raw:
                    logger.info("Cache HIT (redis) for: %.60s", query)
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis GET failed, falling back to memory: %s", exc)

        # In-memory lookup
        entry = self._mem.get(key)
        if entry is not None:
            expires_at, value = entry
            if time.monotonic() < expires_at:
                logger.info("Cache HIT (memory) for: %.60s", query)
                return value
            del self._mem[key]  # expired

        return None

    async def set(
        self, query: str, value: Dict[str, Any], ttl: int | None = None
    ) -> None:
        key = self._make_key(query)
        effective_ttl = ttl or self._DEFAULT_TTL

        # Write to Redis if available
        if self._redis:
            try:
                import json
                await self._redis.set(key, json.dumps(value, default=str), ex=effective_ttl)
            except Exception as exc:
                logger.warning("Redis SET failed: %s", exc)

        # Always write to memory
        self._evict_expired()
        self._mem[key] = (time.monotonic() + effective_ttl, value)
        logger.info("Cache SET for: %.60s (ttl=%ds, entries=%d)", query, effective_ttl, len(self._mem))

    # -- internals -----------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove expired entries; also cap total size."""
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._mem.items() if now >= exp]
        for k in expired:
            del self._mem[k]
        # If still over limit, drop oldest entries
        if len(self._mem) >= self._MAX_ENTRIES:
            sorted_keys = sorted(self._mem, key=lambda k: self._mem[k][0])
            for k in sorted_keys[: len(self._mem) - self._MAX_ENTRIES + 1]:
                del self._mem[k]

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.lower().split())

    @classmethod
    def _make_key(cls, query: str) -> str:
        normalized = cls._normalize(query)
        h = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"{cls._PREFIX}{h}"


# Module-level singleton — import and use directly.
query_cache = QueryCache()
