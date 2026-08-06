from __future__ import annotations

import hashlib
import json
from typing import Any

import duckdb


class ResponseCache:
    """Byte-identical response cache persisted in DuckDB."""

    def __init__(self, path: str = ":memory:"):
        """Open a cache; creates the backing table if it does not exist."""
        self._conn = duckdb.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses("
            "key VARCHAR PRIMARY KEY, response VARCHAR, created_at TIMESTAMP)"
        )
        self._hits = 0
        self._misses = 0

    def cache_key(
        self, model_id: str, prompt_hash: str, params: dict[str, Any], seed: int, input_hash: str
    ) -> str:
        """Stable key: SHA-256 over the canonical JSON of the input tuple."""
        canonical = json.dumps(
            (model_id, prompt_hash, params, seed, input_hash),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        """Return the cached response for key, or None on a miss."""
        row = self._conn.execute(
            "SELECT response FROM responses WHERE key = ?", [key]
        ).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        return row[0]

    def put(self, key: str, response: str) -> None:
        """Upsert response under key; duplicate puts are ignored."""
        self._conn.execute(
            "INSERT INTO responses (key, response, created_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT (key) DO NOTHING",
            [key, response],
        )

    def hit_rate(self) -> float:
        """Return hits / (hits + misses); 1.0 when no lookups have occurred."""
        total = self._hits + self._misses
        if total == 0:
            return 1.0
        return self._hits / total

    def count(self) -> dict[str, int]:
        """Return hit, miss, and stored-entry counters."""
        stored = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return {"hits": self._hits, "misses": self._misses, "stored": stored}

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()
