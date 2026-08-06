from pathlib import Path

from lossbench.cache import ResponseCache

KEY_ARGS = {
    "model_id": "deepseek-r1",
    "prompt_hash": "abc123",
    "params": {"temperature": 0.2, "top_p": 0.9},
    "seed": 42,
    "input_hash": "def456",
}


def test_roundtrip():
    cache = ResponseCache()
    key = cache.cache_key(**KEY_ARGS)
    cache.put(key, "the response text")
    assert cache.get(key) == "the response text"
    cache.close()


def test_miss_returns_none():
    cache = ResponseCache()
    assert cache.get("missing-key") is None
    assert cache.count()["misses"] == 1
    cache.close()


def test_key_deterministic():
    cache = ResponseCache()
    base = cache.cache_key(**KEY_ARGS)
    assert cache.cache_key(**KEY_ARGS) == base
    for field, new_value in [
        ("model_id", "other-model"),
        ("prompt_hash", "xyz789"),
        ("seed", 7),
        ("input_hash", "feedface"),
    ]:
        varied = dict(KEY_ARGS)
        varied[field] = new_value
        assert cache.cache_key(**varied) != base
    varied = dict(KEY_ARGS)
    varied["params"] = {"temperature": 0.9}
    assert cache.cache_key(**varied) != base
    cache.close()


def test_params_order_insensitive():
    cache = ResponseCache()
    a = dict(KEY_ARGS, params={"a": 1, "b": 2})
    b = dict(KEY_ARGS, params={"b": 2, "a": 1})
    assert cache.cache_key(**a) == cache.cache_key(**b)
    cache.close()


def test_hit_rate_after_sequence():
    cache = ResponseCache()
    key = cache.cache_key(**KEY_ARGS)
    cache.put(key, "body")
    assert cache.get(key) == "body"
    assert cache.get("unknown") is None
    assert cache.hit_rate() == 0.5
    cache.close()


def test_persistence(tmp_path: Path):
    path = str(tmp_path / "cache.duckdb")
    cache = ResponseCache(path)
    key = cache.cache_key(**KEY_ARGS)
    cache.put(key, "persisted body")
    assert cache.count()["stored"] == 1
    cache.close()
    reopened = ResponseCache(path)
    assert reopened.get(key) == "persisted body"
    assert reopened.count()["stored"] == 1
    reopened.close()


def test_duplicate_put_does_not_grow():
    cache = ResponseCache()
    key = cache.cache_key(**KEY_ARGS)
    cache.put(key, "v1")
    cache.put(key, "v1")
    assert cache.count()["stored"] == 1
    cache.close()
