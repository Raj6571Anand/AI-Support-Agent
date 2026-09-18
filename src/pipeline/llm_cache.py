"""
LLM Response Cache — keyed on prompt hash to avoid re-consuming Groq quota.
Uses a JSON-lines file for simplicity and crash resilience.
"""
import hashlib
import json
import os
import time
import threading
from pathlib import Path

# Default cache location
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_CACHE_FILE = _DEFAULT_CACHE_DIR / "llm_cache.jsonl"

_lock = threading.Lock()
_memory_cache: dict[str, dict] = {}
_cache_loaded = False
_cache_file = _DEFAULT_CACHE_FILE

# Stats
_hits = 0
_misses = 0


def _make_key(model: str, messages: list[dict], temperature: float = 0.0,
              max_tokens: int = 0, extra: str = "") -> str:
    """Create a deterministic cache key from the request parameters."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra,
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_loaded():
    """Load the cache file into memory on first access."""
    global _cache_loaded, _memory_cache
    if _cache_loaded:
        return
    with _lock:
        if _cache_loaded:
            return
        if _cache_file.exists():
            try:
                with open(_cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            _memory_cache[entry["key"]] = entry
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception:
                pass
        _cache_loaded = True


def get(model: str, messages: list[dict], temperature: float = 0.0,
        max_tokens: int = 0, extra: str = "") -> dict | None:
    """Look up a cached response. Returns the cached response dict or None."""
    global _hits, _misses
    _ensure_loaded()
    key = _make_key(model, messages, temperature, max_tokens, extra)
    entry = _memory_cache.get(key)
    if entry is not None:
        _hits += 1
        return entry.get("response")
    _misses += 1
    return None


def put(model: str, messages: list[dict], temperature: float, max_tokens: int,
        response: dict, extra: str = ""):
    """Store a response in the cache."""
    _ensure_loaded()
    key = _make_key(model, messages, temperature, max_tokens, extra)
    entry = {
        "key": key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
        "response": response,
    }
    with _lock:
        _memory_cache[key] = entry
        # Append to file
        _cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def cached_completion(client, model: str, messages: list[dict],
                      temperature: float = 0.1, max_tokens: int = 200,
                      response_format: dict | None = None,
                      extra: str = "", use_cache: bool = True) -> object:
    """
    Drop-in wrapper around client.chat.completions.create() with caching.
    Returns the full response object-like dict, or a SimpleNamespace that
    mimics the Groq response structure.
    """
    if use_cache:
        cached = get(model, messages, temperature, max_tokens, extra)
        if cached is not None:
            return _make_response_obj(cached)

    # Actually call the API
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)

    # Extract the content for caching
    content = response.choices[0].message.content
    cached_data = {"content": content}

    put(model, messages, temperature, max_tokens, cached_data, extra)

    return response


class _SimpleMessage:
    def __init__(self, content):
        self.content = content

class _SimpleChoice:
    def __init__(self, content):
        self.message = _SimpleMessage(content)

class _SimpleResponse:
    def __init__(self, content):
        self.choices = [_SimpleChoice(content)]


def _make_response_obj(cached: dict):
    """Create a response-like object from cached data."""
    return _SimpleResponse(cached.get("content", ""))


def cache_stats() -> dict:
    """Return cache statistics."""
    _ensure_loaded()
    return {
        "hits": _hits,
        "misses": _misses,
        "total_cached": len(_memory_cache),
        "hit_rate": _hits / max(1, _hits + _misses),
        "cache_file": str(_cache_file),
    }


def clear_cache():
    """Clear the entire cache."""
    global _memory_cache, _cache_loaded
    with _lock:
        _memory_cache = {}
        if _cache_file.exists():
            _cache_file.unlink()
        _cache_loaded = True


if __name__ == "__main__":
    print("LLM Cache module")
    print(f"Cache file: {_cache_file}")
    _ensure_loaded()
    print(f"Cached entries: {len(_memory_cache)}")
    print(f"Stats: {cache_stats()}")
