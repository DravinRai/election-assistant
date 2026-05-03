"""
Google Custom Search Service — election news search.

Uses Google Custom Search JSON API to search for election-related
news articles with result caching and fallback mode.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Optional

import requests

from config import (
    ENV_SEARCH_API_KEY,
    ENV_SEARCH_ENGINE_ID,
    ENV_SEARCH_TIMEOUT,
    SEARCH_BASE_URL,
    SEARCH_CACHE_TTL_SECONDS,
    SEARCH_MAX_RESULTS,
    SEARCH_MAX_RESULTS_UPPER,
    SEARCH_TIMEOUT_DEFAULT,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Fetch election news via Google Custom Search JSON API."""

    _instance: SearchService | None = None

    def __init__(self) -> None:
        """Initialise with environment configuration."""
        self._api_key = os.environ.get(ENV_SEARCH_API_KEY, "")
        self._cx = os.environ.get(ENV_SEARCH_ENGINE_ID, "")
        self._base_url = SEARCH_BASE_URL
        self._timeout = int(os.environ.get(ENV_SEARCH_TIMEOUT, str(SEARCH_TIMEOUT_DEFAULT)))
        self._cache: dict[str, dict[str, Any]] = {}

        if self._api_key and self._cx:
            logger.info("SearchService initialised.")
        else:
            logger.info("SearchService in disabled mode.")

    @classmethod
    def get_instance(cls) -> SearchService:
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search_news(self, query: str, num_results: int = SEARCH_MAX_RESULTS) -> dict[str, Any]:
        """Search for election-related news.

        Args:
            query: Search query string.
            num_results: Number of results (1-10).

        Returns:
            Dict with results, query, and success fields.
        """
        if not query or not query.strip():
            return {"error": "Query is required.", "success": False}
        num_results = max(1, min(num_results, SEARCH_MAX_RESULTS_UPPER))
        query = query.strip()
        cache_key = self._cache_key(query, num_results)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return {"results": cached, "query": query, "cached": True, "success": True}
        if not self._api_key or not self._cx:
            return self._fallback_results(query)
        return self._perform_search(query, num_results)

    def _perform_search(self, query: str, num: int) -> dict[str, Any]:
        """Execute the API call."""
        try:
            params = {"key": self._api_key, "cx": self._cx, "q": f"election {query}", "num": num, "sort": "date", "safe": "active"}
            resp = requests.get(self._base_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            results = [self._parse_item(i) for i in data.get("items", [])]
            self._put_cache(self._cache_key(query, num), results)
            return {"results": results, "query": query, "total_results": data.get("searchInformation", {}).get("totalResults", "0"), "success": True}
        except requests.Timeout:
            return self._fallback_results(query, error="Search request timed out.")
        except requests.RequestException as exc:
            return self._fallback_results(query, error=str(exc))

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> dict[str, Any]:
        """Parse a single search result item."""
        thumbnail = ""
        thumbnails = item.get("pagemap", {}).get("cse_thumbnail", [])
        if thumbnails:
            thumbnail = thumbnails[0].get("src", "")
        return {"title": item.get("title", ""), "snippet": item.get("snippet", ""), "url": item.get("link", ""), "display_url": item.get("displayLink", ""), "thumbnail": thumbnail}

    def _get_cached(self, key: str) -> list | None:
        """Get cached results if not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry["timestamp"] > SEARCH_CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        return entry["results"]

    def _put_cache(self, key: str, results: list) -> None:
        """Store results in cache."""
        self._cache[key] = {"results": results, "timestamp": time.time()}

    @staticmethod
    def _cache_key(query: str, num: int) -> str:
        """Create deterministic cache key."""
        return hashlib.sha256(f"{num}::{query.lower().strip()}".encode()).hexdigest()

    @staticmethod
    def _fallback_results(query: str, error: Optional[str] = None) -> dict[str, Any]:
        """Return placeholder results when API is unavailable."""
        return {
            "results": [{"title": f"Search results for: {query}", "snippet": "Google Custom Search is not configured. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables.", "url": f"https://www.google.com/search?q=election+{query.replace(' ', '+')}", "display_url": "google.com", "thumbnail": ""}],
            "query": query, "fallback": True, "error": error, "success": True,
        }
