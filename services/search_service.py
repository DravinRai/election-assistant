"""
Google Custom Search Service — election news search.

Uses Google Custom Search JSON API to search for election-related
news articles with result caching and fallback mode.

Example:
    >>> from services.search_service import SearchService
    >>> svc = SearchService.get_instance()
    >>> news = svc.search_news("election 2026")
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
    MIN_SEARCH_RESULTS,
)

logger: logging.Logger = logging.getLogger(__name__)

__all__: list[str] = ["SearchService"]


class SearchService:
    """Fetch election news via Google Custom Search JSON API.
    
    Provides search capabilities for current election news, caching
    recent queries and providing hardcoded fallback results when the
    API is unavailable.
    
    Attributes:
        _instance: Singleton instance of SearchService.
        _api_key: The API key for custom search.
        _cx: The Custom Search Engine ID.
        _base_url: The base URL for the API.
        _timeout: Request timeout in seconds.
        _cache: In-memory cache of recent search results.
    """

    _instance: SearchService | None = None
    _api_key: str
    _cx: str
    _base_url: str
    _timeout: int
    _cache: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        """Initialise with environment configuration.
        
        Detailed description:
            Reads the API keys and settings from the environment, setting
            up the internal cache and configuring the timeout.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc = SearchService()
        """
        self._api_key = os.environ.get(ENV_SEARCH_API_KEY, "")
        self._cx = os.environ.get(ENV_SEARCH_ENGINE_ID, "")
        self._base_url = SEARCH_BASE_URL
        self._timeout = int(
            os.environ.get(ENV_SEARCH_TIMEOUT, str(SEARCH_TIMEOUT_DEFAULT))
        )
        self._cache = {}

        if self._api_key and self._cx:
            logger.info("SearchService initialised.")
        else:
            logger.info("SearchService in disabled mode.")

    @classmethod
    def get_instance(cls) -> SearchService:
        """Return the singleton instance.

        Detailed description:
            Provides singleton access to the SearchService to share the cache 
            and configuration.
            
        Args:
            None
            
        Returns:
            The singleton SearchService instance.
            
        Raises:
            None
            
        Example:
            >>> svc = SearchService.get_instance()
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search_news(
        self, query: str, num_results: int = SEARCH_MAX_RESULTS
    ) -> dict[str, Any]:
        """Search for election-related news.

        Detailed description:
            Searches for news articles, sanitising inputs and clamping the
            number of results requested. Checks the cache first, falling
            back to API or hardcoded results.

        Args:
            query: Search query string.
            num_results: Number of results (1-10).

        Returns:
            Dict with results, query, and success fields.
            
        Raises:
            None
            
        Example:
            >>> results = svc.search_news("debates")
        """
        if not query or not query.strip():
            return {"error": "Query is required.", "success": False}
        num_results = max(MIN_SEARCH_RESULTS, min(num_results, SEARCH_MAX_RESULTS_UPPER))
        query = query.strip()
        cache_key: str = self._cache_key(query, num_results)
        cached: list[Any] | None = self._get_cached(cache_key)
        if cached is not None:
            return {
                "results": cached,
                "query": query,
                "cached": True,
                "success": True,
            }
        if not self._api_key or not self._cx:
            return self._fallback_results(query)
        return self._perform_search(query, num_results)

    def _perform_search(self, query: str, num: int) -> dict[str, Any]:
        """Execute the API call.
        
        Detailed description:
            Makes the HTTP GET request to the Custom Search API, parses
            the results, and caches them upon success.
            
        Args:
            query: Search query string.
            num: Number of results to fetch.
            
        Returns:
            Dict with search results or fallback on error.
            
        Raises:
            None
            
        Example:
            >>> res = svc._perform_search("debates", 5)
        """
        try:
            params: dict[str, Any] = {
                "key": self._api_key,
                "cx": self._cx,
                "q": f"election {query}",
                "num": num,
                "sort": "date",
                "safe": "active",
            }
            resp: requests.Response = requests.get(
                self._base_url, params=params, timeout=self._timeout
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            results: list[dict[str, Any]] = [self._parse_item(i) for i in data.get("items", [])]
            self._put_cache(self._cache_key(query, num), results)
            return {
                "results": results,
                "query": query,
                "total_results": data.get("searchInformation", {}).get(
                    "totalResults", "0"
                ),
                "success": True,
            }
        except requests.Timeout:
            return self._fallback_results(
                query, error="Search request timed out."
            )
        except requests.RequestException as exc:
            return self._fallback_results(query, error=str(exc))

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> dict[str, Any]:
        """Parse a single search result item.
        
        Detailed description:
            Extracts relevant fields from the raw API item dict, safely handling
            missing nested structures.
            
        Args:
            item: Raw API item dict.
            
        Returns:
            Dict of parsed values.
            
        Raises:
            None
            
        Example:
            >>> parsed = SearchService._parse_item({'title': 'News'})
        """
        thumbnail: str = ""
        thumbnails: list[dict[str, Any]] = item.get("pagemap", {}).get("cse_thumbnail", [])
        if thumbnails:
            thumbnail = thumbnails[0].get("src", "")
        return {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "display_url": item.get("displayLink", ""),
            "thumbnail": thumbnail,
        }

    def _get_cached(self, key: str) -> list[Any] | None:
        """Get cached results if not expired.
        
        Detailed description:
            Retrieves results from cache and removes them if they have expired
            past SEARCH_CACHE_TTL_SECONDS.
            
        Args:
            key: Cache key string.
            
        Returns:
            List of results or None.
            
        Raises:
            None
            
        Example:
            >>> cached = svc._get_cached("key123")
        """
        entry: dict[str, Any] | None = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry["timestamp"] > SEARCH_CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        return entry["results"]

    def _put_cache(self, key: str, results: list[Any]) -> None:
        """Store results in cache.
        
        Detailed description:
            Saves results with the current timestamp into the dictionary cache.
            
        Args:
            key: Cache key string.
            results: Results to cache.
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc._put_cache("key123", [])
        """
        self._cache[key] = {"results": results, "timestamp": time.time()}

    @staticmethod
    def _cache_key(query: str, num: int) -> str:
        """Create deterministic cache key.
        
        Detailed description:
            Hashes the lowercased query and requested count.
            
        Args:
            query: Search query string.
            num: Number of results.
            
        Returns:
            SHA-256 hex string.
            
        Raises:
            None
            
        Example:
            >>> key = SearchService._cache_key("debates", 5)
        """
        return hashlib.sha256(
            f"{num}::{query.lower().strip()}".encode()
        ).hexdigest()

    @staticmethod
    def _fallback_results(
        query: str, error: Optional[str] = None
    ) -> dict[str, Any]:
        """Return placeholder results when API is unavailable.
        
        Detailed description:
            Provides generic hardcoded results so the application can remain
            functional when the search API fails.
            
        Args:
            query: Original search query string.
            error: Optional error message.
            
        Returns:
            Dict containing fallback search results.
            
        Raises:
            None
            
        Example:
            >>> res = SearchService._fallback_results("debates")
        """
        hardcoded_news: list[dict[str, str]] = [
            {
                "title": f"Live Updates: Unprecedented Turnout in Recent {query.title()} Elections",
                "snippet": "Voters are heading to the polls in record numbers. Analysts predict this election will be historic in its voter participation.",
                "url": f"https://www.google.com/search?q=election+{query.replace(' ', '+')}",
                "display_url": "news.example.com",
                "thumbnail": "",
            },
            {
                "title": f"Key Issues Taking Center Stage in the {query.title()} Campaign Trail",
                "snippet": "Economy, healthcare, and climate change are among the top issues motivating voters in this election cycle.",
                "url": f"https://www.google.com/search?q=election+{query.replace(' ', '+')}",
                "display_url": "politics.example.com",
                "thumbnail": "",
            },
            {
                "title": "Understanding the Electoral Process: A Guide for First-Time Voters",
                "snippet": "An in-depth look at how votes are counted and the steps taken to ensure election integrity and security.",
                "url": f"https://www.google.com/search?q=election+{query.replace(' ', '+')}",
                "display_url": "civic-education.example.com",
                "thumbnail": "",
            },
        ]
        if error:
            logger.error(f"Search API Error: {error}", exc_info=True)
        return {
            "results": hardcoded_news,
            "query": query,
            "fallback": True,
            "error": error,
            "success": True,
        }
