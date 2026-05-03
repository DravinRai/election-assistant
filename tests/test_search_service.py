"""
Integration tests for Google Custom Search Service.

Run with:
    pytest tests/test_search_service.py -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    from services.search_service import SearchService
    SearchService._instance = None
    yield
    SearchService._instance = None


class TestSearchService:
    """Tests for the SearchService."""

    def test_singleton_pattern(self):
        from services.search_service import SearchService
        a = SearchService.get_instance()
        b = SearchService.get_instance()
        assert a is b

    def test_search_empty_query(self):
        from services.search_service import SearchService
        svc = SearchService()
        result = svc.search_news("")
        assert result["success"] is False
        assert "required" in result.get("error", "").lower()

    def test_search_whitespace_query(self):
        from services.search_service import SearchService
        svc = SearchService()
        result = svc.search_news("   ")
        assert result["success"] is False

    def test_fallback_when_no_api_key(self):
        from services.search_service import SearchService
        svc = SearchService()
        svc._api_key = ""
        svc._cx = ""
        result = svc.search_news("voter registration")
        assert result["success"] is True
        assert result.get("fallback") is True
        assert len(result["results"]) > 0

    def test_num_results_clamped(self):
        from services.search_service import SearchService
        svc = SearchService()
        svc._api_key = ""
        # Should not crash with out-of-range values
        result = svc.search_news("test", num_results=100)
        assert result["success"] is True

    def test_cache_key_deterministic(self):
        from services.search_service import SearchService
        k1 = SearchService._cache_key("test query", 5)
        k2 = SearchService._cache_key("test query", 5)
        k3 = SearchService._cache_key("test query", 3)
        assert k1 == k2
        assert k1 != k3

    def test_cache_hit(self):
        from services.search_service import SearchService
        import time
        svc = SearchService()
        svc._api_key = "key"
        svc._cx = "cx"
        key = svc._cache_key("election news", 5)
        svc._cache[key] = {
            "results": [{"title": "cached"}],
            "timestamp": time.time(),
        }
        result = svc.search_news("election news", 5)
        assert result["success"] is True
        assert result.get("cached") is True
        assert result["results"][0]["title"] == "cached"

    def test_cache_expiry(self):
        from services.search_service import SearchService
        import time
        svc = SearchService()
        key = svc._cache_key("old query", 5)
        svc._cache[key] = {
            "results": [{"title": "old"}],
            "timestamp": time.time() - 600,  # 10 min old, TTL is 5 min
        }
        cached = svc._get_cached(key)
        assert cached is None

    @patch("services.search_service.requests.get")
    def test_successful_api_call(self, mock_get):
        from services.search_service import SearchService
        svc = SearchService()
        svc._api_key = "test-key"
        svc._cx = "test-cx"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Election News 2024",
                    "snippet": "Latest updates on the election.",
                    "link": "https://example.com/news",
                    "displayLink": "example.com",
                }
            ],
            "searchInformation": {"totalResults": "1"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = svc.search_news("2024 election")
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Election News 2024"

    @patch("services.search_service.requests.get")
    def test_timeout_handling(self, mock_get):
        import requests as req
        from services.search_service import SearchService
        svc = SearchService()
        svc._api_key = "test-key"
        svc._cx = "test-cx"
        mock_get.side_effect = req.Timeout("Connection timed out")

        result = svc.search_news("test")
        assert result["success"] is True  # returns fallback
        assert result.get("fallback") is True

    @patch("services.search_service.requests.get")
    def test_request_exception_handling(self, mock_get):
        import requests as req
        from services.search_service import SearchService
        svc = SearchService()
        svc._api_key = "test-key"
        svc._cx = "test-cx"
        mock_get.side_effect = req.ConnectionError("Connection failed")

        result = svc.search_news("test")
        assert result["success"] is True
        assert result.get("fallback") is True

    def test_search_news_none_query(self):
        from services.search_service import SearchService
        svc = SearchService()
        result = svc.search_news(None)
        assert result["success"] is False
        assert "Query is required" in result["error"]

    def test_parse_item_missing_pagemap(self):
        from services.search_service import SearchService
        svc = SearchService()
        item = {"title": "T", "snippet": "S", "link": "L", "displayLink": "D"}
        result = svc._parse_item(item)
        assert result["thumbnail"] == ""

    def test_init_no_config(self):
        with patch.dict(os.environ, {"GOOGLE_SEARCH_API_KEY": "", "GOOGLE_SEARCH_ENGINE_ID": ""}, clear=True):
            from services.search_service import SearchService
            svc = SearchService()
            assert svc._api_key == ""
            assert svc._cx == ""

    def test_parse_item_with_thumbnail(self):
        from services.search_service import SearchService
        svc = SearchService()
        item = {
            "title": "T", "snippet": "S", "link": "L", "displayLink": "D",
            "pagemap": {"cse_thumbnail": [{"src": "https://thumb.jpg"}]}
        }
        result = svc._parse_item(item)
        assert result["thumbnail"] == "https://thumb.jpg"

    def test_get_cached_valid(self):
        from services.search_service import SearchService
        import time
        svc = SearchService()
        key = "test_key"
        svc._cache[key] = {"results": ["item1"], "timestamp": time.time()}
        assert svc._get_cached(key) == ["item1"]

    def test_put_cache_logic(self):
        from services.search_service import SearchService
        svc = SearchService()
        svc._put_cache("new_key", ["itemA"])
        assert "new_key" in svc._cache
        assert svc._cache["new_key"]["results"] == ["itemA"]

    def test_perform_search_timeout(self):
        from services.search_service import SearchService
        import requests
        svc = SearchService()
        svc._api_key = "k"
        svc._cx = "c"
        with patch("services.search_service.requests.get", side_effect=requests.Timeout("Timeout")):
            result = svc._perform_search("query", 5)
            assert result["success"] is True
            assert result["fallback"] is True
            assert "timed out" in result["error"].lower()

    def test_perform_search_request_exception(self):
        from services.search_service import SearchService
        import requests
        svc = SearchService()
        svc._api_key = "k"
        svc._cx = "c"
        with patch("services.search_service.requests.get", side_effect=requests.RequestException("Error")):
            result = svc._perform_search("query", 5)
            assert result["success"] is True
            assert result["fallback"] is True
            assert result["error"] == "Error"
            
    def test_init_log_success(self):
        with patch.dict(os.environ, {"GOOGLE_SEARCH_API_KEY": "k", "GOOGLE_SEARCH_ENGINE_ID": "c"}):
            from services.search_service import SearchService
            with patch("services.search_service.logger.info") as mock_log:
                SearchService()
                mock_log.assert_any_call("SearchService initialised.")
