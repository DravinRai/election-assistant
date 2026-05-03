"""
Comprehensive tests for TranslateService.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from services.translate_service import TranslateService


@pytest.fixture(autouse=True)
def _reset_singleton():
    TranslateService._instance = None
    yield
    TranslateService._instance = None


class TestTranslateService:
    """Tests for TranslateService coverage."""

    def test_singleton_pattern(self):
        a = TranslateService.get_instance()
        b = TranslateService.get_instance()
        assert a is b

    def test_get_supported_languages(self):
        svc = TranslateService()
        result = svc.get_supported_languages()
        assert result["success"] is True
        assert "en" in result["languages"]
        assert len(result["languages"]) > 0

    def test_init_client_success(self):
        with patch("services.translate_service.translate.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            os.environ["GOOGLE_TRANSLATE_API_KEY"] = "fake-key"
            
            svc = TranslateService()
            assert svc._client is not None
            assert mock_client._connection.API_KEY == "fake-key"

    def test_init_client_failure(self):
        with patch("services.translate_service.translate.Client", side_effect=Exception("Init error")):
            os.environ["GOOGLE_TRANSLATE_API_KEY"] = "fake-key"
            svc = TranslateService()
            assert svc._client is None

    def test_detect_language_success(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.detect_language.return_value = {"language": "hi", "confidence": 0.99}
        svc._client = mock_client
        
        result = svc.detect_language("नमस्ते")
        assert result["success"] is True
        assert result["language"] == "hi"
        assert result["confidence"] == 0.99

    def test_detect_language_failure(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.detect_language.side_effect = Exception("Detection error")
        svc._client = mock_client
        
        result = svc.detect_language("नमस्ते")
        assert result["success"] is False
        assert result["language"] == "en"
        assert "Detection error" in result["error"]

    def test_detect_language_no_client(self):
        svc = TranslateService()
        svc._client = None
        result = svc.detect_language("Hello")
        assert result["success"] is True
        assert result["language"] == "en"

    def test_translate_text_unsupported_language(self):
        svc = TranslateService()
        result = svc.translate_text("Hello", "unsupported")
        assert result["success"] is False
        assert "Unsupported language" in result["error"]

    def test_translate_text_cache_hit(self):
        svc = TranslateService()
        key = svc._cache_key("Hello", "hi")
        svc._cache[key] = "नमस्ते"
        
        result = svc.translate_text("Hello", "hi")
        assert result["success"] is True
        assert result["translated_text"] == "नमस्ते"
        assert result["cached"] is True

    def test_translate_text_no_client_passthrough(self):
        svc = TranslateService()
        svc._client = None
        result = svc.translate_text("Hello", "hi")
        assert result["success"] is True
        assert result["translated_text"] == "Hello"
        assert result["passthrough"] is True

    def test_translate_text_success_with_source(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.translate.return_value = {
            "translatedText": "नमस्ते",
            "detectedSourceLanguage": "en"
        }
        svc._client = mock_client
        
        result = svc.translate_text("Hello", "hi", source_language="en")
        assert result["success"] is True
        assert result["translated_text"] == "नमस्ते"
        assert result["source_language"] == "en"
        mock_client.translate.assert_called_with(values="Hello", target_language="hi", source_language="en")

    def test_translate_text_success_auto_source(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.translate.return_value = {
            "translatedText": "नमस्ते",
            "detectedSourceLanguage": "en"
        }
        svc._client = mock_client
        
        result = svc.translate_text("Hello", "hi")
        assert result["success"] is True
        assert result["translated_text"] == "नमस्ते"
        mock_client.translate.assert_called_with(values="Hello", target_language="hi")

    def test_translate_text_failure(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.translate.side_effect = Exception("Translate error")
        svc._client = mock_client
        
        result = svc.translate_text("Hello", "hi")
        assert result["success"] is False
        assert result["translated_text"] == "Hello"
        assert "Translate error" in result["error"]

    def test_cache_key_deterministic(self):
        svc = TranslateService()
        k1 = svc._cache_key("test", "hi")
        k2 = svc._cache_key("test", "hi")
        assert k1 == k2
        assert k1 != svc._cache_key("test", "fr")

    def test_detect_language_none_input(self):
        svc = TranslateService()
        # If no client, it returns default en
        svc._client = None
        result = svc.detect_language(None)
        assert result["success"] is True
        assert result["language"] == "en"

    def test_translate_text_none_input(self):
        svc = TranslateService()
        svc._client = None
        result = svc.translate_text(None, "hi")
        assert result["success"] is True
        assert result["translated_text"] is None

    def test_translate_text_empty_input(self):
        svc = TranslateService()
        result = svc.translate_text("", "hi")
        assert result["success"] is True
        assert result["translated_text"] == ""

    def test_init_no_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": ""}, clear=True):
            svc = TranslateService()
            assert svc._client is None

    @patch("services.translate_service._TRANSLATE_AVAILABLE", False)
    def test_init_sdk_not_available(self):
        svc = TranslateService()
        assert svc._client is None

    def test_unsupported_language_response_helper(self):
        svc = TranslateService()
        result = svc._unsupported_language_response("text", "xx")
        assert result["success"] is False
        assert "Unsupported language" in result["error"]

    def test_check_cache_hit(self):
        svc = TranslateService()
        svc._cache[svc._cache_key("hello", "hi")] = "नमस्ते"
        result = svc._check_cache("hello", "hi", None)
        assert result["translated_text"] == "नमस्ते"
        assert result["cached"] is True
        assert result["success"] is True

    def test_check_cache_miss(self):
        svc = TranslateService()
        result = svc._check_cache("missing", "hi", None)
        assert result is None

    def test_passthrough_response_helper(self):
        svc = TranslateService()
        result = svc._passthrough_response("hello", "en", "hi")
        assert result["translated_text"] == "hello"
        assert result["passthrough"] is True
        assert result["success"] is True

    def test_perform_translation_success_internal(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.translate.return_value = {
            "translatedText": "translated",
            "detectedSourceLanguage": "en"
        }
        svc._client = mock_client
        result = svc._perform_translation("text", "hi", "en")
        assert result["translated_text"] == "translated"
        assert result["success"] is True

    def test_perform_translation_exception_internal(self):
        svc = TranslateService()
        mock_client = MagicMock()
        mock_client.translate.side_effect = Exception("API error")
        svc._client = mock_client
        result = svc._perform_translation("text", "hi", "en")
        assert result["success"] is False
        assert result["translated_text"] == "text"
        assert "API error" in result["error"]

    def test_cache_key_generation(self):
        svc = TranslateService()
        key = svc._cache_key("text", "hi")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex length
