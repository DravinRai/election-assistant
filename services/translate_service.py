"""
Google Translate Service — multilingual support for the Election Assistant.

Uses Google Cloud Translation API to:
    - Auto-detect input language
    - Translate AI responses to the user's preferred language
    - Support: Hindi, Tamil, Telugu, Bengali, Marathi, English, Spanish, French

Gracefully falls back to passthrough mode when the Translate SDK
is not installed or API key is not configured.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional

from config import ENV_TRANSLATE_API_KEY

logger = logging.getLogger(__name__)

# Supported language codes and display names
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "bn": "বাংলা (Bengali)",
    "mr": "मराठी (Marathi)",
    "es": "Español (Spanish)",
    "fr": "Français (French)",
}

# Guard import so tests/dev can run without the SDK
try:
    from google.cloud import translate_v2 as translate
    _TRANSLATE_AVAILABLE = True
except ImportError:
    _TRANSLATE_AVAILABLE = False
    logger.info("Google Cloud Translate SDK not installed — using passthrough mode.")


class TranslateService:
    """Translate text via Google Cloud Translation API with caching.

    Uses an in-memory cache keyed by (text, target_language) to avoid
    redundant API calls. Falls back to passthrough when no client is
    available.

    Attributes:
        _client: The Google Cloud Translate client (or None).
        _cache: In-memory translation cache.
    """

    _instance: TranslateService | None = None

    def __init__(self) -> None:
        """Initialise the TranslateService with optional API client."""
        self._client = None
        self._api_key = os.environ.get(ENV_TRANSLATE_API_KEY, "")

        if _TRANSLATE_AVAILABLE and self._api_key:
            self._init_client()
        else:
            logger.info("TranslateService running in passthrough mode.")

        self._cache: dict[str, str] = {}

    def _init_client(self) -> None:
        """Attempt to initialise the Google Translate client.

        Catches all exceptions for graceful degradation.
        """
        try:
            self._client = translate.Client(
                target_language="en",
                credentials=None,
            )
            self._client._connection.API_KEY = self._api_key
            logger.info("TranslateService initialised with API key.")
        except Exception:
            logger.exception("Failed to initialise Translate client.")

    @classmethod
    def get_instance(cls) -> TranslateService:
        """Return the singleton TranslateService instance.

        Returns:
            The shared TranslateService instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> dict[str, Any]:
        """Detect the language of the given text.

        Args:
            text: The text to detect language for.

        Returns:
            Dict with 'language', 'confidence', and 'success' fields.
        """
        if not self._client:
            return {"language": "en", "confidence": 1.0, "success": True}

        try:
            result = self._client.detect_language(text)
            return {
                "language": result.get("language", "en"),
                "confidence": float(result.get("confidence", 0.0)),
                "success": True,
            }
        except Exception as exc:
            logger.exception("Language detection failed: %s", exc)
            return {"language": "en", "confidence": 0.0, "success": False, "error": str(exc)}

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Translate text to the target language.

        Args:
            text: Source text to translate.
            target_language: ISO 639-1 target language code.
            source_language: Optional source language (auto-detected if omitted).

        Returns:
            Dict with 'translated_text', language info, and 'success' flag.
        """
        if target_language not in SUPPORTED_LANGUAGES:
            return self._unsupported_language_response(text, target_language)

        cached = self._check_cache(text, target_language, source_language)
        if cached:
            return cached

        if not self._client:
            return self._passthrough_response(text, source_language, target_language)

        return self._perform_translation(text, target_language, source_language)

    def get_supported_languages(self) -> dict[str, Any]:
        """Return the list of supported languages.

        Returns:
            Dict with 'languages' mapping and 'success' flag.
        """
        return {"languages": SUPPORTED_LANGUAGES, "success": True}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unsupported_language_response(
        text: str, target: str
    ) -> dict[str, Any]:
        """Build response for unsupported target language.

        Args:
            text: The original text.
            target: The unsupported language code.

        Returns:
            Error response dictionary.
        """
        return {
            "translated_text": text,
            "error": f"Unsupported language: {target}",
            "success": False,
        }

    def _check_cache(
        self,
        text: str,
        target: str,
        source: Optional[str],
    ) -> dict[str, Any] | None:
        """Check the translation cache for a hit.

        Args:
            text: The text to translate.
            target: Target language code.
            source: Source language code.

        Returns:
            Cached translation response if found, else None.
        """
        cache_key = self._cache_key(text, target)
        if cache_key in self._cache:
            logger.debug("Translation cache hit for key %s", cache_key[:12])
            return {
                "translated_text": self._cache[cache_key],
                "source_language": source or "auto",
                "target_language": target,
                "cached": True,
                "success": True,
            }
        return None

    @staticmethod
    def _passthrough_response(
        text: str,
        source: Optional[str],
        target: str,
    ) -> dict[str, Any]:
        """Build passthrough response when no client is available.

        Args:
            text: The original text.
            source: Source language code.
            target: Target language code.

        Returns:
            Passthrough response dictionary.
        """
        return {
            "translated_text": text,
            "source_language": source or "en",
            "target_language": target,
            "success": True,
            "passthrough": True,
        }

    def _perform_translation(
        self,
        text: str,
        target: str,
        source: Optional[str],
    ) -> dict[str, Any]:
        """Execute the actual translation API call.

        Args:
            text: The text to translate.
            target: Target language code.
            source: Optional source language code.

        Returns:
            Translation result dictionary.
        """
        try:
            kwargs: dict[str, Any] = {"values": text, "target_language": target}
            if source:
                kwargs["source_language"] = source

            result = self._client.translate(**kwargs)
            translated = result.get("translatedText", text)
            detected_src = result.get("detectedSourceLanguage", source or "en")

            cache_key = self._cache_key(text, target)
            self._cache[cache_key] = translated

            return {
                "translated_text": translated,
                "source_language": detected_src,
                "target_language": target,
                "success": True,
            }
        except Exception as exc:
            logger.exception("Translation failed: %s", exc)
            return {"translated_text": text, "error": str(exc), "success": False}

    @staticmethod
    def _cache_key(text: str, target: str) -> str:
        """Create a deterministic cache key from text + target language.

        Args:
            text: The source text.
            target: The target language code.

        Returns:
            SHA-256 hex digest cache key.
        """
        raw = f"{target}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()
