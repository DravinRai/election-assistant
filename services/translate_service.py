"""
Google Translate Service — multilingual support for the Election Assistant.

Uses Google Cloud Translation API to:
    - Auto-detect input language
    - Translate AI responses to the user's preferred language
    - Support: Hindi, Tamil, Telugu, Bengali, Marathi, English, Spanish, French

Gracefully falls back to passthrough mode when the Translate SDK
is not installed or API key is not configured.

Example:
    >>> from services.translate_service import TranslateService
    >>> svc = TranslateService.get_instance()
    >>> res = svc.translate_text("Hello", "es")
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional

from config import (
    ENV_TRANSLATE_API_KEY, 
    DEFAULT_LANGUAGE,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
)

logger: logging.Logger = logging.getLogger(__name__)

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

    _TRANSLATE_AVAILABLE: bool = True
except ImportError:
    _TRANSLATE_AVAILABLE: bool = False
    logger.info(
        "Google Cloud Translate SDK not installed — using passthrough mode."
    )
    
__all__: list[str] = ["TranslateService", "SUPPORTED_LANGUAGES"]


class TranslateService:
    """Translate text via Google Cloud Translation API with caching.

    Uses an in-memory cache keyed by (text, target_language) to avoid
    redundant API calls. Falls back to passthrough when no client is
    available.

    Attributes:
        _client: The Google Cloud Translate client (or None).
        _api_key: The API key for translation.
        _cache: In-memory translation cache.
        _instance: Singleton instance.
    """

    _instance: TranslateService | None = None
    _client: Any
    _api_key: str
    _cache: dict[str, str]

    def __init__(self) -> None:
        """Initialise the TranslateService with optional API client.
        
        Detailed description:
            Sets up the translation service by reading the API key from 
            environment and initialising the Google Cloud Translate client.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc = TranslateService()
        """
        self._client = None
        self._api_key = os.environ.get(ENV_TRANSLATE_API_KEY, "")

        if _TRANSLATE_AVAILABLE and self._api_key:
            self._init_client()
        else:
            logger.info("TranslateService running in passthrough mode.")

        self._cache = {}

    def _init_client(self) -> None:
        """Attempt to initialise the Google Translate client.

        Detailed description:
            Initialises the Google Cloud Translate client using the API key.
            Catches exceptions for graceful degradation.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc._init_client()
        """
        try:
            self._client = translate.Client(
                target_language=DEFAULT_LANGUAGE,
                credentials=None,
            )
            self._client._connection.API_KEY = self._api_key
            logger.info("TranslateService initialised with API key.")
        except (ValueError, RuntimeError) as exc:
            logger.error("Failed to initialise Translate client: %s", exc)

    @classmethod
    def get_instance(cls) -> TranslateService:
        """Return the singleton TranslateService instance.

        Detailed description:
            Provides a singleton access pattern for the translate service.
            
        Args:
            None
            
        Returns:
            The shared TranslateService instance.
            
        Raises:
            None
            
        Example:
            >>> svc = TranslateService.get_instance()
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> dict[str, Any]:
        """Detect the language of the given text.

        Detailed description:
            Calls the Google Translate detect_language method. Falls back 
            to returning 'en' if not available.
            
        Args:
            text: The text to detect language for.

        Returns:
            Dict with 'language', 'confidence', and 'success' fields.
            
        Raises:
            None
            
        Example:
            >>> lang = svc.detect_language("Bonjour")
        """
        if not self._client:
            return {"language": DEFAULT_LANGUAGE, "confidence": MAX_CONFIDENCE, "success": True}

        try:
            result: Any = self._client.detect_language(text)
            return {
                "language": result.get("language", DEFAULT_LANGUAGE),
                "confidence": float(result.get("confidence", MIN_CONFIDENCE)),
                "success": True,
            }
        except (ValueError, RuntimeError, ConnectionError) as exc:
            logger.error("Language detection failed: %s", exc)
            return {
                "language": DEFAULT_LANGUAGE,
                "confidence": MIN_CONFIDENCE,
                "success": False,
                "error": str(exc),
            }

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Translate text to the target language.

        Detailed description:
            Translates the provided text into the target language. Uses an 
            in-memory cache to avoid redundant calls.
            
        Args:
            text: Source text to translate.
            target_language: ISO 639-1 target language code.
            source_language: Optional source language (auto-detected if omitted).

        Returns:
            Dict with 'translated_text', language info, and 'success' flag.
            
        Raises:
            None
            
        Example:
            >>> result = svc.translate_text("Hello", "es")
        """
        if target_language not in SUPPORTED_LANGUAGES:
            return self._unsupported_language_response(text, target_language)

        cached: dict[str, Any] | None = self._check_cache(text, target_language, source_language)
        if cached:
            return cached

        if not self._client:
            return self._passthrough_response(
                text, source_language, target_language
            )

        return self._perform_translation(
            text, target_language, source_language
        )

    def get_supported_languages(self) -> dict[str, Any]:
        """Return the list of supported languages.

        Detailed description:
            Returns the statically defined mapping of supported languages.
            
        Args:
            None
            
        Returns:
            Dict with 'languages' mapping and 'success' flag.
            
        Raises:
            None
            
        Example:
            >>> langs = svc.get_supported_languages()
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

        Detailed description:
            Constructs an error dictionary when the requested language is
            not supported.
            
        Args:
            text: The original text.
            target: The unsupported language code.

        Returns:
            Error response dictionary.
            
        Raises:
            None
            
        Example:
            >>> res = TranslateService._unsupported_language_response("Hi", "xx")
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

        Detailed description:
            Calculates the SHA-256 hash of the text and checks if it exists 
            in the cache for the specific target language.
            
        Args:
            text: The text to translate.
            target: Target language code.
            source: Source language code.

        Returns:
            Cached translation response if found, else None.
            
        Raises:
            None
            
        Example:
            >>> cached = svc._check_cache("Hi", "es", "en")
        """
        cache_key: str = self._cache_key(text, target)
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

        Detailed description:
            Constructs a successful-looking dictionary without actually 
            translating the text. Used as a fallback mechanism.
            
        Args:
            text: The original text.
            source: Source language code.
            target: Target language code.

        Returns:
            Passthrough response dictionary.
            
        Raises:
            None
            
        Example:
            >>> res = TranslateService._passthrough_response("Hi", "en", "es")
        """
        return {
            "translated_text": text,
            "source_language": source or DEFAULT_LANGUAGE,
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

        Detailed description:
            Communicates with the Google Cloud Translate API, caches the 
            result upon success, and returns the constructed dict.
            
        Args:
            text: The text to translate.
            target: Target language code.
            source: Optional source language code.

        Returns:
            Translation result dictionary.
            
        Raises:
            None
            
        Example:
            >>> res = svc._perform_translation("Hi", "es", "en")
        """
        try:
            kwargs: dict[str, Any] = {
                "values": text,
                "target_language": target,
            }
            if source:
                kwargs["source_language"] = source

            result: Any = self._client.translate(**kwargs)
            translated: str = result.get("translatedText", text)
            detected_src: str = result.get("detectedSourceLanguage", source or DEFAULT_LANGUAGE)

            cache_key: str = self._cache_key(text, target)
            self._cache[cache_key] = translated

            return {
                "translated_text": translated,
                "source_language": detected_src,
                "target_language": target,
                "success": True,
            }
        except (ValueError, RuntimeError, ConnectionError) as exc:
            logger.error("Translation failed: %s", exc)
            return {
                "translated_text": text,
                "error": str(exc),
                "success": False,
            }

    @staticmethod
    def _cache_key(text: str, target: str) -> str:
        """Create a deterministic cache key from text + target language.

        Detailed description:
            Encodes the target language and string into a SHA-256 hash
            to be used as an efficient in-memory dictionary key.
            
        Args:
            text: The source text.
            target: The target language code.

        Returns:
            SHA-256 hex digest cache key.
            
        Raises:
            None
            
        Example:
            >>> key = TranslateService._cache_key("Hi", "es")
        """
        raw: str = f"{target}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()
