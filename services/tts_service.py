"""
Google Cloud Text-to-Speech Service — audio responses for the Election Assistant.

Uses Google Cloud TTS API to:
    - Convert AI responses to spoken audio
    - Use WaveNet voices for natural-sounding speech
    - Cache generated audio to avoid redundant API calls
    - Support multiple languages matching the Translate service

Author: Ankit Rai
Version: 2.1.0
Usage example:
    from services.tts_service import TTSService
    svc = TTSService.get_instance()
    audio = svc.synthesize("Hello")
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any

from config import ENV_TTS_API_KEY, MAX_TTS_TEXT_LENGTH, TTS_MAX_CACHE_SIZE

logger = logging.getLogger(__name__)

# WaveNet voice map: language code → (language_code, voice_name, ssml_gender)
VOICE_MAP: dict[str, tuple[str, str, str]] = {
    "en": ("en-US", "en-US-Wavenet-D", "MALE"),
    "hi": ("hi-IN", "hi-IN-Wavenet-A", "FEMALE"),
    "ta": ("ta-IN", "ta-IN-Wavenet-A", "FEMALE"),
    "te": ("te-IN", "te-IN-Standard-A", "FEMALE"),
    "bn": ("bn-IN", "bn-IN-Wavenet-A", "FEMALE"),
    "mr": ("mr-IN", "mr-IN-Wavenet-A", "FEMALE"),
    "es": ("es-ES", "es-ES-Wavenet-B", "MALE"),
    "fr": ("fr-FR", "fr-FR-Wavenet-B", "MALE"),
}

# Guard import
try:
    from google.cloud import texttospeech

    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    logger.info("Google Cloud TTS SDK not installed — TTS disabled.")

__all__ = ["TTSService"]


class TTSService:
    """Convert text to speech via Google Cloud Text-to-Speech API.

    Provides audio synthesis with WaveNet voices and an in-memory
    cache to avoid redundant API calls. Falls back gracefully when
    the SDK or API key is unavailable.

    Attributes:
        _client: The TTS API client (or None).
        _cache: In-memory audio cache (hash → base64 audio).
        _max_cache_size: Maximum cache entries before eviction.
    """

    _instance: TTSService | None = None

    def __init__(self) -> None:
        """Initialise the TTSService with optional TTS client."""
        self._client = None
        self._api_key = os.environ.get(ENV_TTS_API_KEY, "")

        if _TTS_AVAILABLE and self._api_key:
            self._init_client()
        else:
            logger.info("TTSService running in disabled mode.")

        self._cache: dict[str, str] = {}
        self._max_cache_size: int = TTS_MAX_CACHE_SIZE

    def _init_client(self) -> None:
        """Attempt to initialise the TTS client.

        Catches all exceptions for graceful degradation.
        """
        try:
            self._client = texttospeech.TextToSpeechClient()
            logger.info("TTSService initialised successfully.")
        except Exception:
            logger.exception("Failed to initialise TTS client.")

    @classmethod
    def get_instance(cls) -> TTSService:
        """Return the singleton TTSService instance.

        Returns:
            The shared TTSService instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaking_rate: float = 1.0,
    ) -> dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: The text to convert to speech (max 5000 characters).
            language: ISO 639-1 language code.
            speaking_rate: Speech speed multiplier (0.25–4.0).

        Returns:
            Dict with 'audio_base64', 'content_type', and 'success' fields.
        """
        if not text or not text.strip():
            return {"error": "Text is required.", "success": False}

        text = self._truncate_text(text)

        cached = self._check_cache(text, language, speaking_rate)
        if cached:
            return cached

        if not self._client:
            return {
                "error": "Text-to-Speech service is not available.",
                "success": False,
            }

        return self._perform_synthesis(text, language, speaking_rate)

    def get_available_voices(self) -> dict[str, Any]:
        """Return the map of supported language voices.

        Returns:
            Dict with 'voices' mapping and 'success' flag.
        """
        voices = {}
        for lang, (code, name, gender) in VOICE_MAP.items():
            voices[lang] = {
                "language_code": code,
                "voice_name": name,
                "gender": gender,
            }
        return {"voices": voices, "success": True}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_text(text: str) -> str:
        """Truncate text to the TTS API character limit.

        Args:
            text: The input text.

        Returns:
            Text truncated to MAX_TTS_TEXT_LENGTH if necessary.
        """
        if len(text) > MAX_TTS_TEXT_LENGTH:
            logger.warning(
                "Text truncated to %d chars for TTS.", MAX_TTS_TEXT_LENGTH
            )
            return text[:MAX_TTS_TEXT_LENGTH] + "..."
        return text

    def _check_cache(
        self, text: str, language: str, rate: float
    ) -> dict[str, Any] | None:
        """Check the audio cache for a hit.

        Args:
            text: The text to synthesize.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            Cached audio response if found, else None.
        """
        cache_key = self._cache_key(text, language, rate)
        if cache_key in self._cache:
            logger.debug("TTS cache hit for key %s", cache_key[:12])
            return {
                "audio_base64": self._cache[cache_key],
                "content_type": "audio/mp3",
                "cached": True,
                "success": True,
            }
        return None

    def _perform_synthesis(
        self, text: str, language: str, rate: float
    ) -> dict[str, Any]:
        """Execute the TTS API call.

        Args:
            text: The text to synthesize.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            Synthesis result with base64 audio or error.
        """
        try:
            voice_cfg = VOICE_MAP.get(language, VOICE_MAP["en"])
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = self._build_voice_params(voice_cfg)
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=rate,
            )

            response = self._client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_b64 = base64.b64encode(response.audio_content).decode(
                "utf-8"
            )

            cache_key = self._cache_key(text, language, rate)
            self._put_cache(cache_key, audio_b64)

            return {
                "audio_base64": audio_b64,
                "content_type": "audio/mp3",
                "voice": voice_cfg[1],
                "success": True,
            }
        except Exception as exc:
            logger.exception("TTS synthesis failed: %s", exc)
            return {"error": str(exc), "success": False}

    @staticmethod
    def _build_voice_params(
        voice_cfg: tuple[str, str, str],
    ) -> Any:
        """Build voice selection parameters from config tuple.

        Args:
            voice_cfg: Tuple of (lang_code, voice_name, gender_str).

        Returns:
            VoiceSelectionParams instance.
        """
        lang_code, voice_name, gender_str = voice_cfg
        gender = getattr(
            texttospeech.SsmlVoiceGender,
            gender_str,
            texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        return texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
            ssml_gender=gender,
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _put_cache(self, key: str, value: str) -> None:
        """Add to cache, evicting the oldest entry if at capacity.

        Args:
            key: Cache key.
            value: Base64-encoded audio data.
        """
        if len(self._cache) >= self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            logger.debug("Evicted oldest TTS cache entry.")
        self._cache[key] = value

    @staticmethod
    def _cache_key(text: str, language: str, rate: float) -> str:
        """Create a deterministic cache key.

        Args:
            text: The source text.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            SHA-256 hex digest cache key.
        """
        raw = f"{language}::{rate}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()
