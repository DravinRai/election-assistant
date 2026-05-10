"""
Google Cloud Text-to-Speech Service — audio responses for the Election Assistant.

Uses Google Cloud TTS API to:
    - Convert AI responses to spoken audio
    - Use WaveNet voices for natural-sounding speech
    - Cache generated audio to avoid redundant API calls
    - Support multiple languages matching the Translate service

Example:
    >>> from services.tts_service import TTSService
    >>> svc = TTSService.get_instance()
    >>> audio = svc.synthesize("Hello")
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any, Tuple

from config import (
    ENV_TTS_API_KEY, 
    MAX_TTS_TEXT_LENGTH, 
    TTS_MAX_CACHE_SIZE,
    DEFAULT_LANGUAGE,
)

logger: logging.Logger = logging.getLogger(__name__)

# WaveNet voice map: language code → (language_code, voice_name, ssml_gender)
VOICE_MAP: dict[str, Tuple[str, str, str]] = {
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

    _TTS_AVAILABLE: bool = True
except ImportError:
    _TTS_AVAILABLE: bool = False
    logger.info("Google Cloud TTS SDK not installed — TTS disabled.")

__all__: list[str] = ["TTSService", "VOICE_MAP"]


class TTSService:
    """Convert text to speech via Google Cloud Text-to-Speech API.

    Provides audio synthesis with WaveNet voices and an in-memory
    cache to avoid redundant API calls. Falls back gracefully when
    the SDK or API key is unavailable.

    Attributes:
        _client: The TTS API client (or None).
        _api_key: The API key for TTS.
        _cache: In-memory audio cache (hash → base64 audio).
        _max_cache_size: Maximum cache entries before eviction.
        _instance: Singleton instance.
    """

    _instance: TTSService | None = None
    _client: Any
    _api_key: str
    _cache: dict[str, str]
    _max_cache_size: int

    def __init__(self) -> None:
        """Initialise the TTSService with optional TTS client.
        
        Detailed description:
            Reads API keys from the environment variables, attempts to init
            the texttospeech client, and sets up the internal cache.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc = TTSService()
        """
        self._client = None
        self._api_key = os.environ.get(ENV_TTS_API_KEY, "")

        if _TTS_AVAILABLE and self._api_key:
            self._init_client()
        else:
            logger.info("TTSService running in disabled mode.")

        self._cache = {}
        self._max_cache_size = TTS_MAX_CACHE_SIZE

    def _init_client(self) -> None:
        """Attempt to initialise the TTS client.

        Detailed description:
            Initialises the Google Cloud TTS client object. Catches exceptions
            gracefully for fallback modes.
            
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
            self._client = texttospeech.TextToSpeechClient()
            logger.info("TTSService initialised successfully.")
        except (ValueError, RuntimeError) as exc:
            logger.error("Failed to initialise TTS client: %s", exc)

    @classmethod
    def get_instance(cls) -> TTSService:
        """Return the singleton TTSService instance.

        Detailed description:
            Provides a singleton access pattern for the TTS service.
            
        Args:
            None
            
        Returns:
            The shared TTSService instance.
            
        Raises:
            None
            
        Example:
            >>> svc = TTSService.get_instance()
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
        language: str = DEFAULT_LANGUAGE,
        speaking_rate: float = 1.0,
    ) -> dict[str, Any]:
        """Synthesize speech from text.

        Detailed description:
            Generates MP3 audio for the given text. Handles truncation, 
            cache checks, and API calls.
            
        Args:
            text: The text to convert to speech (max 5000 characters).
            language: ISO 639-1 language code.
            speaking_rate: Speech speed multiplier (0.25–4.0).

        Returns:
            Dict with 'audio_base64', 'content_type', and 'success' fields.
            
        Raises:
            None
            
        Example:
            >>> audio = svc.synthesize("Hello world")
        """
        if not text or not text.strip():
            return {"error": "Text is required.", "success": False}

        text = self._truncate_text(text)

        cached: dict[str, Any] | None = self._check_cache(text, language, speaking_rate)
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

        Detailed description:
            Transforms the statically defined VOICE_MAP into a JSON-serialisable
            dictionary indicating available voices.
            
        Args:
            None
            
        Returns:
            Dict with 'voices' mapping and 'success' flag.
            
        Raises:
            None
            
        Example:
            >>> voices = svc.get_available_voices()
        """
        voices: dict[str, Any] = {}
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

        Detailed description:
            Limits the input string length to avoid breaching API payload 
            constraints.
            
        Args:
            text: The input text.

        Returns:
            Text truncated to MAX_TTS_TEXT_LENGTH if necessary.
            
        Raises:
            None
            
        Example:
            >>> text = TTSService._truncate_text("A" * 6000)
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

        Detailed description:
            Calculates the key hash and retrieves base64 audio data if present.
            
        Args:
            text: The text to synthesize.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            Cached audio response if found, else None.
            
        Raises:
            None
            
        Example:
            >>> cached = svc._check_cache("Hi", "en", 1.0)
        """
        cache_key: str = self._cache_key(text, language, rate)
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

        Detailed description:
            Configures the VoiceSelectionParams, executes synthesize_speech, 
            encodes the resulting binary audio to base64, and caches it.
            
        Args:
            text: The text to synthesize.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            Synthesis result with base64 audio or error.
            
        Raises:
            None
            
        Example:
            >>> res = svc._perform_synthesis("Hi", "en", 1.0)
        """
        try:
            voice_cfg: Tuple[str, str, str] = VOICE_MAP.get(language, VOICE_MAP["en"])
            synthesis_input: Any = texttospeech.SynthesisInput(text=text)
            voice: Any = self._build_voice_params(voice_cfg)
            audio_config: Any = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=rate,
            )

            response: Any = self._client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            audio_b64: str = base64.b64encode(response.audio_content).decode(
                "utf-8"
            )

            cache_key: str = self._cache_key(text, language, rate)
            self._put_cache(cache_key, audio_b64)

            return {
                "audio_base64": audio_b64,
                "content_type": "audio/mp3",
                "voice": voice_cfg[1],
                "success": True,
            }
        except (ValueError, RuntimeError, ConnectionError) as exc:
            logger.error("TTS synthesis failed: %s", exc)
            return {"error": str(exc), "success": False}

    @staticmethod
    def _build_voice_params(
        voice_cfg: Tuple[str, str, str],
    ) -> Any:
        """Build voice selection parameters from config tuple.

        Detailed description:
            Converts the internal tuple config for a voice into the 
            API-required VoiceSelectionParams.
            
        Args:
            voice_cfg: Tuple of (lang_code, voice_name, gender_str).

        Returns:
            VoiceSelectionParams instance.
            
        Raises:
            None
            
        Example:
            >>> params = TTSService._build_voice_params(VOICE_MAP['en'])
        """
        lang_code, voice_name, gender_str = voice_cfg
        gender: Any = getattr(
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

        Detailed description:
            Limits memory growth by enforcing the max cache size for audio clips.
            
        Args:
            key: Cache key.
            value: Base64-encoded audio data.
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc._put_cache("key123", "base64audio")
        """
        if len(self._cache) >= self._max_cache_size:
            oldest: str = next(iter(self._cache))
            del self._cache[oldest]
            logger.debug("Evicted oldest TTS cache entry.")
        self._cache[key] = value

    @staticmethod
    def _cache_key(text: str, language: str, rate: float) -> str:
        """Create a deterministic cache key.

        Detailed description:
            Constructs a unique SHA-256 hash based on language, rate, and text.
            
        Args:
            text: The source text.
            language: Target language code.
            rate: Speaking rate.

        Returns:
            SHA-256 hex digest cache key.
            
        Raises:
            None
            
        Example:
            >>> key = TTSService._cache_key("Hi", "en", 1.0)
        """
        raw: str = f"{language}::{rate}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()
