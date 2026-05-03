"""
Comprehensive tests for TTSService.
"""

from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock, patch

import pytest
from services.tts_service import TTSService


@pytest.fixture(autouse=True)
def _reset_singleton():
    TTSService._instance = None
    yield
    TTSService._instance = None


class TestTTSService:
    """Tests for TTSService coverage."""

    def test_singleton_pattern(self):
        a = TTSService.get_instance()
        b = TTSService.get_instance()
        assert a is b

    def test_get_available_voices(self):
        svc = TTSService()
        result = svc.get_available_voices()
        assert result["success"] is True
        assert "en" in result["voices"]
        assert result["voices"]["en"]["language_code"] == "en-US"

    def test_init_client_success(self):
        with patch("google.cloud.texttospeech.TextToSpeechClient") as mock_client_cls:
            os.environ["GOOGLE_TTS_API_KEY"] = "fake-key"
            with patch("services.tts_service._TTS_AVAILABLE", True):
                svc = TTSService()
                assert svc._client is not None

    def test_init_client_failure(self):
        with patch("google.cloud.texttospeech.TextToSpeechClient", side_effect=Exception("Init error")):
            os.environ["GOOGLE_TTS_API_KEY"] = "fake-key"
            with patch("services.tts_service._TTS_AVAILABLE", True):
                svc = TTSService()
                assert svc._client is None

    def test_synthesize_empty_text(self):
        svc = TTSService()
        result = svc.synthesize("")
        assert result["success"] is False
        assert "required" in result["error"]

    def test_synthesize_none_text(self):
        svc = TTSService()
        result = svc.synthesize(None)
        assert result["success"] is False
        assert "required" in result["error"]

    def test_synthesize_no_client(self):
        svc = TTSService()
        svc._client = None
        result = svc.synthesize("Hello")
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_synthesize_cache_hit(self):
        svc = TTSService()
        key = svc._cache_key("Hello", "en", 1.0)
        svc._cache[key] = "fakebase64"
        
        result = svc.synthesize("Hello", "en", 1.0)
        assert result["success"] is True
        assert result["audio_base64"] == "fakebase64"
        assert result["cached"] is True

    @patch("google.cloud.texttospeech.TextToSpeechClient")
    def test_synthesize_success(self, mock_client_cls):
        svc = TTSService()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.audio_content = b"audio"
        mock_client.synthesize_speech.return_value = mock_response
        svc._client = mock_client
        
        with patch("google.cloud.texttospeech.VoiceSelectionParams"), \
             patch("google.cloud.texttospeech.SynthesisInput"), \
             patch("google.cloud.texttospeech.AudioConfig"):
            result = svc.synthesize("Hello", "en", 1.2)
            assert result["success"] is True
            assert result["audio_base64"] == base64.b64encode(b"audio").decode("utf-8")
            assert result["voice"] == "en-US-Wavenet-D"

    def test_synthesize_failure(self):
        svc = TTSService()
        mock_client = MagicMock()
        mock_client.synthesize_speech.side_effect = Exception("Synthesis error")
        svc._client = mock_client
        
        result = svc.synthesize("Hello")
        assert result["success"] is False
        assert "Synthesis error" in result["error"]

    def test_truncate_text(self):
        svc = TTSService()
        long_text = "a" * 6000
        truncated = svc._truncate_text(long_text)
        assert len(truncated) <= 5003 
        assert truncated.endswith("...")

    def test_cache_eviction(self):
        svc = TTSService()
        svc._max_cache_size = 2
        svc._put_cache("k1", "v1")
        svc._put_cache("k2", "v2")
        assert len(svc._cache) == 2
        svc._put_cache("k3", "v3")
        assert len(svc._cache) == 2
        assert "k1" not in svc._cache
        assert "k3" in svc._cache

    def test_build_voice_params(self):
        from google.cloud import texttospeech
        svc = TTSService()
        voice_cfg = ("en-US", "en-US-Wavenet-D", "MALE")
        with patch("google.cloud.texttospeech.VoiceSelectionParams") as mock_params:
            svc._build_voice_params(voice_cfg)
            mock_params.assert_called_once()
            args, kwargs = mock_params.call_args
            assert kwargs["ssml_gender"] == texttospeech.SsmlVoiceGender.MALE

    def test_build_voice_params_neutral(self):
        from google.cloud import texttospeech
        svc = TTSService()
        voice_cfg = ("xx-XX", "voice-X", "UNKNOWN")
        with patch("google.cloud.texttospeech.VoiceSelectionParams") as mock_params:
            svc._build_voice_params(voice_cfg)
            args, kwargs = mock_params.call_args
            assert kwargs["ssml_gender"] == texttospeech.SsmlVoiceGender.NEUTRAL

    def test_get_available_voices_structure(self):
        svc = TTSService()
        result = svc.get_available_voices()
        assert "voices" in result
        assert result["success"] is True
        for lang, info in result["voices"].items():
            assert "language_code" in info
            assert "voice_name" in info
            assert "gender" in info

    def test_truncate_text_logic(self):
        svc = TTSService()
        text = "a" * 10
        assert svc._truncate_text(text) == text
        
        long_text = "a" * 6000
        truncated = svc._truncate_text(long_text)
        assert len(truncated) == 5003
        assert truncated.endswith("...")

    def test_check_cache_logic(self):
        svc = TTSService()
        key = svc._cache_key("text", "en", 1.0)
        svc._cache[key] = "audio_data"
        result = svc._check_cache("text", "en", 1.0)
        assert result["audio_base64"] == "audio_data"
        assert result["cached"] is True

    def test_perform_synthesis_error(self):
        svc = TTSService()
        mock_client = MagicMock()
        mock_client.synthesize_speech.side_effect = Exception("TTS error")
        svc._client = mock_client
        result = svc._perform_synthesis("text", "en", 1.0)
        assert result["success"] is False
        assert "TTS error" in result["error"]

    def test_build_voice_params_default_gender(self):
        from google.cloud import texttospeech
        svc = TTSService()
        # Test fallback to NEUTRAL if gender string is invalid
        voice_cfg = ("en-US", "name", "INVALID_GENDER")
        with patch("google.cloud.texttospeech.VoiceSelectionParams") as mock_params:
            svc._build_voice_params(voice_cfg)
            _, kwargs = mock_params.call_args
            assert kwargs["ssml_gender"] == texttospeech.SsmlVoiceGender.NEUTRAL

    def test_put_cache_full(self):
        svc = TTSService()
        svc._max_cache_size = 2
        svc._put_cache("k1", "v1")
        svc._put_cache("k2", "v2")
        svc._put_cache("k3", "v3")
        assert len(svc._cache) == 2
        assert "k1" not in svc._cache

    def test_cache_key_generation(self):
        svc = TTSService()
        key = svc._cache_key("text", "en", 1.0)
        assert isinstance(key, str)
        assert len(key) == 64
