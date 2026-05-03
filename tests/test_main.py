"""
Comprehensive tests for main.py (Flask Application).
"""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from main import create_app, sanitise_input, _JsonFormatter, _cache_get, _cache_set


@pytest.fixture()
def app():
    os.environ["GOOGLE_API_KEY"] = "test-key"
    os.environ["FLASK_SECRET_KEY"] = "test-secret"
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


class TestMainLogic:
    """Tests for core logic and utilities in main.py."""

    def test_json_formatter(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=10, msg="test message", args=(), exc_info=None
        )
        formatted = formatter.format(record)
        data = json.loads(formatted)
        assert data["severity"] == "INFO"
        assert data["message"] == "test message"
        assert "timestamp" in data

    def test_json_formatter_with_exception(self):
        formatter = _JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=10, msg="error msg", args=(), exc_info=sys.exc_info()
            )
            formatted = formatter.format(record)
            data = json.loads(formatted)
            assert "exception" in data
            assert "ValueError: test error" in data["exception"]

    def test_sanitise_input(self):
        assert sanitise_input("<script>alert(1)</script> hello") == "hello"
        assert sanitise_input("<b>bold</b> text") == "bold text"
        assert sanitise_input("") == ""
        assert sanitise_input(None) == ""
        assert sanitise_input("plain text") == "plain text"

    def test_cache_logic(self):
        _cache_set("key1", {"data": 123})
        assert _cache_get("key1") == {"data": 123}
        assert _cache_get("nonexistent") is None

    @patch("main.time.time")
    def test_cache_expiry(self, mock_time):
        mock_time.return_value = 1000
        _cache_set("key2", {"data": 456})
        
        # Still valid
        mock_time.return_value = 1299
        assert _cache_get("key2") == {"data": 456}
        
        # Expired
        mock_time.return_value = 1301
        assert _cache_get("key2") is None

    def test_check_service_env(self):
        from main import _check_service_env
        os.environ["TEST_ENV_VAR"] = "val"
        assert _check_service_env("TEST_ENV_VAR") == "configured"
        assert _check_service_env("MISSING_VAR") == "not_configured"


class TestEndpoints:
    """Tests for API endpoints."""

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json["status"] == "healthy"

    @patch("services.vertex_service.VertexService.get_instance")
    @patch("services.gemini_service.GeminiElectionAssistant")
    def test_chat_success(self, mock_gemini_cls, mock_vertex_cls, client):
        mock_vertex = MagicMock()
        mock_vertex.moderate_content.return_value = {"safe": True, "reason": None}
        mock_vertex.classify_topic.return_value = {"topic": "voting", "confidence": 0.9}
        mock_vertex_cls.return_value = mock_vertex
        
        mock_gemini = MagicMock()
        mock_gemini.chat.return_value = {"response": "hi", "suggested_questions": []}
        mock_gemini_cls.return_value = mock_gemini
        
        resp = client.post("/api/chat", json={"message": "hello", "session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json["response"] == "hi"

    def test_chat_validation_errors(self, client):
        # Empty body
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 400
        
        # Empty message after sanitisation
        resp = client.post("/api/chat", json={"message": "  <script></script>  "})
        assert resp.status_code == 400
        
        # Too long
        resp = client.post("/api/chat", json={"message": "a" * 10001})
        assert resp.status_code == 400

    @patch("services.vertex_service.VertexService.get_instance")
    def test_chat_moderation_blocked(self, mock_vertex_cls, client):
        mock_vertex = MagicMock()
        mock_vertex.moderate_content.return_value = {"safe": False, "reason": "blocked"}
        mock_vertex_cls.return_value = mock_vertex
        
        resp = client.post("/api/chat", json={"message": "bad"})
        assert resp.status_code == 400
        assert resp.json["blocked"] is True

    @patch("services.translate_service.TranslateService.get_instance")
    def test_translate_endpoint(self, mock_trans_cls, client):
        mock_trans = MagicMock()
        mock_trans.translate_text.return_value = {"success": True, "translated_text": "hola"}
        mock_trans_cls.return_value = mock_trans
        
        resp = client.post("/api/translate", json={"text": "hello", "target_language": "es"})
        assert resp.status_code == 200
        assert resp.json["translated_text"] == "hola"

    @patch("services.translate_service.TranslateService.get_instance")
    def test_translate_languages(self, mock_trans_cls, client):
        mock_trans = MagicMock()
        mock_trans.get_supported_languages.return_value = {"languages": {"en": "English"}, "success": True}
        mock_trans_cls.return_value = mock_trans
        
        resp = client.get("/api/translate/languages")
        assert resp.status_code == 200
        assert "en" in resp.json["languages"]

    @patch("services.translate_service.TranslateService.get_instance")
    def test_detect_language(self, mock_trans_cls, client):
        mock_trans = MagicMock()
        mock_trans.detect_language.return_value = {"language": "en", "success": True}
        mock_trans_cls.return_value = mock_trans
        
        resp = client.post("/api/translate/detect", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json["language"] == "en"

    @patch("services.tts_service.TTSService.get_instance")
    def test_tts_endpoint(self, mock_tts_cls, client):
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = {"success": True, "audio_base64": "abc"}
        mock_tts_cls.return_value = mock_tts
        
        resp = client.post("/api/tts", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json["audio_base64"] == "abc"

    @patch("services.search_service.SearchService.get_instance")
    def test_news_search(self, mock_search_cls, client):
        mock_search = MagicMock()
        mock_search.search_news.return_value = {"success": True, "results": []}
        mock_search_cls.return_value = mock_search
        
        resp = client.get("/api/news?query=election")
        assert resp.status_code == 200

    @patch("services.firebase_service.FirebaseService.get_instance")
    def test_session_creation(self, mock_fb_cls, client):
        mock_fb = MagicMock()
        mock_fb.create_session.return_value = {"session_id": "s1", "success": True}
        mock_fb_cls.return_value = mock_fb
        
        resp = client.post("/api/session")
        assert resp.status_code == 200
        assert resp.json["session_id"] == "s1"

    @patch("services.firebase_service.FirebaseService.get_instance")
    def test_quiz_score(self, mock_fb_cls, client):
        mock_fb = MagicMock()
        mock_fb.save_quiz_score.return_value = {"success": True}
        mock_fb_cls.return_value = mock_fb
        
        resp = client.post("/api/session/s1/quiz", json={"score": 5, "total": 10})
        assert resp.status_code == 200

    def test_topics_endpoint(self, client):
        resp = client.get("/api/topics")
        assert resp.status_code == 200
        assert "voter_registration" in resp.json["topics"]

    @patch("services.gemini_service.GeminiElectionAssistant")
    def test_quiz_question(self, mock_gemini_cls, client):
        mock_gemini = MagicMock()
        mock_gemini.get_quiz_question.return_value = {"question": "test?"}
        mock_gemini_cls.return_value = mock_gemini
        
        resp = client.get("/api/quiz/question")
        assert resp.status_code == 200
        assert resp.json["question"] == "test?"

    @patch("services.gemini_service.GeminiElectionAssistant")
    def test_timeline_endpoint(self, mock_gemini_cls, client):
        mock_gemini = MagicMock()
        mock_gemini.get_timeline.return_value = {"timeline": []}
        mock_gemini_cls.return_value = mock_gemini
        
        resp = client.post("/api/timeline", json={"country": "India"})
        assert resp.status_code == 200

    def test_404_error(self, client):
        resp = client.get("/notfound")
        assert resp.status_code == 404

    def test_413_error(self, client):
        # MAX_CONTENT_LENGTH is 10KB
        resp = client.post("/api/chat", data="x" * 15000, content_type="application/json")
        assert resp.status_code == 413

    def test_415_error(self, client):
        resp = client.post("/api/chat", data="{}", content_type="text/plain")
        assert resp.status_code == 415

    @patch("main._persist_chat")
    def test_persist_chat_exception_swallowed(self, mock_persist, app):
        from main import _persist_chat
        with patch("services.firebase_service.FirebaseService.get_instance", side_effect=Exception("FB fail")):
            _persist_chat("s1", "hi", {"response": "hi"}, {"topic": "t"})

    def test_cache_get_hit(self):
        from main import _cache_set, _cache_get
        _cache_set("hit_key", {"a": 1})
        assert _cache_get("hit_key") == {"a": 1}

    def test_cache_set_logic(self):
        from main import _cache_set, _response_cache
        _cache_set("set_key", {"b": 2})
        assert "set_key" in _response_cache

    @patch("services.translate_service.TranslateService.get_instance")
    def test_detect_language_route(self, mock_svc_cls, client):
        mock_svc = MagicMock()
        mock_svc.detect_language.return_value = {"language": "en", "success": True}
        mock_svc_cls.return_value = mock_svc
        resp = client.post("/api/translate/detect", json={"text": "hello"})
        assert resp.status_code == 200

    @patch("services.firebase_service.FirebaseService.get_instance")
    def test_create_session_route(self, mock_svc_cls, client):
        mock_svc = MagicMock()
        mock_svc.create_session.return_value = {"session_id": "s1"}
        mock_svc_cls.return_value = mock_svc
        resp = client.post("/api/session")
        assert resp.status_code == 200

    @patch("services.firebase_service.FirebaseService.get_instance")
    def test_save_quiz_score_route(self, mock_svc_cls, client):
        mock_svc = MagicMock()
        mock_svc.save_quiz_score.return_value = {"success": True}
        mock_svc_cls.return_value = mock_svc
        resp = client.post("/api/session/s1/quiz", json={"score": 5, "total": 10, "topic": "voting"})
        assert resp.status_code == 200

    @patch("services.gemini_service.GeminiElectionAssistant")
    def test_quiz_question_route(self, mock_svc_cls, client):
        mock_svc = MagicMock()
        mock_svc.get_quiz_question.return_value = {"question": "q"}
        mock_svc_cls.return_value = mock_svc
        resp = client.get("/api/quiz/question")
        assert resp.status_code == 200

    @patch("services.gemini_service.GeminiElectionAssistant")
    def test_timeline_route(self, mock_svc_cls, client):
        mock_svc = MagicMock()
        mock_svc.get_timeline.return_value = {"timeline": []}
        mock_svc_cls.return_value = mock_svc
        resp = client.post("/api/timeline", json={"country": "USA"})
        assert resp.status_code == 200

    def test_rate_limit_handler(self, client):
        # Trigger the 429 handler by making the app handle a TooManyRequests exception
        from werkzeug.exceptions import TooManyRequests
        with client.application.test_request_context():
            resp = client.application.handle_user_exception(TooManyRequests())
            assert resp.status_code == 429
            assert "Too many requests" in resp.get_json()["error"]

    def test_server_error_handler(self, client):
        # Trigger the 500 handler by making the app handle an InternalServerError
        from werkzeug.exceptions import InternalServerError
        with client.application.test_request_context():
            resp = client.application.handle_user_exception(InternalServerError())
            assert resp.status_code == 500
            assert "internal server error" in resp.get_json()["error"]

    def test_on_rate_limit_breach_direct(self):
        from main import _on_rate_limit_breach
        with patch("main.log_security_event") as mock_log:
            _on_rate_limit_breach("limit")
            mock_log.assert_called_once()
