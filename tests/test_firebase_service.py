"""
Comprehensive tests for FirebaseService.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from services.firebase_service import FirebaseService


@pytest.fixture(autouse=True)
def _reset_singleton():
    FirebaseService._instance = None
    yield
    FirebaseService._instance = None


class TestFirebaseService:
    """Tests for FirebaseService coverage."""

    def test_singleton_pattern(self):
        a = FirebaseService.get_instance()
        b = FirebaseService.get_instance()
        assert a is b

    def test_init_firebase_success_no_creds(self):
        with patch("firebase_admin.initialize_app") as mock_init:
            with patch("firebase_admin.get_app", side_effect=ValueError):
                with patch("firebase_admin.firestore.client") as mock_client:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
                    svc = FirebaseService()
                    assert svc.is_available is True
                    mock_init.assert_called_once()

    def test_init_firebase_success_with_creds(self):
        with patch("firebase_admin.initialize_app"):
            with patch("firebase_admin.get_app", side_effect=ValueError):
                with patch("firebase_admin.credentials.Certificate") as mock_cert:
                    with patch("os.path.exists", return_value=True):
                        os.environ["FIREBASE_CREDENTIALS_PATH"] = "path/to/json"
                        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
                        svc = FirebaseService()
                        mock_cert.assert_called_with("path/to/json")

    def test_init_firebase_already_initialized(self):
        with patch("firebase_admin.get_app") as mock_get_app:
            with patch("firebase_admin.firestore.client"):
                svc = FirebaseService()
                mock_get_app.assert_called_once()

    def test_init_firebase_failure(self):
        with patch("firebase_admin.get_app", side_effect=Exception("Init error")):
            svc = FirebaseService()
            assert svc.is_available is False

    def test_create_session_success(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        svc._db = mock_db
        
        result = svc.create_session()
        assert result["success"] is True
        assert result["persisted"] is True
        mock_db.collection.assert_called_with("sessions")

    def test_create_session_failure(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_db.collection.side_effect = Exception("DB error")
        svc._db = mock_db
        
        result = svc.create_session()
        assert result["success"] is True # still returns ID
        assert result["persisted"] is False

    def test_save_message_success(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        svc._db = mock_db
        
        # Mock firestore.Increment
        with patch("firebase_admin.firestore.Increment", return_value="inc"):
            result = svc.save_message("sess-1", "user", "hello", {"meta": "data"})
            assert result["success"] is True
            assert result["persisted"] is True
            mock_db.collection.assert_called()

    def test_save_message_failure(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_db.collection.side_effect = Exception("DB error")
        svc._db = mock_db
        
        result = svc.save_message("sess-1", "user", "hello")
        assert result["success"] is False
        assert "DB error" in result["error"]

    def test_get_conversation_history_success(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {"role": "user", "content": "hi"}
        mock_db.collection.return_value.document.return_value.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = [mock_doc]
        svc._db = mock_db
        
        result = svc.get_conversation_history("sess-1")
        assert result["success"] is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "hi"

    def test_get_conversation_history_failure(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_db.collection.side_effect = Exception("Stream error")
        svc._db = mock_db
        
        result = svc.get_conversation_history("sess-1")
        assert result["success"] is False
        assert result["messages"] == []

    def test_save_quiz_score_success(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        svc._db = mock_db
        
        result = svc.save_quiz_score("sess-1", 8, 10, "voting")
        assert result["success"] is True
        assert result["persisted"] is True

    def test_save_quiz_score_failure(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_db.collection.side_effect = Exception("DB error")
        svc._db = mock_db
        
        result = svc.save_quiz_score("sess-1", 8, 10, "voting")
        assert result["success"] is False
        assert "DB error" in result["error"]

    def test_get_quiz_scores_success(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {"score": 8}
        # Deep mock for order_by.limit.stream
        mock_db.collection.return_value.document.return_value.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = [mock_doc]
        svc._db = mock_db
        
        result = svc.get_quiz_scores("sess-1")
        assert result["success"] is True
        assert len(result["scores"]) == 1
        assert result["scores"][0]["score"] == 8

    def test_get_quiz_scores_failure(self):
        svc = FirebaseService()
        mock_db = MagicMock()
        mock_db.collection.side_effect = Exception("DB error")
        svc._db = mock_db
        
        result = svc.get_quiz_scores("sess-1")
        assert result["success"] is False
        assert result["scores"] == []

    def test_is_available(self):
        svc = FirebaseService()
        svc._db = None
        assert svc.is_available is False
        svc._db = MagicMock()
        assert svc.is_available is True
