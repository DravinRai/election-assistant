"""
Unit tests for the GeminiElectionAssistant service.

Tests cover:
    - Initialisation with and without API keys
    - Chat, timeline, quiz, and term explanation methods
    - History trimming
    - JSON parsing error handling
    - Retry decorator behaviour

Run with:
    pytest tests/test_gemini_service.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from services.gemini_service import (
    GeminiElectionAssistant,
    retry_with_exponential_backoff,
)


@pytest.fixture
def mock_genai():
    """Mock the google.generativeai module.

    Yields:
        Mocked genai module.
    """
    with patch("services.gemini_service.genai") as mock:
        yield mock


@pytest.fixture
def service(mock_genai, monkeypatch):
    """Provide a GeminiElectionAssistant with mocked API key.

    Args:
        mock_genai: Mocked genai module fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        GeminiElectionAssistant instance.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key")
    return GeminiElectionAssistant()


class TestGeminiInitialisation:
    """Tests for GeminiElectionAssistant initialisation."""

    def test_no_api_key_raises(self, monkeypatch):
        """Should raise ValueError when GOOGLE_API_KEY is not set."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GeminiElectionAssistant()

    def test_success(self, mock_genai, monkeypatch):
        """Should initialise correctly with a valid API key."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key")
        svc = GeminiElectionAssistant()

        mock_genai.configure.assert_called_once_with(api_key="test_api_key")
        assert svc.model_name == "gemini-1.5-pro"
        assert svc.model is not None
        assert svc.chat_session is not None
        assert svc.history_limit == 10


class TestChat:
    """Tests for the chat method."""

    def test_success(self, service):
        """Should return parsed JSON from the model response."""
        expected = {
            "response": "Hello, here is the info.",
            "topic": "greeting",
            "suggested_questions": ["What's next?"],
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(expected)
        service.chat_session.send_message.return_value = mock_response

        result = service.chat("Hello")

        service.chat_session.send_message.assert_called_once()
        assert result == expected


class TestTimeline:
    """Tests for the get_timeline method."""

    def test_success(self, service):
        """Should return structured timeline JSON."""
        expected = {
            "country": "India",
            "timeline": [{"phase": "Registration", "description": "D", "approximate_timeframe": "Jan"}],
            "summary": "Summary",
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(expected)
        service.model.generate_content.return_value = mock_response

        result = service.get_timeline("India")
        assert result == expected


class TestQuiz:
    """Tests for the get_quiz_question method."""

    def test_success(self, service):
        """Should return structured quiz JSON."""
        expected = {
            "question": "What is an EVM?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Explanation",
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(expected)
        service.model.generate_content.return_value = mock_response

        result = service.get_quiz_question()
        assert result == expected


class TestExplainTerm:
    """Tests for the explain_term method."""

    def test_success(self, service):
        """Should return structured term explanation JSON."""
        expected = {
            "term": "EVM",
            "definition": "Def",
            "analogy": "Analogy",
            "example": "Example",
        }
        mock_response = MagicMock()
        mock_response.text = json.dumps(expected)
        service.model.generate_content.return_value = mock_response

        result = service.explain_term("EVM")
        assert result == expected


class TestHistoryTrimming:
    """Tests for chat history management."""

    def test_trims_correctly(self, service):
        """Should trim history to history_limit * 2 messages."""
        service.history_limit = 2
        service.chat_session.history = [1, 2, 3, 4, 5, 6]
        service._trim_history()
        assert service.chat_session.history == [3, 4, 5, 6]


class TestJsonParsing:
    """Tests for JSON response parsing."""

    def test_invalid_json(self, service):
        """Should return error dict for invalid JSON."""
        result = service._parse_json_response("not valid json")
        assert "error" in result
        assert result["raw_response"] == "not valid json"


class TestRetryDecorator:
    """Tests for the exponential backoff retry decorator."""

    def test_eventually_succeeds(self):
        """Should succeed after initial failures."""
        mock_fn = MagicMock(
            side_effect=[Exception("fail"), Exception("fail"), "success"]
        )

        @retry_with_exponential_backoff(max_retries=3, initial_delay=0.01)
        def test_func():
            return mock_fn()

        result = test_func()
        assert result == "success"
        assert mock_fn.call_count == 3

    def test_exhausts_retries(self):
        """Should raise after max retries exhausted."""
        mock_fn = MagicMock(side_effect=Exception("always fail"))

        @retry_with_exponential_backoff(max_retries=2, initial_delay=0.01)
        def test_func():
            return mock_fn()

        with pytest.raises(Exception, match="always fail"):
            test_func()
        assert mock_fn.call_count == 2
