"""
Comprehensive tests for VertexService.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from services.vertex_service import VertexService


@pytest.fixture(autouse=True)
def _reset_singleton():
    VertexService._instance = None
    yield
    VertexService._instance = None


class TestVertexService:
    """Tests for VertexService coverage."""

    def test_singleton_pattern(self):
        a = VertexService.get_instance()
        b = VertexService.get_instance()
        assert a is b

    def test_init_vertex_success(self):
        with patch("vertexai.init") as mock_init:
            with patch(
                "services.vertex_service.GenerativeModel"
            ) as mock_model:
                os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
                with patch("services.vertex_service._VERTEX_AVAILABLE", True):
                    svc = VertexService()
                    assert svc._model is not None
                    mock_init.assert_called_once_with(
                        project="test-project", location="us-central1"
                    )

    def test_init_vertex_failure(self):
        with patch("vertexai.init", side_effect=Exception("Init error")):
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
            with patch("services.vertex_service._VERTEX_AVAILABLE", True):
                svc = VertexService()
                assert svc._model is None

    def test_moderate_content_blocked_pattern(self):
        svc = VertexService()
        result = svc.moderate_content("how to steal election")
        assert result["safe"] is False
        assert "integrity" in result["reason"]

    def test_moderate_content_safe_heuristic(self):
        svc = VertexService()
        svc._model = None
        result = svc.moderate_content("How do I vote?")
        assert result["safe"] is True
        assert result["reason"] is None

    def test_moderate_content_vertex_success(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"safe": true, "reason": null}'
        mock_model.generate_content.return_value = mock_response
        svc._model = mock_model

        result = svc.moderate_content("How do I vote?")
        assert result["safe"] is True
        mock_model.generate_content.assert_called()

    def test_moderate_content_vertex_unsafe(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"safe": false, "reason": "off topic"}'
        mock_model.generate_content.return_value = mock_response
        svc._model = mock_model

        result = svc.moderate_content("What is for lunch?")
        assert result["safe"] is False
        assert result["reason"] == "off topic"

    def test_moderate_content_vertex_failure_fallback(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Vertex error")
        svc._model = mock_model

        result = svc.moderate_content("How do I vote?")
        assert result["safe"] is True
        assert result["reason"] is None

    def test_classify_topic_vertex_success(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"topic": "voter_registration", "confidence": 0.95}'
        )
        mock_model.generate_content.return_value = mock_response
        svc._model = mock_model

        result = svc.classify_topic("How to register?")
        assert result["topic"] == "voter_registration"
        assert result["confidence"] == 0.95

    def test_classify_topic_vertex_failure_fallback_heuristic(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Vertex error")
        svc._model = mock_model

        result = svc.classify_topic("How do I register to vote?")
        assert result["topic"] == "voter_registration"
        assert result["confidence"] == 0.7

    def test_heuristic_classify_all_rules(self):
        svc = VertexService()

        tests = [
            ("register to vote", "voter_registration"),
            ("absentee ballot", "voting_methods"),
            ("electoral college", "electoral_college"),
            ("ballot measure", "ballot_measures"),
            ("election security", "election_security"),
            ("campaign finance", "campaign_finance"),
            ("gerrymandering", "redistricting"),
            ("just voting", "general_election_info"),
            ("weather", "off_topic"),
        ]

        for text, expected_topic in tests:
            result = svc._heuristic_classify(text)
            assert result["topic"] == expected_topic, f"Failed for {text}"

    def test_build_moderation_prompt(self):
        prompt = VertexService._build_moderation_prompt("test")
        assert "test" in prompt
        assert "JSON" in prompt

    def test_build_classification_prompt(self):
        prompt = VertexService._build_classification_prompt("test")
        assert "test" in prompt
        assert "voter_registration" in prompt

    def test_moderate_content_none_input(self):
        svc = VertexService()
        svc._model = None
        # Heuristic moderate handles it (no blocked patterns in None)
        result = svc.moderate_content(str(None))
        assert result["safe"] is True

    def test_classify_topic_none_input(self):
        svc = VertexService()
        svc._model = None
        # Heuristic classify handles it
        result = svc.classify_topic(str(None))
        assert "topic" in result

    def test_init_vertex_internal_success(self):
        with patch("vertexai.init") as mock_init:
            with patch(
                "services.vertex_service.GenerativeModel"
            ) as mock_model:
                svc = VertexService()
                svc.project_id = "p"
                svc._init_vertex()
                assert svc._model is not None
                mock_init.assert_called()

    def test_init_vertex_internal_exception(self):
        with patch("vertexai.init", side_effect=Exception("Fail")):
            svc = VertexService()
            svc.project_id = "p"
            svc._init_vertex()
            assert svc._model is None

    def test_vertex_moderate_internal_exception(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Vertex fail")
        svc._model = mock_model
        result = svc._vertex_moderate("text")
        assert result["safe"] is True

    def test_vertex_classify_internal_exception(self):
        svc = VertexService()
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Vertex fail")
        svc._model = mock_model
        result = svc._vertex_classify("text")
        assert (
            result["topic"] == "general_election_info"
        )  # Heuristic fallback result for "text"

    def test_heuristic_classify_election_keywords(self):
        svc = VertexService()
        # "president" is in _ELECTION_KEYWORDS but not in rules
        result = svc._heuristic_classify("Who is the president?")
        assert result["topic"] == "general_election_info"
        assert result["confidence"] == 0.5

    def test_build_prompts_direct(self):
        svc = VertexService()
        mod_prompt = svc._build_moderation_prompt("moderate me")
        assert "moderate me" in mod_prompt
        class_prompt = svc._build_classification_prompt("classify me")
        assert "classify me" in class_prompt
        assert "voter_registration" in class_prompt
