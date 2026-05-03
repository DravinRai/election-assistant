"""
Vertex AI Service — content moderation and topic classification.

Uses Vertex AI's text models to:
    - Classify incoming user queries by election-related topic.
    - Moderate content to reject harmful / off-topic requests before
      they reach the main Gemini conversation model.

Falls back to keyword-based heuristics when the Vertex AI SDK
is not installed or the project is not configured.

Author: Ankit Rai
Version: 2.1.0
Usage example:
    from services.vertex_service import VertexService
    svc = VertexService.get_instance()
    res = svc.moderate_content("Hello")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from config import (
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_VERTEX_LOCATION,
    VERTEX_DEFAULT_LOCATION,
    VERTEX_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Guard the import so tests can mock without installing the full SDK
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    _VERTEX_AVAILABLE = True
except ImportError:
    _VERTEX_AVAILABLE = False
    logger.info(
        "Vertex AI SDK not installed — falling back to heuristic moderation."
    )

# ---------------------------------------------------------------------------
# Election-related topic taxonomy
# ---------------------------------------------------------------------------
ELECTION_TOPICS: list[str] = [
    "voter_registration",
    "voting_methods",
    "electoral_college",
    "ballot_measures",
    "election_security",
    "campaign_finance",
    "redistricting",
    "poll_worker_info",
    "election_results",
    "civic_participation",
    "general_election_info",
    "off_topic",
]

# Simple keyword sets for the heuristic fallback
_ELECTION_KEYWORDS: set[str] = {
    "vote",
    "voting",
    "voter",
    "ballot",
    "election",
    "candidate",
    "register",
    "registration",
    "poll",
    "polling",
    "precinct",
    "electoral",
    "college",
    "primary",
    "caucus",
    "runoff",
    "absentee",
    "mail-in",
    "early voting",
    "gerrymandering",
    "redistricting",
    "campaign",
    "referendum",
    "proposition",
    "civic",
    "democracy",
    "democratic",
    "republic",
    "senator",
    "representative",
    "congress",
    "president",
    "governor",
    "mayor",
    "legislature",
    "certification",
}

_BLOCKED_PATTERNS: set[str] = {
    "hack",
    "exploit",
    "steal election",
    "voter fraud scheme",
    "suppress votes",
    "fake ballots",
    "rig election",
}
__all__ = ["VertexService"]


class VertexService:
    """Content moderation and topic classification via Vertex AI.

    Uses the Vertex AI Generative Model for deep classification and
    moderation, with keyword-based heuristic fallbacks when the
    SDK or credentials are unavailable.

    Attributes:
        project_id: Google Cloud project identifier.
        location: Vertex AI serving region.
    """

    _instance: VertexService | None = None

    def __init__(self) -> None:
        """Initialise the VertexService with optional Vertex AI backend."""
        self.project_id: str = os.environ.get(ENV_GOOGLE_CLOUD_PROJECT, "")
        self.location: str = os.environ.get(
            ENV_VERTEX_LOCATION, VERTEX_DEFAULT_LOCATION
        )
        self._model = None

        if _VERTEX_AVAILABLE and self.project_id:
            self._init_vertex()
        else:
            logger.info("VertexService running in heuristic-only mode.")

    def _init_vertex(self) -> None:
        """Attempt to initialise the Vertex AI backend.

        Catches all exceptions to allow graceful fallback.
        """
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(VERTEX_MODEL_NAME)
            logger.info(
                "VertexService initialised (project=%s, location=%s)",
                self.project_id,
                self.location,
            )
        except Exception:
            logger.exception(
                "Failed to initialise Vertex AI — using fallback."
            )

    @classmethod
    def get_instance(cls) -> VertexService:
        """Return the singleton VertexService instance.

        Returns:
            The shared VertexService instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Content Moderation
    # ------------------------------------------------------------------

    def moderate_content(self, text: str) -> dict[str, Any]:
        """Check whether text is safe and on-topic.

        First checks against known blocked patterns, then optionally
        delegates to Vertex AI for deeper analysis.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'safe' (bool) and 'reason' (str or None).
        """
        blocked_reason = self._check_blocked_patterns(text)
        if blocked_reason:
            return {"safe": False, "reason": blocked_reason}

        if self._model:
            return self._vertex_moderate(text)

        return {"safe": True, "reason": None}

    @staticmethod
    def _check_blocked_patterns(text: str) -> str | None:
        """Check text against known blocked patterns.

        Args:
            text: The user's input text.

        Returns:
            Rejection reason if blocked, else None.
        """
        lower = text.lower()
        for pattern in _BLOCKED_PATTERNS:
            if pattern in lower:
                return (
                    "Your message appears to contain content that could "
                    "undermine election integrity. I can only provide "
                    "educational information about election processes."
                )
        return None

    def _vertex_moderate(self, text: str) -> dict[str, Any]:
        """Use Vertex AI for deep content moderation.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'safe' (bool) and 'reason' (str or None).
        """
        try:
            prompt = self._build_moderation_prompt(text)
            response = self._model.generate_content(prompt)
            result = json.loads(
                response.text.strip().strip("```json").strip("```")
            )
            return {
                "safe": result.get("safe", True),
                "reason": result.get("reason"),
            }
        except Exception:
            logger.exception("Vertex moderation failed — allowing through.")
            return {"safe": True, "reason": None}

    @staticmethod
    def _build_moderation_prompt(text: str) -> str:
        """Build the moderation prompt for Vertex AI.

        Args:
            text: The user's input text.

        Returns:
            Formatted moderation prompt string.
        """
        return (
            "You are a content moderator for an election education platform.\n"
            "Evaluate if the following user message is appropriate and on-topic "
            "for an election education assistant.\n\n"
            f'User message: """{text}"""\n\n'
            "Respond with ONLY a JSON object:\n"
            '{"safe": true/false, "reason": "explanation if not safe"}'
        )

    # ------------------------------------------------------------------
    # Topic Classification
    # ------------------------------------------------------------------

    def classify_topic(self, text: str) -> dict[str, Any]:
        """Classify text into one of the ELECTION_TOPICS.

        Uses Vertex AI when available, otherwise falls back to
        keyword-based heuristics.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' (str) and 'confidence' (float).
        """
        if self._model:
            return self._vertex_classify(text)
        return self._heuristic_classify(text)

    def _vertex_classify(self, text: str) -> dict[str, Any]:
        """Use Vertex AI for topic classification.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' and 'confidence'.
        """
        try:
            prompt = self._build_classification_prompt(text)
            response = self._model.generate_content(prompt)
            result = json.loads(
                response.text.strip().strip("```json").strip("```")
            )
            return {
                "topic": result.get("topic", "general_election_info"),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except Exception:
            logger.exception("Vertex classification failed — using heuristic.")
            return self._heuristic_classify(text)

    @staticmethod
    def _build_classification_prompt(text: str) -> str:
        """Build the classification prompt for Vertex AI.

        Args:
            text: The user's input text.

        Returns:
            Formatted classification prompt string.
        """
        topics_str = ", ".join(ELECTION_TOPICS)
        return (
            "Classify the following user question into one of these election "
            f"education topics: {topics_str}\n\n"
            f'User question: """{text}"""\n\n'
            "Respond with ONLY a JSON object:\n"
            '{"topic": "topic_name", "confidence": 0.0-1.0}'
        )

    @staticmethod
    def _heuristic_classify(text: str) -> dict[str, Any]:
        """Keyword-based fallback classification.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' and 'confidence'.
        """
        lower = text.lower()

        # Ordered by specificity (most specific first)
        rules: list[tuple[tuple[str, ...], str, float]] = [
            (
                ("register", "registration", "sign up to vote"),
                "voter_registration",
                0.7,
            ),
            (
                ("absentee", "mail-in", "early voting", "how to vote"),
                "voting_methods",
                0.7,
            ),
            (("electoral college",), "electoral_college", 0.8),
            (
                ("ballot measure", "referendum", "proposition"),
                "ballot_measures",
                0.7,
            ),
            (("security", "secure", "integrity"), "election_security", 0.6),
            (
                ("campaign", "donation", "finance", "pac"),
                "campaign_finance",
                0.7,
            ),
            (("redistrict", "gerrymander"), "redistricting", 0.8),
        ]

        for keywords, topic, confidence in rules:
            if any(kw in lower for kw in keywords):
                return {"topic": topic, "confidence": confidence}

        if any(kw in lower for kw in _ELECTION_KEYWORDS):
            return {"topic": "general_election_info", "confidence": 0.5}

        return {"topic": "off_topic", "confidence": 0.4}
