"""
Vertex AI Service — content moderation and topic classification.

Uses Vertex AI's text models to:
    - Classify incoming user queries by election-related topic.
    - Moderate content to reject harmful / off-topic requests before
      they reach the main Gemini conversation model.

Falls back to keyword-based heuristics when the Vertex AI SDK
is not installed or the project is not configured.

Example:
    >>> from services.vertex_service import VertexService
    >>> svc = VertexService.get_instance()
    >>> res = svc.moderate_content("Hello")
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
    DEFAULT_CONFIDENCE,
)

logger: logging.Logger = logging.getLogger(__name__)

# Guard the import so tests can mock without installing the full SDK
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel

    _VERTEX_AVAILABLE: bool = True
except ImportError:
    _VERTEX_AVAILABLE: bool = False
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

__all__: list[str] = ["VertexService", "ELECTION_TOPICS"]


class VertexService:
    """Content moderation and topic classification via Vertex AI.

    Uses the Vertex AI Generative Model for deep classification and
    moderation, with keyword-based heuristic fallbacks when the
    SDK or credentials are unavailable.

    Attributes:
        _instance: Singleton instance of VertexService.
        project_id: Google Cloud project identifier.
        location: Vertex AI serving region.
        _model: The configured GenerativeModel instance.
    """

    _instance: VertexService | None = None
    project_id: str
    location: str
    _model: Any

    def __init__(self) -> None:
        """Initialise the VertexService with optional Vertex AI backend.
        
        Detailed description:
            Configures the project ID, location, and attempts to initialise
            the Vertex AI model.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc = VertexService()
        """
        self.project_id = os.environ.get(ENV_GOOGLE_CLOUD_PROJECT, "")
        self.location = os.environ.get(
            ENV_VERTEX_LOCATION, VERTEX_DEFAULT_LOCATION
        )
        self._model = None

        if _VERTEX_AVAILABLE and self.project_id:
            self._init_vertex()
        else:
            logger.info("VertexService running in heuristic-only mode.")

    def _init_vertex(self) -> None:
        """Attempt to initialise the Vertex AI backend.

        Detailed description:
            Calls vertexai.init and creates a GenerativeModel object. Catches 
            exceptions gracefully for fallback modes.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> svc._init_vertex()
        """
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(VERTEX_MODEL_NAME)
            logger.info(
                "VertexService initialised (project=%s, location=%s)",
                self.project_id,
                self.location,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error(
                "Failed to initialise Vertex AI — using fallback: %s", exc
            )

    @classmethod
    def get_instance(cls) -> VertexService:
        """Return the singleton VertexService instance.

        Detailed description:
            Provides singleton access to the configured vertex models.
            
        Args:
            None
            
        Returns:
            The shared VertexService instance.
            
        Raises:
            None
            
        Example:
            >>> svc = VertexService.get_instance()
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Content Moderation
    # ------------------------------------------------------------------

    def moderate_content(self, text: str) -> dict[str, Any]:
        """Check whether text is safe and on-topic.

        Detailed description:
            First checks against known blocked patterns, then optionally
            delegates to Vertex AI for deeper analysis if available.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'safe' (bool) and 'reason' (str or None).
            
        Raises:
            None
            
        Example:
            >>> res = svc.moderate_content("How to vote?")
        """
        blocked_reason: str | None = self._check_blocked_patterns(text)
        if blocked_reason:
            return {"safe": False, "reason": blocked_reason}

        if self._model:
            return self._vertex_moderate(text)

        return {"safe": True, "reason": None}

    @staticmethod
    def _check_blocked_patterns(text: str) -> str | None:
        """Check text against known blocked patterns.

        Detailed description:
            Provides basic string matching to reject known negative phrases.
            
        Args:
            text: The user's input text.

        Returns:
            Rejection reason if blocked, else None.
            
        Raises:
            None
            
        Example:
            >>> reason = VertexService._check_blocked_patterns("rig election")
        """
        lower: str = text.lower()
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

        Detailed description:
            Sends a prompt asking the model to evaluate text safety. Handles JSON 
            responses safely.
            
        Args:
            text: The user's input text.

        Returns:
            Dict with 'safe' (bool) and 'reason' (str or None).
            
        Raises:
            None
            
        Example:
            >>> res = svc._vertex_moderate("Safe text")
        """
        try:
            prompt: str = self._build_moderation_prompt(text)
            response: Any = self._model.generate_content(prompt)
            result: dict[str, Any] = json.loads(
                response.text.strip().strip("```json").strip("```")
            )
            return {
                "safe": result.get("safe", True),
                "reason": result.get("reason"),
            }
        except (ValueError, RuntimeError, json.JSONDecodeError, ConnectionError) as exc:
            logger.error("Vertex moderation failed — allowing through: %s", exc)
            return {"safe": True, "reason": None}

    @staticmethod
    def _build_moderation_prompt(text: str) -> str:
        """Build the moderation prompt for Vertex AI.

        Detailed description:
            Formats the instruction to the generative model for safety evaluation.
            
        Args:
            text: The user's input text.

        Returns:
            Formatted moderation prompt string.
            
        Raises:
            None
            
        Example:
            >>> prompt = VertexService._build_moderation_prompt("Hi")
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

        Detailed description:
            Uses Vertex AI when available, otherwise falls back to
            keyword-based heuristics.

        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' (str) and 'confidence' (float).
            
        Raises:
            None
            
        Example:
            >>> res = svc.classify_topic("Where is the polling place?")
        """
        if self._model:
            return self._vertex_classify(text)
        return self._heuristic_classify(text)

    def _vertex_classify(self, text: str) -> dict[str, Any]:
        """Use Vertex AI for topic classification.

        Detailed description:
            Prompts the model to categorise user text into known topics.
            
        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' and 'confidence'.
            
        Raises:
            None
            
        Example:
            >>> res = svc._vertex_classify("Where is my ballot?")
        """
        try:
            prompt: str = self._build_classification_prompt(text)
            response: Any = self._model.generate_content(prompt)
            result: dict[str, Any] = json.loads(
                response.text.strip().strip("```json").strip("```")
            )
            return {
                "topic": result.get("topic", "general_election_info"),
                "confidence": float(result.get("confidence", DEFAULT_CONFIDENCE)),
            }
        except (ValueError, RuntimeError, json.JSONDecodeError, ConnectionError) as exc:
            logger.error("Vertex classification failed — using heuristic: %s", exc)
            return self._heuristic_classify(text)

    @staticmethod
    def _build_classification_prompt(text: str) -> str:
        """Build the classification prompt for Vertex AI.

        Detailed description:
            Injects the allowed topics and user text into a strict classification prompt.
            
        Args:
            text: The user's input text.

        Returns:
            Formatted classification prompt string.
            
        Raises:
            None
            
        Example:
            >>> prompt = VertexService._build_classification_prompt("Hi")
        """
        topics_str: str = ", ".join(ELECTION_TOPICS)
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

        Detailed description:
            Scans the lowercased text against tuples of keywords for topics.
            
        Args:
            text: The user's input text.

        Returns:
            Dict with 'topic' and 'confidence'.
            
        Raises:
            None
            
        Example:
            >>> res = VertexService._heuristic_classify("How to register")
        """
        lower: str = text.lower()

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
            return {"topic": "general_election_info", "confidence": DEFAULT_CONFIDENCE}

        return {"topic": "off_topic", "confidence": 0.4}
