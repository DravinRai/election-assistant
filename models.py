"""
Pydantic-style dataclass models for request/response validation.

Provides structured data models for all API endpoints, replacing
raw dict access with validated, typed objects.

Author: Ankit Rai
Version: 2.1.0
Usage example:
    from models import ChatRequest
    req = ChatRequest(message="Hello")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "ChatRequest",
    "TranslateRequest",
    "TTSRequest",
    "QuizScoreRequest",
    "TimelineRequest",
    "APIResponse",
    "ChatResponse",
    "ModerationResult",
    "ClassificationResult",
    "HealthStatus",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


@dataclass
class ChatRequest:
    """Validated chat endpoint request body.

    Attributes:
        message: The user's chat message.
        history: Optional conversation history for multi-turn context.
        session_id: Optional Firebase session identifier.
    """

    message: str
    history: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""

    def __post_init__(self) -> None:
        """Strip whitespace from message after construction."""
        self.message = self.message.strip()


@dataclass
class TranslateRequest:
    """Validated translate endpoint request body.

    Attributes:
        text: The text to translate.
        target_language: ISO 639-1 target language code.
        source_language: Optional source language code (auto-detected if omitted).
    """

    text: str
    target_language: str = "en"
    source_language: Optional[str] = None

    def __post_init__(self) -> None:
        """Strip whitespace from text after construction."""
        self.text = self.text.strip()


@dataclass
class TTSRequest:
    """Validated text-to-speech endpoint request body.

    Attributes:
        text: The text to convert to audio.
        language: ISO 639-1 language code.
        speaking_rate: Speech speed multiplier (0.25–4.0).
    """

    text: str
    language: str = "en"
    speaking_rate: float = 1.0

    def __post_init__(self) -> None:
        """Strip whitespace and clamp speaking rate."""
        self.text = self.text.strip()
        self.speaking_rate = max(0.25, min(4.0, self.speaking_rate))


@dataclass
class QuizScoreRequest:
    """Validated quiz score request body.

    Attributes:
        score: Number of correct answers.
        total: Total number of questions.
        topic: The quiz topic identifier.
    """

    score: int = 0
    total: int = 0
    topic: str = ""


@dataclass
class TimelineRequest:
    """Validated timeline request body.

    Attributes:
        country: The country name to query.
    """

    country: str = "India"

    def __post_init__(self) -> None:
        """Strip whitespace from country."""
        self.country = self.country.strip()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


@dataclass
class APIResponse:
    """Standard API response envelope.

    Attributes:
        success: Whether the request succeeded.
        data: The response payload.
        error: Error message if the request failed.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with success, data, and optional error fields.
        """
        result: dict[str, Any] = {"success": self.success, **self.data}
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class ChatResponse:
    """Structured chat endpoint response.

    Attributes:
        response: The AI assistant's markdown response.
        topic: Classified election topic.
        confidence: Topic classification confidence score.
        suggested_questions: Follow-up question suggestions.
        success: Whether the request succeeded.
    """

    response: str
    topic: str
    confidence: float
    suggested_questions: list[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with all chat response fields.
        """
        return {
            "response": self.response,
            "topic": self.topic,
            "confidence": self.confidence,
            "suggested_questions": self.suggested_questions,
            "success": self.success,
        }


@dataclass
class ModerationResult:
    """Content moderation result.

    Attributes:
        safe: Whether the content passed moderation.
        reason: Explanation if the content was blocked.
    """

    safe: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with safe and reason fields.
        """
        return {"safe": self.safe, "reason": self.reason}


@dataclass
class ClassificationResult:
    """Topic classification result.

    Attributes:
        topic: The identified election topic.
        confidence: Confidence score (0.0–1.0).
    """

    topic: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with topic and confidence fields.
        """
        return {"topic": self.topic, "confidence": self.confidence}


@dataclass
class HealthStatus:
    """Health check response model.

    Attributes:
        status: Overall health status string.
        timestamp: ISO 8601 timestamp.
        version: Application version.
        services: Per-service availability status.
    """

    status: str
    timestamp: str
    version: str
    services: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dict with all health check fields.
        """
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "version": self.version,
            "services": self.services,
        }
