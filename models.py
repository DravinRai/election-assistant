"""
Pydantic-style dataclass models for request/response validation.

This module provides structured data models for all API endpoints, replacing
raw dict access with validated, typed objects.

Example:
    >>> from models import ChatRequest
    >>> req = ChatRequest(message="Hello")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__: list[str] = [
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
        """Strip whitespace from message after construction.
        
        Detailed description:
            Trims any leading or trailing whitespace from the user's message 
            to ensure clean processing.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> req = ChatRequest(" hello ")
            >>> req.message
            'hello'
        """
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
        """Strip whitespace from text after construction.
        
        Detailed description:
            Trims any leading or trailing whitespace from the text 
            to ensure clean processing.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> req = TranslateRequest(" hello ")
            >>> req.text
            'hello'
        """
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
        """Strip whitespace and clamp speaking rate.
        
        Detailed description:
            Ensures text is stripped of surrounding whitespace and that the
            speaking rate is strictly within the allowed range [0.25, 4.0].
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> req = TTSRequest(" hello ", speaking_rate=5.0)
            >>> req.speaking_rate
            4.0
        """
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
        """Strip whitespace from country.
        
        Detailed description:
            Trims any leading or trailing whitespace from the country string.
            
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
            
        Example:
            >>> req = TimelineRequest(" USA ")
            >>> req.country
            'USA'
        """
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

        Detailed description:
            Converts the dataclass instance into a dictionary structure 
            ready for JSON serialisation.
            
        Args:
            None
            
        Returns:
            A dictionary containing the success flag, merged data fields, 
            and an optional error string.
            
        Raises:
            None
            
        Example:
            >>> res = APIResponse(success=True, data={"key": "value"})
            >>> res.to_dict()
            {'success': True, 'key': 'value'}
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

        Detailed description:
            Converts the ChatResponse object to a primitive dict for the
            Flask response payload.
            
        Args:
            None
            
        Returns:
            A dictionary containing the chat response data.
            
        Raises:
            None
            
        Example:
            >>> res = ChatResponse("Hello", "General", 0.9)
            >>> res.to_dict()['success']
            True
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

        Detailed description:
            Formats the moderation outcome into a serialisable dictionary.
            
        Args:
            None
            
        Returns:
            A dictionary indicating if the input is safe and the reason.
            
        Raises:
            None
            
        Example:
            >>> res = ModerationResult(safe=False, reason="Spam")
            >>> res.to_dict()
            {'safe': False, 'reason': 'Spam'}
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

        Detailed description:
            Formats the topic classification into a serialisable dictionary.
            
        Args:
            None
            
        Returns:
            A dictionary with the topic and its associated confidence.
            
        Raises:
            None
            
        Example:
            >>> res = ClassificationResult("General", 0.8)
            >>> res.to_dict()
            {'topic': 'General', 'confidence': 0.8}
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

        Detailed description:
            Generates a dictionary representation of the health status.
            
        Args:
            None
            
        Returns:
            A dictionary detailing the health status and service connectivity.
            
        Raises:
            None
            
        Example:
            >>> res = HealthStatus("healthy", "2023", "1.0", {})
            >>> res.to_dict()['status']
            'healthy'
        """
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "version": self.version,
            "services": self.services,
        }
