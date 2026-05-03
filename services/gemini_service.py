"""
Gemini AI Service — Election Process Education Assistant.

Wraps the Google Generative AI SDK to provide non-partisan,
educational answers about election processes, voting rights,
and civic participation.

Key features:
    - Gemini 1.5 Pro with structured JSON responses
    - Multi-turn conversation with history trimming
    - Retry logic with exponential backoff
    - Comprehensive safety settings

Author: Ankit Rai
Version: 2.1.0
Usage example:
    from services.gemini_service import GeminiElectionAssistant
    assistant = GeminiElectionAssistant()
    response = assistant.chat("How do I vote?")
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from config import (
    ENV_GOOGLE_API_KEY,
    GEMINI_HISTORY_LIMIT,
    GEMINI_MAX_RETRIES,
    GEMINI_RESPONSE_MIME,
    GEMINI_RETRY_BASE,
    GEMINI_RETRY_DELAY,
    GEMINI_TEMPERATURE,
)

logger = logging.getLogger(__name__)

__all__ = ["GeminiElectionAssistant", "retry_with_exponential_backoff"]

# ---------------------------------------------------------------------------
# System prompt — non-partisan election education
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """
You are an expert Election Process Education Assistant, specializing in global election processes with a focus on India, USA, UK, and the EU.
Your goal is to educate first-time voters and general citizens about democratic processes.

Guidelines:
1. Explain in simple, engaging, and accessible language suitable for first-time voters.
2. Provide step-by-step timelines with dates and milestones when asked about election schedules.
3. Use relatable analogies and real-world examples to clarify complex terms.
4. Offer interactive quizzes to test understanding if the user seems ready or asks for them.
5. Cover key topics comprehensively: voter registration, nomination process, campaigning rules, voting day procedures, vote counting, result declaration, and electoral college/systems.
6. Remain strictly factual, objective, and politically neutral. Do not express political opinions or biases.
7. Detect the language from the user input and respond in the same language.
8. Format responses with clear sections, bullet points, and markdown for readability.
9. Your responses MUST ALWAYS be valid JSON objects with the structure appropriate for the user's request.
"""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry_with_exponential_backoff(
    max_retries: int = GEMINI_MAX_RETRIES,
    initial_delay: float = GEMINI_RETRY_DELAY,
    exponential_base: float = GEMINI_RETRY_BASE,
    allowed_exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator to retry a function with exponential backoff.

    Retries the decorated function up to ``max_retries`` times,
    sleeping for an exponentially increasing delay between attempts.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        exponential_base: Multiplier applied to the delay after each retry.
        allowed_exceptions: Exception types that trigger a retry.

    Returns:
        The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as exc:
                    if attempt == max_retries - 1:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d failed for %s: %s. Retrying in %.1fs…",
                        attempt + 1,
                        func.__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= exponential_base
            return None  # unreachable but satisfies type checkers

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# GeminiElectionAssistant
# ---------------------------------------------------------------------------


class GeminiElectionAssistant:
    """Service class for interacting with Google's Gemini AI model.

    Provides educational, non-partisan election information through
    structured JSON responses. Manages a multi-turn chat session
    with automatic history trimming.

    Attributes:
        api_key: Google API key for authentication.
        model_name: The Gemini model identifier.
        model: The configured GenerativeModel instance.
        chat_session: Active chat session with history.
        history_limit: Maximum conversation turns to retain.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialise the GeminiElectionAssistant.

        Args:
            api_key: Google API key. Falls back to the GOOGLE_API_KEY
                environment variable if not provided.
        """
        self.api_key = (
            api_key
            or os.environ.get(ENV_GOOGLE_API_KEY)
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not self.api_key:
            logger.warning(
                "GOOGLE_API_KEY is not set. Gemini API calls will use fallbacks."
            )
        else:
            genai.configure(api_key=self.api_key)

        self.model_name: str = "gemini-2.0-flash"
        self.generation_config = self._build_generation_config()
        self.safety_settings = self._build_safety_settings()
        self.model = self._create_model()
        try:
            self.chat_session = self.model.start_chat(history=[])
        except Exception as e:
            logger.error(f"{type(e).__name__}: {str(e)}", exc_info=True)
            self.chat_session = None
        self.history_limit: int = GEMINI_HISTORY_LIMIT

    # ------------------------------------------------------------------
    # Private: construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generation_config() -> Any:
        """Create the generation configuration for Gemini.

        Returns:
            GenerationConfig with low temperature and JSON mime type.
        """
        return genai.GenerationConfig(
            temperature=GEMINI_TEMPERATURE,
            response_mime_type=GEMINI_RESPONSE_MIME,
        )

    @staticmethod
    def _build_safety_settings() -> dict:
        """Create safety settings to block harmful content.

        Returns:
            Dictionary mapping HarmCategory to HarmBlockThreshold.
        """
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def _create_model(self) -> Any:
        """Instantiate the GenerativeModel with system prompt.

        Returns:
            Configured GenerativeModel instance.
        """
        return genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )

    # ------------------------------------------------------------------
    # Private: history and parsing
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Trim chat history to the last ``history_limit`` turns.

        One turn = 2 messages (user + model). Keeps conversation
        context manageable and within model token limits.
        """
        max_messages = self.history_limit * 2
        if len(self.chat_session.history) > max_messages:
            self.chat_session.history = self.chat_session.history[
                -max_messages:
            ]
            logger.debug(
                "Trimmed history to last %d turns.", self.history_limit
            )

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """Parse raw model text into a JSON dictionary.

        Args:
            text: Raw text response from the model.

        Returns:
            Parsed dictionary, or an error dict if parsing fails.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse JSON: %s (error: %s)", text[:200], exc
            )
            return {
                "error": "Failed to parse model response as JSON",
                "raw_response": text,
            }

    @staticmethod
    def _build_chat_prompt(user_message: str) -> str:
        """Build the structured chat prompt.

        Args:
            user_message: The user's input message.

        Returns:
            Formatted prompt string requesting JSON output.
        """
        return (
            f"User message: {user_message}\n\n"
            "Please respond with a JSON object. "
            "Structure it with fields: 'response' (markdown text addressing the user), "
            "'topic' (briefly identify the election topic discussed), "
            "and 'suggested_questions' (list of 2-3 relevant follow-up questions)."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry_with_exponential_backoff(max_retries=GEMINI_MAX_RETRIES)
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Send a chat message and retrieve a structured response.

        Args:
            user_message: The user's input message.

        Returns:
            Parsed JSON with 'response', 'topic', 'suggested_questions'.
        """
        prompt = self._build_chat_prompt(user_message)
        logger.info("Sending chat message: %s", user_message[:100])
        try:
            if not self.api_key or not self.chat_session:
                raise ValueError(
                    "Gemini API is not configured or chat session failed to initialize."
                )
            response = self.chat_session.send_message(prompt)
            self._trim_history()
            logger.debug("Received chat response: %s", response.text[:200])
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"{type(e).__name__}: {str(e)}", exc_info=True)
            return {
                "response": "I'm currently operating in offline mode. I can answer basic questions about elections, but my full knowledge base is temporarily unavailable.",
                "topic": "General",
                "suggested_questions": [
                    "What is a democracy?",
                    "How do I register to vote?",
                ],
            }

    @retry_with_exponential_backoff(max_retries=GEMINI_MAX_RETRIES)
    def get_timeline(self, country: str) -> Dict[str, Any]:
        """Retrieve the election timeline for a specific country.

        Args:
            country: The country name to query.

        Returns:
            Parsed JSON with 'country', 'timeline', 'summary'.
        """
        prompt = (
            f"Provide the general election timeline and key milestones for {country}. "
            "Respond ONLY with a JSON object containing exactly these fields:\n"
            "- 'country': The requested country name\n"
            "- 'timeline': A list of objects, each containing 'phase', "
            "'description', and 'approximate_timeframe'.\n"
            "- 'summary': A brief markdown summary."
        )
        logger.info("Requesting timeline for: %s", country)
        try:
            if not self.api_key:
                raise ValueError("Gemini API is not configured.")
            response = self.model.generate_content(prompt)
            logger.debug("Timeline response: %s", response.text[:200])
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"{type(e).__name__}: {str(e)}", exc_info=True)
            raise

    @retry_with_exponential_backoff(max_retries=GEMINI_MAX_RETRIES)
    def get_quiz_question(self) -> Dict[str, Any]:
        """Generate a multiple-choice quiz question about elections.

        Returns:
            Parsed JSON with 'question', 'options', 'correct_answer',
            'explanation'.
        """
        prompt = (
            "Generate a random, informative multiple-choice quiz question about "
            "global election processes (USA, UK, EU, or India). "
            "Respond ONLY with a JSON object containing:\n"
            "- 'question': The question text\n"
            "- 'options': A list of 4 distinct answers\n"
            "- 'correct_answer': The exact string of the correct option\n"
            "- 'explanation': A brief markdown explanation."
        )
        logger.info("Requesting quiz question")
        try:
            if not self.api_key:
                raise ValueError("Gemini API is not configured.")
            response = self.model.generate_content(prompt)
            logger.debug("Quiz response: %s", response.text[:200])
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"{type(e).__name__}: {str(e)}", exc_info=True)
            raise

    @retry_with_exponential_backoff(max_retries=GEMINI_MAX_RETRIES)
    def explain_term(self, term: str) -> Dict[str, Any]:
        """Explain an election-related term in simple language.

        Args:
            term: The election term to explain.

        Returns:
            Parsed JSON with 'term', 'definition', 'analogy', 'example'.
        """
        prompt = (
            f"Explain the election term '{term}' for a first-time voter. "
            "Respond ONLY with a JSON object containing:\n"
            "- 'term': The requested term\n"
            "- 'definition': A simple, engaging definition\n"
            "- 'analogy': A relatable real-world analogy\n"
            "- 'example': An example from a specific country's elections."
        )
        logger.info("Requesting explanation for: %s", term)
        try:
            if not self.api_key:
                raise ValueError("Gemini API is not configured.")
            response = self.model.generate_content(prompt)
            logger.debug("Explanation response: %s", response.text[:200])
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"{type(e).__name__}: {str(e)}")
            raise
