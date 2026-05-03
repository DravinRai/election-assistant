"""
Election Process Education Assistant — Flask Application.

Production-grade entry point implementing MVC separation:
  Controller layer: Route handlers with input validation
  Service layer: Business logic in services/ package
  Model layer: Typed dataclasses in models.py

Security features:
  • Flask-Talisman for CSP, HSTS, X-Frame-Options
  • Flask-Limiter for rate limiting (10 req/min per IP on chat)
  • bleach-based input sanitisation on all user inputs
  • Content-Type validation on POST requests
  • 10 KB max request payload
  • Security event logging
  • Flask-Compress for gzip responses
  • Redis-backed caching with in-memory fallback
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

import re
import bleach
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_compress import Compress
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

from config import (
    APP_VERSION,
    CSP_DIRECTIVES,
    DEFAULT_PORT,
    MAX_CONTENT_LENGTH,
    MAX_MESSAGE_LENGTH,
    RATE_LIMIT_CHAT,
    RATE_LIMIT_DEFAULT,
    RATE_LIMIT_DETECT,
    RATE_LIMIT_HOURLY,
    RATE_LIMIT_LANGUAGES,
    RATE_LIMIT_NEWS,
    RATE_LIMIT_QUIZ,
    RATE_LIMIT_SESSION,
    RATE_LIMIT_TOPICS,
    RATE_LIMIT_TRANSLATE,
    RATE_LIMIT_TTS,
    AppConfig,
)
from models import QuizScoreRequest, TranslateRequest, TTSRequest

__all__ = ["create_app", "app"]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — structured JSON to stdout (Cloud Run picks it up)
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for Cloud Logging ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-encoded string with severity, message, timestamp, and logger.
        """
        log_entry: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": record.name,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _setup_logging() -> None:
    """Configure structured JSON logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]


_setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache (Redis fallback)
# ---------------------------------------------------------------------------
_response_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str) -> dict[str, Any] | None:
    """Retrieve a value from the in-memory cache.

    Args:
        key: The cache key to look up.

    Returns:
        Cached dictionary if found and not expired, else None.
    """
    entry = _response_cache.get(key)
    if entry and time.time() - entry.get("_ts", 0) < _CACHE_TTL:
        return entry.get("_data")
    return None


def _cache_set(key: str, data: dict[str, Any]) -> None:
    """Store a value in the in-memory cache with a timestamp.

    Args:
        key: The cache key.
        data: The dictionary to cache.
    """
    _response_cache[key] = {"_data": data, "_ts": time.time()}


# ---------------------------------------------------------------------------
# Input sanitisation helpers
# ---------------------------------------------------------------------------


def sanitise_input(text: str | None) -> str:
    """Strip HTML tags and dangerous content from user input.

    Removes <script> tags and their inner content entirely.
    Removes all other HTML tags but preserves their text content.
    Strips leading/trailing whitespace and handles None input.

    Args:
        text: Raw user input string or None.

    Returns:
        Sanitised plain-text string.
    """
    if text is None:
        return ""

    # 1. Remove <script> blocks (tag and content) entirely
    # [^>] is safer than .*? for the tag itself, but .*? works for the inner content
    text = re.sub(
        r"<script\b[^>]*>.*?</script>",
        "",
        str(text),
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Remove all other tags but keep their text content
    clean_text = bleach.clean(text, tags=[], attributes={}, strip=True)

    # 3. Strip whitespace
    return clean_text.strip()


def validate_content_type() -> Response | None:
    """Validate that POST requests include application/json Content-Type.

    Returns:
        A 415 JSON error response if content type is invalid, else None.
    """
    content_type = request.content_type or ""
    if "application/json" not in content_type:
        logger.warning(
            "SECURITY: Invalid Content-Type '%s' from %s",
            content_type,
            request.remote_addr,
        )
        return (
            jsonify(
                {
                    "error": "Content-Type must be application/json.",
                    "success": False,
                }
            ),
            415,
        )
    return None


def require_json(func: Callable) -> Callable:
    """Decorator to enforce application/json Content-Type on POST endpoints.

    Args:
        func: The route handler to wrap.

    Returns:
        Wrapped function that validates Content-Type before proceeding.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        error = validate_content_type()
        if error:
            return error
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Security event logging
# ---------------------------------------------------------------------------


def log_security_event(event: str, details: str = "") -> None:
    """Log a security-relevant event with client context.

    Args:
        event: Short event identifier (e.g. 'RATE_LIMIT_HIT').
        details: Additional context about the event.
    """
    logger.warning(
        "SECURITY_EVENT: %s | ip=%s | path=%s | %s",
        event,
        request.remote_addr,
        request.path,
        details,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_service_env(key: str) -> str:
    """Return 'configured' or 'not_configured' based on env var presence.

    Args:
        key: The environment variable name to check.

    Returns:
        'configured' if the env var is set, else 'not_configured'.
    """
    return "configured" if os.environ.get(key) else "not_configured"


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def _setup_security_and_cors(application, cfg):
    """Setup security headers and CORS."""
    Compress(application)
    Talisman(
        application, content_security_policy=CSP_DIRECTIVES, force_https=False
    )
    origins = cfg.allowed_origins.split(",")
    CORS(application, resources={r"/api/*": {"origins": origins}})


def _setup_rate_limiter(application):
    """Setup and return rate limiter."""
    return Limiter(
        key_func=get_remote_address,
        app=application,
        default_limits=[RATE_LIMIT_DEFAULT, RATE_LIMIT_HOURLY],
        storage_uri="memory://",
        on_breach=_on_rate_limit_breach,
    )


def _setup_context_processor(application, cfg):
    """Setup template context processor."""

    @application.context_processor
    def inject_config():
        return {
            "ga_measurement_id": cfg.ga_measurement_id,
            "google_maps_api_key": cfg.google_maps_api_key,
        }


def _register_routes(application, limiter):
    """Register all routes to the application."""
    application.add_url_rule("/", view_func=index)
    application.add_url_rule("/health", view_func=limiter.exempt(health))
    application.add_url_rule(
        "/api/chat",
        view_func=limiter.limit(RATE_LIMIT_CHAT)(require_json(chat)),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/translate",
        view_func=limiter.limit(RATE_LIMIT_TRANSLATE)(require_json(translate)),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/translate/languages",
        view_func=limiter.limit(RATE_LIMIT_LANGUAGES)(translate_languages),
    )
    application.add_url_rule(
        "/api/translate/detect",
        view_func=limiter.limit(RATE_LIMIT_DETECT)(
            require_json(detect_language)
        ),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/tts",
        view_func=limiter.limit(RATE_LIMIT_TTS)(require_json(text_to_speech)),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/news", view_func=limiter.limit(RATE_LIMIT_NEWS)(news_search)
    )
    application.add_url_rule(
        "/api/session",
        view_func=limiter.limit(RATE_LIMIT_SESSION)(create_session),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/session/<session_id>/quiz",
        view_func=limiter.limit(RATE_LIMIT_QUIZ)(
            require_json(save_quiz_score)
        ),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/topics", view_func=limiter.limit(RATE_LIMIT_TOPICS)(topics)
    )
    application.add_url_rule(
        "/api/quiz/question",
        view_func=limiter.limit(RATE_LIMIT_QUIZ)(quiz_question),
    )
    application.add_url_rule(
        "/api/timeline",
        view_func=limiter.limit(RATE_LIMIT_TOPICS)(require_json(timeline)),
        methods=["POST"],
    )
    application.add_url_rule(
        "/api/map", view_func=limiter.limit(RATE_LIMIT_DEFAULT)(map_endpoint)
    )


def _register_error_handlers(application):
    """Register error handlers."""
    application.register_error_handler(404, not_found)
    application.register_error_handler(413, payload_too_large)
    application.register_error_handler(429, rate_limited)
    application.register_error_handler(500, server_error)


def index():
    """Serve the main UI."""
    return render_template("index.html")


def health():
    """Liveness / readiness probe."""
    return _build_health_response()


def chat():
    """Accept a user message, moderate, classify, and respond."""
    return _handle_chat()


def translate():
    """Translate text to a target language."""
    return _handle_translate()


def translate_languages():
    """Return supported languages for translation."""
    from services.translate_service import TranslateService

    svc = TranslateService.get_instance()
    return jsonify(svc.get_supported_languages())


def detect_language():
    """Detect the language of input text."""
    return _handle_detect_language()


def text_to_speech():
    """Convert text to speech audio."""
    return _handle_tts()


def news_search():
    """Search for election-related news."""
    return _handle_news_search()


def create_session():
    """Create a new anonymous session for persistence."""
    from services.firebase_service import FirebaseService

    svc = FirebaseService.get_instance()
    return jsonify(svc.create_session())


def save_quiz_score(session_id):
    """Save a quiz score for a session."""
    return _handle_quiz_score(session_id)


def topics():
    """Return the list of supported election topics."""
    from services.vertex_service import ELECTION_TOPICS

    return jsonify({"topics": ELECTION_TOPICS, "success": True})


def quiz_question():
    """Generate a random election quiz question."""
    return _handle_quiz_question()


def timeline():
    """Retrieve an election timeline for a country."""
    return _handle_timeline()


def map_endpoint():
    """Return map configuration or fallback embed URL."""
    return _handle_map()


def not_found(error: Exception):
    """Handle 404 errors with JSON response."""
    return jsonify({"error": "Resource not found.", "success": False}), 404


def payload_too_large(error: Exception):
    """Handle oversized request payloads."""
    log_security_event("PAYLOAD_TOO_LARGE", f"ip={request.remote_addr}")
    return (
        jsonify({"error": "Request payload exceeds limit.", "success": False}),
        413,
    )


def rate_limited(error: Exception):
    """Handle rate limit exceeded errors."""
    return jsonify({"error": "Too many requests.", "success": False}), 429


def server_error(error: Exception):
    """Handle internal server errors."""
    return (
        jsonify(
            {"error": "An internal server error occurred.", "success": False}
        ),
        500,
    )


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    application = Flask(__name__)
    cfg = AppConfig()
    application.config["SECRET_KEY"] = cfg.flask_secret_key
    application.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    _setup_security_and_cors(application, cfg)
    limiter = _setup_rate_limiter(application)
    _setup_context_processor(application, cfg)
    _register_routes(application, limiter)
    _register_error_handlers(application)

    logger.info("Flask application created with all Google Services.")
    return application


# ---------------------------------------------------------------------------
# Rate limit breach callback
# ---------------------------------------------------------------------------


def _on_rate_limit_breach(limit: Any) -> None:
    """Log rate limit breaches for security monitoring.

    Args:
        limit: The rate limit rule that was breached.
    """
    log_security_event("RATE_LIMIT_HIT", f"limit={limit}")


# ---------------------------------------------------------------------------
# Route handler functions (Controller layer)
# ---------------------------------------------------------------------------


def _build_health_response() -> tuple[Response, int]:
    """Build the health check response payload.

    Returns:
        Tuple of JSON health status and HTTP 200.
    """
    services_status = {
        "gemini": _check_service_env("GOOGLE_API_KEY"),
        "translate": _check_service_env("GOOGLE_TRANSLATE_API_KEY"),
        "tts": _check_service_env("GOOGLE_TTS_API_KEY"),
        "search": _check_service_env("GOOGLE_SEARCH_API_KEY"),
        "firebase": _check_service_env("GOOGLE_CLOUD_PROJECT"),
        "analytics": _check_service_env("GA_MEASUREMENT_ID"),
        "maps": _check_service_env("GOOGLE_MAPS_API_KEY"),
    }
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": APP_VERSION,
                "services": services_status,
            }
        ),
        200,
    )


def _handle_chat() -> tuple[Response, int]:
    """Process a chat request through moderation → classification → Gemini.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.gemini_service import GeminiElectionAssistant
    from services.vertex_service import VertexService

    data = request.get_json(silent=True)
    if not data or not data.get("message"):
        return (
            jsonify(
                {"error": "A 'message' field is required.", "success": False}
            ),
            400,
        )

    raw_message = str(data.get("message", ""))
    user_message = sanitise_input(raw_message)
    if not user_message:
        return (
            jsonify(
                {"error": "A 'message' field is required.", "success": False}
            ),
            400,
        )

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Message exceeds the {MAX_MESSAGE_LENGTH:,}-character limit.",
                    "success": False,
                }
            ),
            400,
        )

    history: list = data.get("history", [])
    session_id: str = sanitise_input(str(data.get("session_id", "")))

    # 1. Content moderation
    vertex = VertexService.get_instance()
    moderation = vertex.moderate_content(user_message)
    if not moderation["safe"]:
        log_security_event("CONTENT_BLOCKED", f"reason={moderation['reason']}")
        return (
            jsonify(
                {
                    "error": moderation["reason"],
                    "success": False,
                    "blocked": True,
                }
            ),
            400,
        )

    # 2. Topic classification
    classification = vertex.classify_topic(user_message)

    # 3. Gemini response
    try:
        gemini = GeminiElectionAssistant()
        result = gemini.chat(user_message)
    except (ValueError, KeyError, ConnectionError, RuntimeError) as exc:
        logger.exception("Gemini chat failed: %s", exc)
        return (
            jsonify(
                {
                    "error": "AI service temporarily unavailable. Please try again.",
                    "success": False,
                }
            ),
            500,
        )

    # 4. Persist to Firestore (fire-and-forget)
    _persist_chat(session_id, user_message, result, classification)

    return (
        jsonify(
            {
                "response": result.get("response", ""),
                "topic": classification["topic"],
                "confidence": classification["confidence"],
                "suggested_questions": result.get("suggested_questions", []),
                "success": True,
            }
        ),
        200,
    )


def _persist_chat(
    session_id: str,
    user_message: str,
    result: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    """Persist chat messages to Firestore (fire-and-forget).

    Args:
        session_id: The session identifier.
        user_message: The user's message.
        result: The Gemini response dictionary.
        classification: The topic classification result.
    """
    if not session_id:
        return
    try:
        from services.firebase_service import FirebaseService

        fb = FirebaseService.get_instance()
        fb.save_message(session_id, "user", user_message)
        fb.save_message(
            session_id,
            "model",
            result.get("response", ""),
            {"topic": classification["topic"]},
        )
    except (ValueError, KeyError, ConnectionError, RuntimeError):
        logger.warning("Firebase persistence failed — continuing without.")


def _handle_translate() -> tuple[Response, int]:
    """Process a translation request.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.translate_service import TranslateService

    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            400,
        )

    req = TranslateRequest(
        text=sanitise_input(str(data["text"])),
        target_language=sanitise_input(str(data.get("target_language", "en"))),
        source_language=data.get("source_language"),
    )
    if not req.text:
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            400,
        )

    svc = TranslateService.get_instance()
    result = svc.translate_text(
        req.text, req.target_language, req.source_language
    )
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


def _handle_detect_language() -> tuple[Response, int]:
    """Process a language detection request.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.translate_service import TranslateService

    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            400,
        )

    svc = TranslateService.get_instance()
    return jsonify(svc.detect_language(sanitise_input(str(data["text"])))), 200


def _handle_tts() -> tuple[Response, int]:
    """Process a text-to-speech request.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.tts_service import TTSService

    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            400,
        )

    req = TTSRequest(
        text=sanitise_input(str(data["text"])),
        language=sanitise_input(str(data.get("language", "en"))),
        speaking_rate=float(data.get("speaking_rate", 1.0)),
    )
    if not req.text:
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            400,
        )

    svc = TTSService.get_instance()
    result = svc.synthesize(req.text, req.language, req.speaking_rate)
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


def _handle_news_search() -> tuple[Response, int]:
    """Process a news search request.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.search_service import SearchService

    query = sanitise_input(request.args.get("query", ""))
    if not query:
        return (
            jsonify(
                {"error": "A 'query' parameter is required.", "success": False}
            ),
            400,
        )

    num = min(int(request.args.get("num", "5")), 10)

    svc = SearchService.get_instance()
    result = svc.search_news(query, num)
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


def _handle_quiz_score(session_id: str) -> tuple[Response, int]:
    """Process a quiz score save request.

    Args:
        session_id: The session identifier from the URL.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.firebase_service import FirebaseService

    data = request.get_json(silent=True)
    if not data:
        return (
            jsonify({"error": "Request body is required.", "success": False}),
            400,
        )

    req = QuizScoreRequest(
        score=int(data.get("score", 0)),
        total=int(data.get("total", 0)),
        topic=sanitise_input(str(data.get("topic", ""))),
    )
    session_id = sanitise_input(session_id)

    svc = FirebaseService.get_instance()
    result = svc.save_quiz_score(session_id, req.score, req.total, req.topic)
    return jsonify(result), 200


FALLBACK_QUESTIONS = [
    {
        "question": "What is the minimum voting age in most democratic countries?",
        "options": ["16", "18", "21", "25"],
        "correct_answer": "18",
        "explanation": "In the majority of democratic nations, including the USA, UK, and India, the legal voting age is 18.",
    },
    {
        "question": "Which of these is NOT a primary function of an election commission?",
        "options": [
            "Registering voters",
            "Counting ballots",
            "Passing new laws",
            "Setting election dates",
        ],
        "correct_answer": "Passing new laws",
        "explanation": "Election commissions are responsible for administering elections, while passing laws is the duty of the legislature.",
    },
    {
        "question": "In the United States, what system determines the winner of a presidential election?",
        "options": [
            "Popular Vote",
            "Electoral College",
            "Parliamentary Majority",
            "Proportional Representation",
        ],
        "correct_answer": "Electoral College",
        "explanation": "The US uses the Electoral College system, where each state has a certain number of electors based on its congressional representation.",
    },
    {
        "question": "In a parliamentary system like the UK or India, who usually becomes the Prime Minister?",
        "options": [
            "The candidate with the most national votes",
            "The leader of the party with the most seats in parliament",
            "The oldest member of parliament",
            "A person appointed by the Supreme Court",
        ],
        "correct_answer": "The leader of the party with the most seats in parliament",
        "explanation": "The Prime Minister is typically the leader of the party or coalition that commands a majority in the lower house of parliament.",
    },
    {
        "question": "What is a 'Swing State' in US elections?",
        "options": [
            "A state that changes its borders",
            "A state where voting is optional",
            "A state where both major parties have similar levels of support",
            "A state that votes first",
        ],
        "correct_answer": "A state where both major parties have similar levels of support",
        "explanation": "Swing states (or battleground states) are states where the race is close and could be won by either Democratic or Republican candidates.",
    },
    {
        "question": "What is the purpose of a primary election?",
        "options": [
            "To elect the president",
            "To select a political party's candidate for an upcoming general election",
            "To vote on local laws",
            "To recall a politician",
        ],
        "correct_answer": "To select a political party's candidate for an upcoming general election",
        "explanation": "Primary elections narrow down the field of candidates within a political party before the general election.",
    },
    {
        "question": "What does EVM stand for in the context of Indian elections?",
        "options": [
            "Electoral Voting Machine",
            "Electronic Voting Machine",
            "Election Verification Mechanism",
            "Early Voting Mandate",
        ],
        "correct_answer": "Electronic Voting Machine",
        "explanation": "India uses Electronic Voting Machines (EVMs) to record votes in state and general elections.",
    },
    {
        "question": "What is 'Proportional Representation'?",
        "options": [
            "An electoral system where parties gain seats in proportion to the number of votes cast for them",
            "A system where the winner takes all seats",
            "A system where only property owners can vote",
            "A system where voting is proportional to income",
        ],
        "correct_answer": "An electoral system where parties gain seats in proportion to the number of votes cast for them",
        "explanation": "In proportional representation systems, if a party wins 30% of the vote, they get roughly 30% of the seats.",
    },
    {
        "question": "What is a referendum?",
        "options": [
            "An election for local mayors",
            "A direct vote by the electorate on a particular proposal or issue",
            "A survey conducted by news agencies",
            "The process of counting votes",
        ],
        "correct_answer": "A direct vote by the electorate on a particular proposal or issue",
        "explanation": "A referendum allows citizens to vote directly on a specific policy, law, or political issue, rather than for a candidate.",
    },
    {
        "question": "What is voter turnout?",
        "options": [
            "The number of people who register to vote",
            "The percentage of eligible voters who cast a ballot in an election",
            "The process of verifying voter identity",
            "The number of invalid ballots",
        ],
        "correct_answer": "The percentage of eligible voters who cast a ballot in an election",
        "explanation": "Voter turnout measures the participation rate of eligible voters in a given election.",
    },
]


def _handle_quiz_question() -> tuple[Response, int]:
    """Generate a quiz question using Gemini.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.gemini_service import GeminiElectionAssistant

    try:
        gemini = GeminiElectionAssistant()
        result = gemini.get_quiz_question()
        return jsonify({**result, "success": True}), 200
    except (ValueError, KeyError, ConnectionError, RuntimeError) as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}", exc_info=True)
        import random

        question = random.choice(FALLBACK_QUESTIONS)
        return jsonify({**question, "success": True, "fallback": True}), 200


FALLBACK_TIMELINES = {
    "india": {
        "country": "India",
        "timeline": [
            {
                "phase": "Announcement",
                "description": "Election Commission of India announces the schedule and the Model Code of Conduct comes into effect.",
                "approximate_timeframe": "45-60 days before voting",
            },
            {
                "phase": "Nominations",
                "description": "Candidates file their nomination papers, which are scrutinised.",
                "approximate_timeframe": "1 week after announcement",
            },
            {
                "phase": "Campaigning",
                "description": "Political parties campaign across constituencies.",
                "approximate_timeframe": "2-3 weeks",
            },
            {
                "phase": "Polling",
                "description": "Voting takes place in multiple phases across the country.",
                "approximate_timeframe": "Spread over several weeks",
            },
            {
                "phase": "Counting & Results",
                "description": "Votes are counted and results are officially declared.",
                "approximate_timeframe": "1 day, usually a few days after final polling phase",
            },
        ],
        "summary": "India's general elections are the largest democratic exercise in the world, managed independently by the Election Commission of India over several phases.",
    },
    "usa": {
        "country": "USA",
        "timeline": [
            {
                "phase": "Primaries & Caucuses",
                "description": "States hold primary elections or caucuses to choose party delegates.",
                "approximate_timeframe": "January - June of election year",
            },
            {
                "phase": "National Conventions",
                "description": "Parties officially nominate their Presidential and Vice-Presidential candidates.",
                "approximate_timeframe": "July - August",
            },
            {
                "phase": "General Election Campaign",
                "description": "Candidates debate and campaign nationally.",
                "approximate_timeframe": "September - October",
            },
            {
                "phase": "Election Day",
                "description": "Voters cast ballots. The Tuesday next after the first Monday in November.",
                "approximate_timeframe": "Early November",
            },
            {
                "phase": "Electoral College Vote",
                "description": "Electors officially cast their votes for President.",
                "approximate_timeframe": "Mid-December",
            },
            {
                "phase": "Inauguration",
                "description": "The newly elected President takes office.",
                "approximate_timeframe": "January 20th",
            },
        ],
        "summary": "The US Presidential election is a lengthy process involving state primaries, national conventions, and the Electoral College system.",
    },
    "uk": {
        "country": "UK",
        "timeline": [
            {
                "phase": "Dissolution of Parliament",
                "description": "Parliament is dissolved ahead of the election.",
                "approximate_timeframe": "25 working days before election",
            },
            {
                "phase": "Nominations",
                "description": "Candidates must submit their nomination papers.",
                "approximate_timeframe": "19 working days before election",
            },
            {
                "phase": "Campaign Period",
                "description": "Parties release manifestos and campaign.",
                "approximate_timeframe": "3-4 weeks",
            },
            {
                "phase": "Polling Day",
                "description": "Voters cast their ballots, typically on a Thursday.",
                "approximate_timeframe": "Election Day",
            },
            {
                "phase": "Counting & Declaration",
                "description": "Votes are counted overnight and winning Members of Parliament (MPs) are announced.",
                "approximate_timeframe": "Night of Election Day / Following Morning",
            },
            {
                "phase": "Formation of Government",
                "description": "The leader of the party with a majority is invited by the Monarch to form a government.",
                "approximate_timeframe": "Immediately following results",
            },
        ],
        "summary": "UK general elections determine the Members of Parliament for the House of Commons, with the majority party leader typically becoming Prime Minister.",
    },
}


def _handle_timeline() -> tuple[Response, int]:
    """Retrieve an election timeline using Gemini.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.gemini_service import GeminiElectionAssistant

    data = request.get_json(silent=True)
    country = sanitise_input(str(data.get("country", "India")))

    try:
        gemini = GeminiElectionAssistant()
        result = gemini.get_timeline(country)
        return jsonify({**result, "success": True}), 200
    except (ValueError, KeyError, ConnectionError, RuntimeError) as exc:
        logger.error(f"{type(exc).__name__}: {str(exc)}", exc_info=True)
        key = country.lower()
        if key not in FALLBACK_TIMELINES:
            key = "india"
        return (
            jsonify(
                {**FALLBACK_TIMELINES[key], "success": True, "fallback": True}
            ),
            200,
        )


def _handle_map() -> tuple[Response, int]:
    """Handle map endpoint request with fallback.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")

    if not api_key:
        logger.warning(
            "GOOGLE_MAPS_API_KEY is not set. Using OpenStreetMap fallback."
        )
        return (
            jsonify(
                {
                    "provider": "openstreetmap",
                    "embed_url": "https://www.openstreetmap.org/export/embed.html",
                    "success": True,
                    "fallback": True,
                }
            ),
            200,
        )

    query = sanitise_input(request.args.get("q", "polling stations near me"))
    import urllib.parse

    encoded_query = urllib.parse.quote(query)
    embed_url = f"https://www.google.com/maps/embed/v1/place?key={api_key}&q={encoded_query}"
    return (
        jsonify(
            {"provider": "google", "embed_url": embed_url, "success": True}
        ),
        200,
    )


# ---------------------------------------------------------------------------
# Entrypoint (used by gunicorn: `gunicorn main:app`)
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    logger.info("Starting development server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
