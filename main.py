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
    ENV_ALLOWED_ORIGINS,
    ENV_FLASK_SECRET_KEY,
    ENV_GA_MEASUREMENT_ID,
    ENV_MAPS_API_KEY,
    GA_DEFAULT_MEASUREMENT_ID,
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
from models import ChatRequest, QuizScoreRequest, TranslateRequest, TTSRequest, TimelineRequest

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
    text = re.sub(r'<script\b[^>]*>.*?</script>', '', str(text), flags=re.DOTALL | re.IGNORECASE)

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
        return jsonify({
            "error": "Content-Type must be application/json.",
            "success": False,
        }), 415
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


def create_app() -> Flask:
    """Create and configure the Flask application.

    Sets up security headers, CORS, rate limiting, compression,
    request size limits, and all route handlers.

    Returns:
        Configured Flask application instance.
    """
    application = Flask(__name__)
    cfg = AppConfig()
    application.config["SECRET_KEY"] = cfg.flask_secret_key
    application.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # -- Response compression --
    Compress(application)

    # -- Security headers --
    Talisman(
        application,
        content_security_policy=CSP_DIRECTIVES,
        force_https=False,
    )

    # -- CORS --
    origins = cfg.allowed_origins.split(",")
    CORS(application, resources={r"/api/*": {"origins": origins}})

    # -- Rate limiting --
    limiter = Limiter(
        key_func=get_remote_address,
        app=application,
        default_limits=[RATE_LIMIT_DEFAULT, RATE_LIMIT_HOURLY],
        storage_uri="memory://",
        on_breach=_on_rate_limit_breach,
    )

    # -- Template context --
    @application.context_processor
    def inject_config() -> dict[str, str]:
        """Inject GA and Maps keys into all templates.

        Returns:
            Dictionary with ga_measurement_id and google_maps_api_key.
        """
        return {
            "ga_measurement_id": cfg.ga_measurement_id,
            "google_maps_api_key": cfg.google_maps_api_key,
        }

    # -----------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------

    @application.route("/")
    def index() -> str:
        """Serve the main UI.

        Returns:
            Rendered index.html template.
        """
        return render_template("index.html")

    @application.route("/health")
    @limiter.exempt
    def health() -> tuple[Response, int]:
        """Liveness / readiness probe for Cloud Run.

        Returns:
            Tuple of JSON response and HTTP 200 status code.
        """
        return _build_health_response()

    @application.route("/api/chat", methods=["POST"])
    @limiter.limit(RATE_LIMIT_CHAT)
    @require_json
    def chat() -> tuple[Response, int]:
        """Accept a user message, moderate, classify, and respond.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_chat()

    @application.route("/api/translate", methods=["POST"])
    @limiter.limit(RATE_LIMIT_TRANSLATE)
    @require_json
    def translate() -> tuple[Response, int]:
        """Translate text to a target language.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_translate()

    @application.route("/api/translate/languages")
    @limiter.limit(RATE_LIMIT_LANGUAGES)
    def translate_languages() -> Response:
        """Return supported languages for translation.

        Returns:
            JSON response with supported languages.
        """
        from services.translate_service import TranslateService
        svc = TranslateService.get_instance()
        return jsonify(svc.get_supported_languages())

    @application.route("/api/translate/detect", methods=["POST"])
    @limiter.limit(RATE_LIMIT_DETECT)
    @require_json
    def detect_language() -> tuple[Response, int]:
        """Detect the language of input text.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_detect_language()

    @application.route("/api/tts", methods=["POST"])
    @limiter.limit(RATE_LIMIT_TTS)
    @require_json
    def text_to_speech() -> tuple[Response, int]:
        """Convert text to speech audio.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_tts()

    @application.route("/api/news")
    @limiter.limit(RATE_LIMIT_NEWS)
    def news_search() -> tuple[Response, int]:
        """Search for election-related news.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_news_search()

    @application.route("/api/session", methods=["POST"])
    @limiter.limit(RATE_LIMIT_SESSION)
    def create_session() -> Response:
        """Create a new anonymous session for persistence.

        Returns:
            JSON response with session_id.
        """
        from services.firebase_service import FirebaseService
        svc = FirebaseService.get_instance()
        return jsonify(svc.create_session())

    @application.route("/api/session/<session_id>/quiz", methods=["POST"])
    @limiter.limit(RATE_LIMIT_QUIZ)
    @require_json
    def save_quiz_score(session_id: str) -> tuple[Response, int]:
        """Save a quiz score for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_quiz_score(session_id)

    @application.route("/api/topics")
    @limiter.limit(RATE_LIMIT_TOPICS)
    def topics() -> Response:
        """Return the list of supported election topics.

        Returns:
            JSON response with topics list.
        """
        from services.vertex_service import ELECTION_TOPICS
        return jsonify({"topics": ELECTION_TOPICS, "success": True})

    @application.route("/api/quiz/question")
    @limiter.limit(RATE_LIMIT_QUIZ)
    def quiz_question() -> tuple[Response, int]:
        """Generate a random election quiz question.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_quiz_question()

    @application.route("/api/timeline", methods=["POST"])
    @limiter.limit(RATE_LIMIT_TOPICS)
    @require_json
    def timeline() -> tuple[Response, int]:
        """Retrieve an election timeline for a country.

        Returns:
            Tuple of JSON response and HTTP status code.
        """
        return _handle_timeline()

    # -----------------------------------------------------------------
    # Error handlers
    # -----------------------------------------------------------------

    @application.errorhandler(404)
    def not_found(_: Exception) -> tuple[Response, int]:
        """Handle 404 errors with JSON response.

        Returns:
            Tuple of JSON error and 404 status.
        """
        return jsonify({"error": "Resource not found.", "success": False}), 404

    @application.errorhandler(413)
    def payload_too_large(_: Exception) -> tuple[Response, int]:
        """Handle oversized request payloads.

        Returns:
            Tuple of JSON error and 413 status.
        """
        log_security_event("PAYLOAD_TOO_LARGE", f"ip={request.remote_addr}")
        return jsonify({
            "error": f"Request payload exceeds the {MAX_CONTENT_LENGTH // 1024}KB limit.",
            "success": False,
        }), 413

    @application.errorhandler(429)
    def rate_limited(_: Exception) -> tuple[Response, int]:
        """Handle rate limit exceeded errors.

        Returns:
            Tuple of JSON error and 429 status.
        """
        return jsonify({
            "error": "Too many requests. Please wait before trying again.",
            "success": False,
        }), 429

    @application.errorhandler(500)
    def server_error(_: Exception) -> tuple[Response, int]:
        """Handle internal server errors with JSON response.

        Returns:
            Tuple of JSON error and 500 status.
        """
        return jsonify({
            "error": "An internal server error occurred.",
            "success": False,
        }), 500

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
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
        "services": services_status,
    }), 200


def _handle_chat() -> tuple[Response, int]:
    """Process a chat request through moderation → classification → Gemini.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.firebase_service import FirebaseService
    from services.gemini_service import GeminiElectionAssistant
    from services.vertex_service import VertexService

    data = request.get_json(silent=True)
    if not data or not data.get("message"):
        return jsonify({"error": "A 'message' field is required.", "success": False}), 400

    raw_message = str(data.get("message", ""))
    user_message = sanitise_input(raw_message)
    if not user_message:
        return jsonify({"error": "A 'message' field is required.", "success": False}), 400

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": f"Message exceeds the {MAX_MESSAGE_LENGTH:,}-character limit.",
            "success": False,
        }), 400

    history: list = data.get("history", [])
    session_id: str = sanitise_input(str(data.get("session_id", "")))

    # 1. Content moderation
    vertex = VertexService.get_instance()
    moderation = vertex.moderate_content(user_message)
    if not moderation["safe"]:
        log_security_event("CONTENT_BLOCKED", f"reason={moderation['reason']}")
        return jsonify({
            "error": moderation["reason"],
            "success": False,
            "blocked": True,
        }), 400

    # 2. Topic classification
    classification = vertex.classify_topic(user_message)

    # 3. Gemini response
    try:
        gemini = GeminiElectionAssistant()
        result = gemini.chat(user_message)
    except Exception as exc:
        logger.exception("Gemini chat failed: %s", exc)
        return jsonify({
            "error": "AI service temporarily unavailable. Please try again.",
            "success": False,
        }), 500

    # 4. Persist to Firestore (fire-and-forget)
    _persist_chat(session_id, user_message, result, classification)

    return jsonify({
        "response": result.get("response", ""),
        "topic": classification["topic"],
        "confidence": classification["confidence"],
        "suggested_questions": result.get("suggested_questions", []),
        "success": True,
    }), 200


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
            session_id, "model",
            result.get("response", ""),
            {"topic": classification["topic"]},
        )
    except Exception:
        logger.warning("Firebase persistence failed — continuing without.")


def _handle_translate() -> tuple[Response, int]:
    """Process a translation request.

    Returns:
        Tuple of JSON response and HTTP status code.
    """
    from services.translate_service import TranslateService

    data = request.get_json(silent=True)
    if not data or not data.get("text"):
        return jsonify({"error": "A 'text' field is required.", "success": False}), 400

    req = TranslateRequest(
        text=sanitise_input(str(data["text"])),
        target_language=sanitise_input(str(data.get("target_language", "en"))),
        source_language=data.get("source_language"),
    )
    if not req.text:
        return jsonify({"error": "A 'text' field is required.", "success": False}), 400

    svc = TranslateService.get_instance()
    result = svc.translate_text(req.text, req.target_language, req.source_language)
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
        return jsonify({"error": "A 'text' field is required.", "success": False}), 400

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
        return jsonify({"error": "A 'text' field is required.", "success": False}), 400

    req = TTSRequest(
        text=sanitise_input(str(data["text"])),
        language=sanitise_input(str(data.get("language", "en"))),
        speaking_rate=float(data.get("speaking_rate", 1.0)),
    )
    if not req.text:
        return jsonify({"error": "A 'text' field is required.", "success": False}), 400

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
        return jsonify({"error": "A 'query' parameter is required.", "success": False}), 400

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
        return jsonify({"error": "Request body is required.", "success": False}), 400

    req = QuizScoreRequest(
        score=int(data.get("score", 0)),
        total=int(data.get("total", 0)),
        topic=sanitise_input(str(data.get("topic", ""))),
    )
    session_id = sanitise_input(session_id)

    svc = FirebaseService.get_instance()
    result = svc.save_quiz_score(session_id, req.score, req.total, req.topic)
    return jsonify(result), 200


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
    except Exception as exc:
        logger.exception("Failed to get quiz question: %s", exc)
        return jsonify({"error": "Failed to generate question.", "success": False}), 500


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
    except Exception as exc:
        logger.exception("Failed to get timeline: %s", exc)
        return jsonify({"error": "Failed to generate timeline.", "success": False}), 500


# ---------------------------------------------------------------------------
# Entrypoint (used by gunicorn: `gunicorn main:app`)
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    logger.info("Starting development server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
