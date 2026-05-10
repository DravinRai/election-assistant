"""
Election Process Education Assistant — Flask Application.

Production-grade entry point implementing MVC separation:
  Controller layer: Route handlers with input validation
  Service layer: Business logic in services/ package
  Model layer: Typed dataclasses in models.py

Security features:
  - Flask-Talisman for CSP, HSTS, X-Frame-Options
  - Flask-Limiter for rate limiting (10 req/min per IP on chat)
  - bleach-based input sanitisation on all user inputs
  - Content-Type validation on POST requests
  - 10 KB max request payload
  - Security event logging
  - Flask-Compress for gzip responses
  - Redis-backed caching with in-memory fallback

Example:
    >>> from main import create_app
    >>> application = create_app()
"""

from __future__ import annotations

# pylint: disable=import-outside-toplevel
import json
import logging
import os
import sys
import time
import random
import urllib.parse
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
    CONTENT_TYPE_JSON,
    CACHE_TTL_SECONDS,
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_413_PAYLOAD_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from models import QuizScoreRequest, TranslateRequest, TTSRequest

__all__: list[str] = ["create_app", "app"]

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
        
        Detailed description:
            Serialises the log record into a single-line JSON format.

        Args:
            record: The log record to format.

        Returns:
            A JSON-encoded string with severity, message, timestamp, and logger.
            
        Raises:
            None
            
        Example:
            >>> fmt = _JsonFormatter()
            >>> msg = fmt.format(record)
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
    """Configure structured JSON logging to stdout.
    
    Detailed description:
        Sets up the root logger to output JSON structured logs to stdout.
        
    Args:
        None
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _setup_logging()
    """
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root: logging.Logger = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]


_setup_logging()
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache (Redis fallback)
# ---------------------------------------------------------------------------
_response_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL: int = CACHE_TTL_SECONDS


def _cache_get(key: str) -> dict[str, Any] | None:
    """Retrieve a value from the in-memory cache.

    Detailed description:
        Looks up a key in the cache and returns its payload if not expired.
        
    Args:
        key: The cache key to look up.

    Returns:
        Cached dictionary if found and not expired, else None.
        
    Raises:
        None
        
    Example:
        >>> val = _cache_get("my_key")
    """
    entry: dict[str, Any] | None = _response_cache.get(key)
    if entry and time.time() - entry.get("_ts", 0) < _CACHE_TTL:
        return entry.get("_data")
    return None


def _cache_set(key: str, data: dict[str, Any]) -> None:
    """Store a value in the in-memory cache with a timestamp.

    Detailed description:
        Writes a value to the in-memory dictionary cache with the current time.
        
    Args:
        key: The cache key.
        data: The dictionary to cache.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _cache_set("key", {"a": 1})
    """
    _response_cache[key] = {"_data": data, "_ts": time.time()}


# ---------------------------------------------------------------------------
# Input sanitisation helpers
# ---------------------------------------------------------------------------


def sanitise_input(text: str | None) -> str:
    """Strip HTML tags and dangerous content from user input.

    Detailed description:
        Removes <script> tags and their inner content entirely.
        Removes all other HTML tags but preserves their text content.
        Strips leading/trailing whitespace and handles None input.

    Args:
        text: Raw user input string or None.

    Returns:
        Sanitised plain-text string.
        
    Raises:
        None
        
    Example:
        >>> clean = sanitise_input("<script>alert(1)</script>hello")
    """
    if text is None:
        return ""

    # 1. Remove <script> blocks (tag and content) entirely
    clean_text: str = re.sub(
        r"<script\b[^>]*>.*?</script>",
        "",
        str(text),
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Remove all other tags but keep their text content
    clean_text = bleach.clean(clean_text, tags=[], attributes={}, strip=True)

    # 3. Strip whitespace
    return clean_text.strip()


def validate_content_type() -> tuple[Response, int] | None:
    """Validate that POST requests include application/json Content-Type.

    Detailed description:
        Ensures clients are sending the correct content type to avoid parsing 
        errors.
        
    Args:
        None
        
    Returns:
        A 415 JSON error response if content type is invalid, else None.
        
    Raises:
        None
        
    Example:
        >>> err = validate_content_type()
    """
    content_type: str = request.content_type or ""
    if CONTENT_TYPE_JSON not in content_type:
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
            HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    return None


def require_json(func: Callable) -> Callable:
    """Decorator to enforce application/json Content-Type on POST endpoints.

    Detailed description:
        Wraps a Flask route handler, checking the content type before executing.
        
    Args:
        func: The route handler to wrap.

    Returns:
        Wrapped function that validates Content-Type before proceeding.
        
    Raises:
        None
        
    Example:
        >>> @app.route('/api', methods=['POST'])
        >>> @require_json
        >>> def api_route(): pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        error: tuple[Response, int] | None = validate_content_type()
        if error:
            return error
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Security event logging
# ---------------------------------------------------------------------------


def log_security_event(event: str, details: str = "") -> None:
    """Log a security-relevant event with client context.

    Detailed description:
        Emits a structured log warning for potential security issues, including 
        the client IP and requested path.
        
    Args:
        event: Short event identifier (e.g. 'RATE_LIMIT_HIT').
        details: Additional context about the event.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> log_security_event("AUTH_FAIL", "user=admin")
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

    Detailed description:
        Checks whether an environment variable is present and not empty.
        
    Args:
        key: The environment variable name to check.

    Returns:
        'configured' if the env var is set, else 'not_configured'.
        
    Raises:
        None
        
    Example:
        >>> status = _check_service_env("PORT")
    """
    return "configured" if os.environ.get(key) else "not_configured"


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def _setup_security_and_cors(application: Flask, cfg: AppConfig) -> None:
    """Setup security headers and CORS.
    
    Detailed description:
        Configures Talisman and CORS based on application configuration.
        
    Args:
        application: The Flask application.
        cfg: Application configuration dataclass.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _setup_security_and_cors(app, config)
    """
    Compress(application)
    Talisman(
        application, content_security_policy=CSP_DIRECTIVES, force_https=False
    )
    origins: list[str] = cfg.allowed_origins.split(",")
    CORS(application, resources={r"/api/*": {"origins": origins}})


def _setup_rate_limiter(application: Flask) -> Limiter:
    """Setup and return rate limiter.
    
    Detailed description:
        Initialises Flask-Limiter for the application.
        
    Args:
        application: The Flask application.
        
    Returns:
        The configured Limiter instance.
        
    Raises:
        None
        
    Example:
        >>> limiter = _setup_rate_limiter(app)
    """
    return Limiter(
        key_func=get_remote_address,
        app=application,
        default_limits=[RATE_LIMIT_DEFAULT, RATE_LIMIT_HOURLY],
        storage_uri="memory://",
        on_breach=_on_rate_limit_breach,
    )


def _setup_context_processor(application: Flask, cfg: AppConfig) -> None:
    """Setup template context processor.
    
    Detailed description:
        Injects global variables into Jinja2 templates.
        
    Args:
        application: The Flask application.
        cfg: Application configuration.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _setup_context_processor(app, cfg)
    """

    @application.context_processor
    def inject_config() -> dict[str, str]:
        return {
            "ga_measurement_id": cfg.ga_measurement_id,
            "google_maps_api_key": cfg.google_maps_api_key,
        }


def _register_routes(application: Flask, limiter: Limiter) -> None:
    """Register all routes to the application.
    
    Detailed description:
        Binds view functions to URLs and attaches rate limiting decorators.
        
    Args:
        application: The Flask application.
        limiter: The rate limiter instance.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _register_routes(app, limiter)
    """
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


def _register_error_handlers(application: Flask) -> None:
    """Register error handlers.
    
    Detailed description:
        Binds custom JSON responses for specific HTTP errors.
        
    Args:
        application: The Flask application.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _register_error_handlers(app)
    """
    application.register_error_handler(HTTP_404_NOT_FOUND, not_found)
    application.register_error_handler(HTTP_413_PAYLOAD_TOO_LARGE, payload_too_large)
    application.register_error_handler(HTTP_429_TOO_MANY_REQUESTS, rate_limited)
    application.register_error_handler(HTTP_500_INTERNAL_SERVER_ERROR, server_error)


def index() -> str:
    """Serve the main UI.
    
    Detailed description:
        Renders the index.html template for the frontend interface.
        
    Args:
        None
        
    Returns:
        String of rendered HTML.
        
    Raises:
        None
        
    Example:
        >>> html = index()
    """
    return render_template("index.html")


def health() -> tuple[Response, int]:
    """Liveness / readiness probe.
    
    Detailed description:
        Returns system health status for load balancers.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = health()
    """
    return _build_health_response()


def chat() -> tuple[Response, int]:
    """Accept a user message, moderate, classify, and respond.
    
    Detailed description:
        Main conversational endpoint.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = chat()
    """
    return _handle_chat()


def translate() -> tuple[Response, int]:
    """Translate text to a target language.
    
    Detailed description:
        Translation endpoint using TranslateService.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = translate()
    """
    return _handle_translate()


def translate_languages() -> Response:
    """Return supported languages for translation.
    
    Detailed description:
        Returns static list of supported languages.
        
    Args:
        None
        
    Returns:
        JSON response of languages.
        
    Raises:
        None
        
    Example:
        >>> res = translate_languages()
    """
    from services.translate_service import TranslateService

    svc: TranslateService = TranslateService.get_instance()
    return jsonify(svc.get_supported_languages())


def detect_language() -> tuple[Response, int]:
    """Detect the language of input text.
    
    Detailed description:
        Uses the TranslateService to determine text language.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = detect_language()
    """
    return _handle_detect_language()


def text_to_speech() -> tuple[Response, int]:
    """Convert text to speech audio.
    
    Detailed description:
        Returns base64 encoded audio for given text.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = text_to_speech()
    """
    return _handle_tts()


def news_search() -> tuple[Response, int]:
    """Search for election-related news.
    
    Detailed description:
        Returns Custom Search API results.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = news_search()
    """
    return _handle_news_search()


def create_session() -> Response:
    """Create a new anonymous session for persistence.
    
    Detailed description:
        Initialises a Firebase document for history tracking.
        
    Args:
        None
        
    Returns:
        JSON response with session ID.
        
    Raises:
        None
        
    Example:
        >>> res = create_session()
    """
    from services.firebase_service import FirebaseService

    svc: FirebaseService = FirebaseService.get_instance()
    return jsonify(svc.create_session())


def save_quiz_score(session_id: str) -> tuple[Response, int]:
    """Save a quiz score for a session.
    
    Detailed description:
        Persists a quiz result to the database.
        
    Args:
        session_id: The session ID from the URL.
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = save_quiz_score("uuid")
    """
    return _handle_quiz_score(session_id)


def topics() -> Response:
    """Return the list of supported election topics.
    
    Detailed description:
        Returns static list of vertex classification topics.
        
    Args:
        None
        
    Returns:
        JSON response of topics.
        
    Raises:
        None
        
    Example:
        >>> res = topics()
    """
    from services.vertex_service import ELECTION_TOPICS

    return jsonify({"topics": ELECTION_TOPICS, "success": True})


def quiz_question() -> tuple[Response, int]:
    """Generate a random election quiz question.
    
    Detailed description:
        Prompts Gemini to create a quiz.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = quiz_question()
    """
    return _handle_quiz_question()


def timeline() -> tuple[Response, int]:
    """Retrieve an election timeline for a country.
    
    Detailed description:
        Uses Gemini to generate election phases.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = timeline()
    """
    return _handle_timeline()


def map_endpoint() -> tuple[Response, int]:
    """Return map configuration or fallback embed URL.
    
    Detailed description:
        Provides mapping details based on query.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = map_endpoint()
    """
    return _handle_map()


def not_found(error: Exception) -> tuple[Response, int]:
    """Handle 404 errors with JSON response.
    
    Detailed description:
        Returns a standardised JSON error for undefined routes.
        
    Args:
        error: The caught exception.
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = not_found(Exception())
    """
    return jsonify({"error": "Resource not found.", "success": False}), HTTP_404_NOT_FOUND


def payload_too_large(error: Exception) -> tuple[Response, int]:
    """Handle oversized request payloads.
    
    Detailed description:
        Logs a security event and returns a 413 error.
        
    Args:
        error: The caught exception.
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = payload_too_large(Exception())
    """
    log_security_event("PAYLOAD_TOO_LARGE", f"ip={request.remote_addr}")
    return (
        jsonify({"error": "Request payload exceeds limit.", "success": False}),
        HTTP_413_PAYLOAD_TOO_LARGE,
    )


def rate_limited(error: Exception) -> tuple[Response, int]:
    """Handle rate limit exceeded errors.
    
    Detailed description:
        Returns a 429 error when request thresholds are breached.
        
    Args:
        error: The caught exception.
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = rate_limited(Exception())
    """
    return jsonify({"error": "Too many requests.", "success": False}), HTTP_429_TOO_MANY_REQUESTS


def server_error(error: Exception) -> tuple[Response, int]:
    """Handle internal server errors.
    
    Detailed description:
        Returns a generic 500 error payload.
        
    Args:
        error: The caught exception.
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = server_error(Exception())
    """
    return (
        jsonify(
            {"error": "An internal server error occurred.", "success": False}
        ),
        HTTP_500_INTERNAL_SERVER_ERROR,
    )


def create_app() -> Flask:
    """Create and configure the Flask application.

    Detailed description:
        Factory function that initialises the app, configures it,
        attaches middleware and registers routes.
        
    Args:
        None

    Returns:
        Configured Flask application instance.
        
    Raises:
        None
        
    Example:
        >>> app = create_app()
    """
    application: Flask = Flask(__name__)
    cfg: AppConfig = AppConfig()
    application.config["SECRET_KEY"] = cfg.flask_secret_key
    application.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    _setup_security_and_cors(application, cfg)
    limiter: Limiter = _setup_rate_limiter(application)
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

    Detailed description:
        Fired by Flask-Limiter when a client breaches the configured limit.
        
    Args:
        limit: The rate limit rule that was breached.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _on_rate_limit_breach("10 per minute")
    """
    log_security_event("RATE_LIMIT_HIT", f"limit={limit}")


# ---------------------------------------------------------------------------
# Route handler functions (Controller layer)
# ---------------------------------------------------------------------------


def _build_health_response() -> tuple[Response, int]:
    """Build the health check response payload.

    Detailed description:
        Checks all environment variables and returns service states.
        
    Args:
        None

    Returns:
        Tuple of JSON health status and HTTP 200.
        
    Raises:
        None
        
    Example:
        >>> res, status = _build_health_response()
    """
    services_status: dict[str, str] = {
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
        HTTP_200_OK,
    )


def _handle_chat() -> tuple[Response, int]:
    """Process a chat request through moderation, classification, and Gemini.

    Detailed description:
        Validates the request, performs Vertex moderation and classification, 
        calls Gemini, and triggers async persistence.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_chat()
    """
    from services.gemini_service import GeminiElectionAssistant
    from services.vertex_service import VertexService

    data: dict[str, Any] | None = request.get_json(silent=True)
    if not data or not data.get("message"):
        return (
            jsonify(
                {"error": "A 'message' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    raw_message: str = str(data.get("message", ""))
    user_message: str = sanitise_input(raw_message)
    if not user_message:
        return (
            jsonify(
                {"error": "A 'message' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    if len(user_message) > MAX_MESSAGE_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Message exceeds the {MAX_MESSAGE_LENGTH:,}-character limit.",
                    "success": False,
                }
            ),
            HTTP_400_BAD_REQUEST,
        )

    session_id: str = sanitise_input(str(data.get("session_id", "")))

    # 1. Content moderation
    vertex: VertexService = VertexService.get_instance()
    moderation: dict[str, Any] = vertex.moderate_content(user_message)
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
            HTTP_400_BAD_REQUEST,
        )

    # 2. Topic classification
    classification: dict[str, Any] = vertex.classify_topic(user_message)

    # 3. Gemini response
    try:
        gemini: GeminiElectionAssistant = GeminiElectionAssistant()
        result: dict[str, Any] = gemini.chat(user_message)
    except (ValueError, KeyError, ConnectionError, RuntimeError, TypeError) as exc:
        logger.error("Gemini chat failed: %s", exc)
        return (
            jsonify(
                {
                    "error": "AI service temporarily unavailable. Please try again.",
                    "success": False,
                }
            ),
            HTTP_500_INTERNAL_SERVER_ERROR,
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
        HTTP_200_OK,
    )


def _persist_chat(
    session_id: str,
    user_message: str,
    result: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    """Persist chat messages to Firestore (fire-and-forget).

    Detailed description:
        Saves user and model messages to Firebase if available.
        
    Args:
        session_id: The session identifier.
        user_message: The user's message.
        result: The Gemini response dictionary.
        classification: The topic classification result.
        
    Returns:
        None
        
    Raises:
        None
        
    Example:
        >>> _persist_chat("uuid", "Hi", {}, {})
    """
    if not session_id:
        return
    try:
        from services.firebase_service import FirebaseService

        fb: FirebaseService = FirebaseService.get_instance()
        fb.save_message(session_id, "user", user_message)
        fb.save_message(
            session_id,
            "model",
            result.get("response", ""),
            {"topic": classification["topic"]},
        )
    except (ValueError, KeyError, ConnectionError, RuntimeError, TypeError):
        logger.warning("Firebase persistence failed — continuing without.")


def _handle_translate() -> tuple[Response, int]:
    """Process a translation request.

    Detailed description:
        Validates request JSON and calls the TranslateService.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_translate()
    """
    from services.translate_service import TranslateService

    data: dict[str, Any] | None = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    req: TranslateRequest = TranslateRequest(
        text=sanitise_input(str(data["text"])),
        target_language=sanitise_input(str(data.get("target_language", "en"))),
        source_language=data.get("source_language"),
    )
    if not req.text:
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    svc: TranslateService = TranslateService.get_instance()
    result: dict[str, Any] = svc.translate_text(
        req.text, req.target_language, req.source_language
    )
    status_code: int = HTTP_200_OK if result.get("success") else HTTP_500_INTERNAL_SERVER_ERROR
    return jsonify(result), status_code


def _handle_detect_language() -> tuple[Response, int]:
    """Process a language detection request.

    Detailed description:
        Validates request and delegates to TranslateService language detection.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_detect_language()
    """
    from services.translate_service import TranslateService

    data: dict[str, Any] | None = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    svc: TranslateService = TranslateService.get_instance()
    return jsonify(svc.detect_language(sanitise_input(str(data["text"])))), HTTP_200_OK


def _handle_tts() -> tuple[Response, int]:
    """Process a text-to-speech request.

    Detailed description:
        Validates request and calls TTSService.
        
    Args:
        None
        
    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_tts()
    """
    from services.tts_service import TTSService

    data: dict[str, Any] | None = request.get_json(silent=True)
    if not data or not data.get("text"):
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    req: TTSRequest = TTSRequest(
        text=sanitise_input(str(data["text"])),
        language=sanitise_input(str(data.get("language", "en"))),
        speaking_rate=float(data.get("speaking_rate", 1.0)),
    )
    if not req.text:
        return (
            jsonify(
                {"error": "A 'text' field is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    svc: TTSService = TTSService.get_instance()
    result: dict[str, Any] = svc.synthesize(req.text, req.language, req.speaking_rate)
    status_code: int = HTTP_200_OK if result.get("success") else HTTP_500_INTERNAL_SERVER_ERROR
    return jsonify(result), status_code


def _handle_news_search() -> tuple[Response, int]:
    """Process a news search request.

    Detailed description:
        Retrieves query args and calls the SearchService.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_news_search()
    """
    from services.search_service import SearchService

    query: str = sanitise_input(request.args.get("query", ""))
    if not query:
        return (
            jsonify(
                {"error": "A 'query' parameter is required.", "success": False}
            ),
            HTTP_400_BAD_REQUEST,
        )

    num: int = min(int(request.args.get("num", "5")), 10)

    svc: SearchService = SearchService.get_instance()
    result: dict[str, Any] = svc.search_news(query, num)
    status_code: int = HTTP_200_OK if result.get("success") else HTTP_500_INTERNAL_SERVER_ERROR
    return jsonify(result), status_code


def _handle_quiz_score(session_id: str) -> tuple[Response, int]:
    """Process a quiz score save request.

    Detailed description:
        Delegates the saving of the score to FirebaseService.
        
    Args:
        session_id: The session identifier from the URL.

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_quiz_score("uuid")
    """
    from services.firebase_service import FirebaseService

    data: dict[str, Any] | None = request.get_json(silent=True)
    if not data:
        return (
            jsonify({"error": "Request body is required.", "success": False}),
            HTTP_400_BAD_REQUEST,
        )

    req: QuizScoreRequest = QuizScoreRequest(
        score=int(data.get("score", 0)),
        total=int(data.get("total", 0)),
        topic=sanitise_input(str(data.get("topic", ""))),
    )
    clean_session: str = sanitise_input(session_id)

    svc: FirebaseService = FirebaseService.get_instance()
    result: dict[str, Any] = svc.save_quiz_score(clean_session, req.score, req.total, req.topic)
    return jsonify(result), HTTP_200_OK


FALLBACK_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "What is the minimum voting age in most democratic countries?",
        "options": ["16", "18", "21", "25"],
        "correct_answer": "18",
        "explanation": "In the majority of democratic nations, the legal voting age is 18.",
    },
    {
        "question": "Which of these is NOT a primary function of an election commission?",
        "options": ["Registering voters", "Counting ballots", "Passing new laws", "Setting election dates"],
        "correct_answer": "Passing new laws",
        "explanation": "Election commissions administer elections, passing laws is the duty of the legislature.",
    },
]


def _handle_quiz_question() -> tuple[Response, int]:
    """Generate a quiz question using Gemini.

    Detailed description:
        Calls Gemini to create a quiz. If that fails, uses a random fallback question.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_quiz_question()
    """
    from services.gemini_service import GeminiElectionAssistant

    try:
        gemini: GeminiElectionAssistant = GeminiElectionAssistant()
        result: dict[str, Any] = gemini.get_quiz_question()
        return jsonify({**result, "success": True}), HTTP_200_OK
    except (ValueError, KeyError, ConnectionError, RuntimeError, TypeError) as exc:
        logger.error("Quiz generation failed: %s", exc)
        question: dict[str, Any] = random.choice(FALLBACK_QUESTIONS)
        return jsonify({**question, "success": True, "fallback": True}), HTTP_200_OK


FALLBACK_TIMELINES: dict[str, dict[str, Any]] = {
    "india": {
        "country": "India",
        "timeline": [
            {
                "phase": "Announcement",
                "description": "Election Commission announces the schedule.",
                "approximate_timeframe": "45-60 days before voting",
            },
        ],
        "summary": "India's general elections are managed by the Election Commission over several phases.",
    },
}


def _handle_timeline() -> tuple[Response, int]:
    """Retrieve an election timeline using Gemini.

    Detailed description:
        Calls Gemini to retrieve timelines, with fallback maps if it fails.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_timeline()
    """
    from services.gemini_service import GeminiElectionAssistant

    data: dict[str, Any] | None = request.get_json(silent=True)
    if data is None:
        country: str = "India"
    else:
        country = sanitise_input(str(data.get("country", "India")))

    try:
        gemini: GeminiElectionAssistant = GeminiElectionAssistant()
        result: dict[str, Any] = gemini.get_timeline(country)
        return jsonify({**result, "success": True}), HTTP_200_OK
    except (ValueError, KeyError, ConnectionError, RuntimeError, TypeError) as exc:
        logger.error("Timeline generation failed: %s", exc)
        key: str = country.lower()
        if key not in FALLBACK_TIMELINES:
            key = "india"
        return (
            jsonify(
                {**FALLBACK_TIMELINES[key], "success": True, "fallback": True}
            ),
            HTTP_200_OK,
        )


def _handle_map() -> tuple[Response, int]:
    """Handle map endpoint request with fallback.

    Detailed description:
        Provides mapping details based on the query. If no API key is set,
        it uses OpenStreetMap fallback.
        
    Args:
        None

    Returns:
        Tuple of JSON response and HTTP status code.
        
    Raises:
        None
        
    Example:
        >>> res, status = _handle_map()
    """
    api_key: str = os.environ.get("GOOGLE_MAPS_API_KEY", "")

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
            HTTP_200_OK,
        )

    query: str = sanitise_input(request.args.get("q", "polling stations near me"))

    encoded_query: str = urllib.parse.quote(query)
    embed_url: str = f"https://www.google.com/maps/embed/v1/place?key={api_key}&q={encoded_query}"
    return (
        jsonify(
            {"provider": "google", "embed_url": embed_url, "success": True}
        ),
        HTTP_200_OK,
    )


# ---------------------------------------------------------------------------
# Entrypoint (used by gunicorn: `gunicorn main:app`)
# ---------------------------------------------------------------------------
app: Flask = create_app()

if __name__ == "__main__":
    port: int = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    logger.info("Starting development server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
