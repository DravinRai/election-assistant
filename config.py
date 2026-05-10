"""
Application configuration constants for the Election Education Assistant.

This module provides centralized access to all magic strings, numbers, and 
environment variable keys. Import from this module instead of using hardcoded values.

Example:
    >>> from config import AppConfig
    >>> cfg = AppConfig()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final, List, Dict

__all__: List[str] = [
    "ENV_GOOGLE_API_KEY",
    "ENV_GOOGLE_CLOUD_PROJECT",
    "ENV_VERTEX_LOCATION",
    "ENV_TRANSLATE_API_KEY",
    "ENV_TTS_API_KEY",
    "ENV_SEARCH_API_KEY",
    "ENV_SEARCH_ENGINE_ID",
    "ENV_GA_MEASUREMENT_ID",
    "ENV_MAPS_API_KEY",
    "ENV_FIREBASE_CREDENTIALS",
    "ENV_FLASK_SECRET_KEY",
    "ENV_ALLOWED_ORIGINS",
    "ENV_PORT",
    "ENV_SEARCH_TIMEOUT",
    "APP_VERSION",
    "DEFAULT_PORT",
    "MAX_CONTENT_LENGTH",
    "RATE_LIMIT_DEFAULT",
    "RATE_LIMIT_HOURLY",
    "RATE_LIMIT_CHAT",
    "RATE_LIMIT_TRANSLATE",
    "RATE_LIMIT_TTS",
    "RATE_LIMIT_NEWS",
    "RATE_LIMIT_SESSION",
    "RATE_LIMIT_QUIZ",
    "RATE_LIMIT_TOPICS",
    "RATE_LIMIT_LANGUAGES",
    "RATE_LIMIT_DETECT",
    "GEMINI_MODEL_NAME",
    "GEMINI_TEMPERATURE",
    "GEMINI_RESPONSE_MIME",
    "GEMINI_HISTORY_LIMIT",
    "GEMINI_MAX_RETRIES",
    "GEMINI_RETRY_DELAY",
    "GEMINI_RETRY_BASE",
    "MAX_MESSAGE_LENGTH",
    "MAX_TTS_TEXT_LENGTH",
    "SEARCH_MAX_RESULTS",
    "SEARCH_MAX_RESULTS_UPPER",
    "SEARCH_CACHE_TTL_SECONDS",
    "SEARCH_BASE_URL",
    "SEARCH_TIMEOUT_DEFAULT",
    "TTS_MAX_CACHE_SIZE",
    "VERTEX_MODEL_NAME",
    "VERTEX_DEFAULT_LOCATION",
    "GA_DEFAULT_MEASUREMENT_ID",
    "PAGINATION_DEFAULT_PAGE",
    "PAGINATION_DEFAULT_PER_PAGE",
    "PAGINATION_MAX_PER_PAGE",
    "CSP_DIRECTIVES",
    "CONTENT_TYPE_JSON",
    "CACHE_TTL_SECONDS",
    "DEFAULT_LANGUAGE",
    "DEFAULT_SPEAKING_RATE",
    "MIN_SPEAKING_RATE",
    "MAX_SPEAKING_RATE",
    "HISTORY_LIMIT_DEFAULT",
    "QUIZ_SCORES_LIMIT",
    "DEFAULT_CONFIDENCE",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
    "MIN_SEARCH_RESULTS",
    "HTTP_200_OK",
    "HTTP_400_BAD_REQUEST",
    "HTTP_404_NOT_FOUND",
    "HTTP_413_PAYLOAD_TOO_LARGE",
    "HTTP_415_UNSUPPORTED_MEDIA_TYPE",
    "HTTP_429_TOO_MANY_REQUESTS",
    "HTTP_500_INTERNAL_SERVER_ERROR",
    "AppConfig",
]


# ---------------------------------------------------------------------------
# Environment variable keys (single source of truth)
# ---------------------------------------------------------------------------
ENV_GOOGLE_API_KEY: Final[str] = "GOOGLE_API_KEY"
ENV_GOOGLE_CLOUD_PROJECT: Final[str] = "GOOGLE_CLOUD_PROJECT"
ENV_VERTEX_LOCATION: Final[str] = "VERTEX_LOCATION"
ENV_TRANSLATE_API_KEY: Final[str] = "GOOGLE_TRANSLATE_API_KEY"
ENV_TTS_API_KEY: Final[str] = "GOOGLE_TTS_API_KEY"
ENV_SEARCH_API_KEY: Final[str] = "GOOGLE_SEARCH_API_KEY"
ENV_SEARCH_ENGINE_ID: Final[str] = "GOOGLE_SEARCH_ENGINE_ID"
ENV_GA_MEASUREMENT_ID: Final[str] = "GA_MEASUREMENT_ID"
ENV_MAPS_API_KEY: Final[str] = "GOOGLE_MAPS_API_KEY"
ENV_FIREBASE_CREDENTIALS: Final[str] = "FIREBASE_CREDENTIALS_PATH"
ENV_FLASK_SECRET_KEY: Final[str] = "FLASK_SECRET_KEY"
ENV_ALLOWED_ORIGINS: Final[str] = "ALLOWED_ORIGINS"
ENV_PORT: Final[str] = "PORT"
ENV_SEARCH_TIMEOUT: Final[str] = "SEARCH_TIMEOUT_SECONDS"

# ---------------------------------------------------------------------------
# Application version
# ---------------------------------------------------------------------------
APP_VERSION: Final[str] = "2.1.0"

# ---------------------------------------------------------------------------
# Flask / HTTP
# ---------------------------------------------------------------------------
DEFAULT_PORT: Final[int] = 8080
MAX_CONTENT_LENGTH: Final[int] = 10 * 1024  # 10 KB payload limit
RATE_LIMIT_DEFAULT: Final[str] = "200 per day"
RATE_LIMIT_HOURLY: Final[str] = "50 per hour"
RATE_LIMIT_CHAT: Final[str] = "10 per minute"
RATE_LIMIT_TRANSLATE: Final[str] = "60 per minute"
RATE_LIMIT_TTS: Final[str] = "20 per minute"
RATE_LIMIT_NEWS: Final[str] = "30 per minute"
RATE_LIMIT_SESSION: Final[str] = "10 per minute"
RATE_LIMIT_QUIZ: Final[str] = "30 per minute"
RATE_LIMIT_TOPICS: Final[str] = "60 per minute"
RATE_LIMIT_LANGUAGES: Final[str] = "120 per minute"
RATE_LIMIT_DETECT: Final[str] = "60 per minute"

# ---------------------------------------------------------------------------
# HTTP Status Codes & Content Types
# ---------------------------------------------------------------------------
HTTP_200_OK: Final[int] = 200
HTTP_400_BAD_REQUEST: Final[int] = 400
HTTP_404_NOT_FOUND: Final[int] = 404
HTTP_413_PAYLOAD_TOO_LARGE: Final[int] = 413
HTTP_415_UNSUPPORTED_MEDIA_TYPE: Final[int] = 415
HTTP_429_TOO_MANY_REQUESTS: Final[int] = 429
HTTP_500_INTERNAL_SERVER_ERROR: Final[int] = 500
CONTENT_TYPE_JSON: Final[str] = "application/json"

# ---------------------------------------------------------------------------
# Gemini AI
# ---------------------------------------------------------------------------
GEMINI_MODEL_NAME: Final[str] = "gemini-2.0-flash"
GEMINI_TEMPERATURE: Final[float] = 0.2
GEMINI_RESPONSE_MIME: Final[str] = "application/json"
GEMINI_HISTORY_LIMIT: Final[int] = 10
GEMINI_MAX_RETRIES: Final[int] = 3
GEMINI_RETRY_DELAY: Final[float] = 1.0
GEMINI_RETRY_BASE: Final[float] = 2.0

# ---------------------------------------------------------------------------
# Chat validation
# ---------------------------------------------------------------------------
MAX_MESSAGE_LENGTH: Final[int] = 2000
MAX_TTS_TEXT_LENGTH: Final[int] = 5000

# ---------------------------------------------------------------------------
# Search service
# ---------------------------------------------------------------------------
SEARCH_MAX_RESULTS: Final[int] = 5
SEARCH_MAX_RESULTS_UPPER: Final[int] = 10
MIN_SEARCH_RESULTS: Final[int] = 1
SEARCH_CACHE_TTL_SECONDS: Final[int] = 300  # 5 minutes
SEARCH_BASE_URL: Final[str] = "https://www.googleapis.com/customsearch/v1"
SEARCH_TIMEOUT_DEFAULT: Final[int] = 10

# ---------------------------------------------------------------------------
# TTS service & Language defaults
# ---------------------------------------------------------------------------
TTS_MAX_CACHE_SIZE: Final[int] = 100
DEFAULT_LANGUAGE: Final[str] = "en"
DEFAULT_SPEAKING_RATE: Final[float] = 1.0
MIN_SPEAKING_RATE: Final[float] = 0.25
MAX_SPEAKING_RATE: Final[float] = 4.0

# ---------------------------------------------------------------------------
# Global App Defaults
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS: Final[int] = 300
HISTORY_LIMIT_DEFAULT: Final[int] = 50
QUIZ_SCORES_LIMIT: Final[int] = 20
DEFAULT_CONFIDENCE: Final[float] = 0.5
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Vertex AI
# ---------------------------------------------------------------------------
VERTEX_MODEL_NAME: Final[str] = "gemini-1.5-pro"
VERTEX_DEFAULT_LOCATION: Final[str] = "us-central1"

# ---------------------------------------------------------------------------
# GA default placeholder
# ---------------------------------------------------------------------------
GA_DEFAULT_MEASUREMENT_ID: Final[str] = "G-XXXXXXXXXX"

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
PAGINATION_DEFAULT_PAGE: Final[int] = 1
PAGINATION_DEFAULT_PER_PAGE: Final[int] = 20
PAGINATION_MAX_PER_PAGE: Final[int] = 100

# ---------------------------------------------------------------------------
# Security — Content-Security-Policy directives
# ---------------------------------------------------------------------------
CSP_DIRECTIVES: Final[Dict[str, str]] = {
    "default-src": "'self'",
    "script-src": (
        "'self' 'unsafe-inline' "
        "https://www.googletagmanager.com "
        "https://www.google-analytics.com "
        "https://maps.googleapis.com"
    ),
    "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src": "'self' https://fonts.gstatic.com",
    "img-src": "'self' data: https: blob:",
    "connect-src": (
        "'self' "
        "https://www.google-analytics.com "
        "https://analytics.google.com "
        "https://region1.google-analytics.com"
    ),
    "frame-src": (
        "'self' " "https://www.google.com " "https://maps.google.com"
    ),
    "media-src": "'self' blob: data:",
}


# ---------------------------------------------------------------------------
# Dataclass for runtime-resolved config (reads env vars at construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration resolved from environment variables."""

    google_api_key: str = field(
        default_factory=lambda: os.environ.get(ENV_GOOGLE_API_KEY, "")
    )
    google_cloud_project: str = field(
        default_factory=lambda: os.environ.get(ENV_GOOGLE_CLOUD_PROJECT, "")
    )
    vertex_location: str = field(
        default_factory=lambda: os.environ.get(
            ENV_VERTEX_LOCATION, VERTEX_DEFAULT_LOCATION
        )
    )
    flask_secret_key: str = field(
        default_factory=lambda: os.environ.get(
            ENV_FLASK_SECRET_KEY, os.urandom(32).hex()
        )
    )
    ga_measurement_id: str = field(
        default_factory=lambda: os.environ.get(
            ENV_GA_MEASUREMENT_ID, GA_DEFAULT_MEASUREMENT_ID
        )
    )
    google_maps_api_key: str = field(
        default_factory=lambda: os.environ.get(ENV_MAPS_API_KEY, "")
    )
    port: int = field(
        default_factory=lambda: int(
            os.environ.get(ENV_PORT, str(DEFAULT_PORT))
        )
    )
    allowed_origins: str = field(
        default_factory=lambda: os.environ.get(ENV_ALLOWED_ORIGINS, "*")
    )
