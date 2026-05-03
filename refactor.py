with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

new_create_app = '''
def _setup_security_and_cors(application, cfg):
    """Setup security headers and CORS."""
    Compress(application)
    Talisman(application, content_security_policy=CSP_DIRECTIVES, force_https=False)
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
    application.add_url_rule("/api/chat", view_func=limiter.limit(RATE_LIMIT_CHAT)(require_json(chat)), methods=["POST"])
    application.add_url_rule("/api/translate", view_func=limiter.limit(RATE_LIMIT_TRANSLATE)(require_json(translate)), methods=["POST"])
    application.add_url_rule("/api/translate/languages", view_func=limiter.limit(RATE_LIMIT_LANGUAGES)(translate_languages))
    application.add_url_rule("/api/translate/detect", view_func=limiter.limit(RATE_LIMIT_DETECT)(require_json(detect_language)), methods=["POST"])
    application.add_url_rule("/api/tts", view_func=limiter.limit(RATE_LIMIT_TTS)(require_json(text_to_speech)), methods=["POST"])
    application.add_url_rule("/api/news", view_func=limiter.limit(RATE_LIMIT_NEWS)(news_search))
    application.add_url_rule("/api/session", view_func=limiter.limit(RATE_LIMIT_SESSION)(create_session), methods=["POST"])
    application.add_url_rule("/api/session/<session_id>/quiz", view_func=limiter.limit(RATE_LIMIT_QUIZ)(require_json(save_quiz_score)), methods=["POST"])
    application.add_url_rule("/api/topics", view_func=limiter.limit(RATE_LIMIT_TOPICS)(topics))
    application.add_url_rule("/api/quiz/question", view_func=limiter.limit(RATE_LIMIT_QUIZ)(quiz_question))
    application.add_url_rule("/api/timeline", view_func=limiter.limit(RATE_LIMIT_TOPICS)(require_json(timeline)), methods=["POST"])
    application.add_url_rule("/api/map", view_func=limiter.limit(RATE_LIMIT_DEFAULT)(map_endpoint))

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
    return jsonify({"error": f"Request payload exceeds limit.", "success": False}), 413

def rate_limited(error: Exception):
    """Handle rate limit exceeded errors."""
    return jsonify({"error": "Too many requests.", "success": False}), 429

def server_error(error: Exception):
    """Handle internal server errors."""
    return jsonify({"error": "An internal server error occurred.", "success": False}), 500

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
'''

start_idx = content.find("def create_app() -> Flask:")
end_idx = content.find(
    'logger.info("Flask application created with all Google Services.")'
)
if end_idx != -1:
    end_idx = content.find("return application", end_idx) + len(
        "return application"
    )

    new_content = content[:start_idx] + new_create_app + content[end_idx:]

    with open("main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
