"""
Firebase Firestore Service — optional persistence for the Election Assistant.

Provides session-scoped conversation history storage,
anonymous auth session management, and quiz score persistence.
Falls back gracefully when Firebase SDK is unavailable.

Author: Ankit Rai
Version: 2.1.0
Usage example:
    from services.firebase_service import FirebaseService
    svc = FirebaseService.get_instance()
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from config import ENV_FIREBASE_CREDENTIALS, ENV_GOOGLE_CLOUD_PROJECT

logger = logging.getLogger(__name__)

# Guard import
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    logger.info("Firebase SDK not installed — persistence disabled.")

__all__ = ["FirebaseService"]


class FirebaseService:
    """Manage Firestore persistence for conversations and quiz scores.

    Attributes:
        _db: Firestore client instance (or None).
        _project_id: Google Cloud project identifier.
    """

    _instance: FirebaseService | None = None

    def __init__(self) -> None:
        """Initialise with optional Firestore backend."""
        self._db = None
        self._project_id = os.environ.get(ENV_GOOGLE_CLOUD_PROJECT, "")
        self._credentials_path = os.environ.get(ENV_FIREBASE_CREDENTIALS, "")

        if _FIREBASE_AVAILABLE and self._project_id:
            self._init_firebase()
        else:
            logger.info("FirebaseService running in disabled mode.")

    def _init_firebase(self) -> None:
        """Attempt to initialise Firebase app and Firestore client."""
        try:
            try:
                firebase_admin.get_app()
            except ValueError:
                if self._credentials_path and os.path.exists(
                    self._credentials_path
                ):
                    cred = credentials.Certificate(self._credentials_path)
                    firebase_admin.initialize_app(
                        cred, {"projectId": self._project_id}
                    )
                else:
                    firebase_admin.initialize_app(
                        options={"projectId": self._project_id}
                    )
            self._db = firestore.client()
            logger.info(
                "FirebaseService initialised (project=%s).", self._project_id
            )
        except Exception:
            logger.exception(
                "Failed to initialise Firebase — persistence disabled."
            )

    @classmethod
    def get_instance(cls) -> FirebaseService:
        """Return the singleton FirebaseService instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_available(self) -> bool:
        """Whether the Firestore backend is connected."""
        return self._db is not None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self) -> dict[str, Any]:
        """Create an anonymous session.

        Returns:
            Dict with 'session_id', 'success', and 'persisted' fields.
        """
        session_id = str(uuid.uuid4())
        if not self._db:
            return {
                "session_id": session_id,
                "success": True,
                "persisted": False,
            }
        try:
            self._db.collection("sessions").document(session_id).set(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "message_count": 0,
                    "quiz_scores": [],
                }
            )
            return {
                "session_id": session_id,
                "success": True,
                "persisted": True,
            }
        except Exception as exc:
            logger.exception("Failed to create session: %s", exc)
            return {
                "session_id": session_id,
                "success": True,
                "persisted": False,
            }

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Save a chat message to the session's conversation history.

        Args:
            session_id: The session identifier.
            role: Message role ('user' or 'model').
            content: Message text content.
            metadata: Optional extra metadata dict.

        Returns:
            Dict with 'success' and 'persisted' fields.
        """
        if not self._db:
            return {"success": True, "persisted": False}
        try:
            msg_data = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            }
            self._db.collection("sessions").document(session_id).collection(
                "messages"
            ).add(msg_data)
            self._db.collection("sessions").document(session_id).update(
                {"message_count": firestore.Increment(1)}
            )
            return {"success": True, "persisted": True}
        except Exception as exc:
            logger.exception("Failed to save message: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> dict[str, Any]:
        """Retrieve conversation history for a session.

        Args:
            session_id: The session identifier.
            limit: Maximum number of messages to return.

        Returns:
            Dict with 'messages' list and 'success' flag.
        """
        if not self._db:
            return {"messages": [], "success": True, "persisted": False}
        try:
            docs = (
                self._db.collection("sessions")
                .document(session_id)
                .collection("messages")
                .order_by("timestamp")
                .limit(limit)
                .stream()
            )
            return {
                "messages": [doc.to_dict() for doc in docs],
                "success": True,
            }
        except Exception as exc:
            logger.exception("Failed to fetch history: %s", exc)
            return {"messages": [], "success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Quiz scores
    # ------------------------------------------------------------------

    def save_quiz_score(
        self, session_id: str, score: int, total: int, topic: str = ""
    ) -> dict[str, Any]:
        """Save a quiz score for the session.

        Args:
            session_id: The session identifier.
            score: Number of correct answers.
            total: Total number of questions.
            topic: The quiz topic identifier.

        Returns:
            Dict with 'success' and 'persisted' fields.
        """
        if not self._db:
            return {"success": True, "persisted": False}
        try:
            score_data = {
                "score": score,
                "total": total,
                "topic": topic,
                "percentage": (
                    round((score / total) * 100, 1) if total > 0 else 0
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._db.collection("sessions").document(session_id).collection(
                "quiz_scores"
            ).add(score_data)
            return {"success": True, "persisted": True}
        except Exception as exc:
            logger.exception("Failed to save quiz score: %s", exc)
            return {"success": False, "error": str(exc)}

    def get_quiz_scores(self, session_id: str) -> dict[str, Any]:
        """Retrieve quiz scores for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Dict with 'scores' list and 'success' flag.
        """
        if not self._db:
            return {"scores": [], "success": True, "persisted": False}
        try:
            docs = (
                self._db.collection("sessions")
                .document(session_id)
                .collection("quiz_scores")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(20)
                .stream()
            )
            return {"scores": [doc.to_dict() for doc in docs], "success": True}
        except Exception as exc:
            logger.exception("Failed to fetch quiz scores: %s", exc)
            return {"scores": [], "success": False, "error": str(exc)}
