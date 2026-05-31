"""FastAPI web UI for code-review-agent (optional ``web`` extra)."""

from code_review_agent.webapp.app import create_app, main

__all__ = ["create_app", "main"]
