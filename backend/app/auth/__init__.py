"""Authentication module."""

from app.auth.backend import auth_backend
from app.auth.users import get_user_manager, fastapi_users, current_active_user, current_user_optional
from app.auth.google import google_oauth_client

__all__ = [
    "auth_backend",
    "current_active_user", 
    "current_user_optional",
    "get_user_manager",
    "fastapi_users",
    "google_oauth_client",
]

