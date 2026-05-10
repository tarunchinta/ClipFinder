"""Authentication backend configuration."""

from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)

from app.config import get_settings

settings = get_settings()

# Cookie transport for web browser sessions
cookie_transport = CookieTransport(
    cookie_name="clipfinder_auth",
    cookie_max_age=settings.jwt_lifetime_seconds,
    cookie_secure=False,  # Set True in production with HTTPS
    cookie_httponly=True,
    cookie_samesite="lax",
)

# Bearer transport for API access
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """Create JWT strategy for authentication."""
    return JWTStrategy(
        secret=settings.jwt_secret,
        lifetime_seconds=settings.jwt_lifetime_seconds,
        algorithm=settings.jwt_algorithm,
    )


# Primary auth backend using cookies (for web app)
auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# Secondary auth backend using bearer tokens (for API)
bearer_auth_backend = AuthenticationBackend(
    name="bearer",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)


# Note: current_active_user and current_user_optional are created in users.py
# after fastapi_users is instantiated. Import them from app.auth.users

