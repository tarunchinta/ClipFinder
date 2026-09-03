"""WorkOS AuthKit session helpers for MCP OAuth (Google social login)."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Request, Response
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from workos import WorkOSClient
from workos.session import AuthenticateWithSessionCookieFailureReason, Session

from app.config import get_settings
from app.models.user import User

logger = logging.getLogger(__name__)

WOS_SESSION_COOKIE = "wos_session"
PENDING_OAUTH_COOKIE = "mcp_oauth_pending"
PENDING_OAUTH_AUD = "clipfinder-mcp-pending"
PENDING_OAUTH_LIFETIME_SECONDS = 600


def _unusable_password_hash() -> str:
    """Placeholder hash for WorkOS-only users (no password login)."""
    return bcrypt.hashpw(secrets.token_urlsafe(32).encode(), bcrypt.gensalt()).decode()


def workos_configured() -> bool:
    settings = get_settings()
    return bool(
        settings.workos_api_key
        and settings.workos_client_id
        and settings.workos_cookie_password
    )


def get_workos_client() -> WorkOSClient:
    settings = get_settings()
    return WorkOSClient(
        api_key=settings.workos_api_key,
        client_id=settings.workos_client_id,
    )


def _cookie_password() -> str:
    return get_settings().workos_cookie_password


def _workos_user_to_dict(workos_user: Any) -> dict[str, Any]:
    if isinstance(workos_user, dict):
        return workos_user
    if hasattr(workos_user, "to_dict"):
        return workos_user.to_dict()
    return {
        "email": getattr(workos_user, "email", None),
        "first_name": getattr(workos_user, "first_name", None),
        "last_name": getattr(workos_user, "last_name", None),
        "profile_picture_url": getattr(workos_user, "profile_picture_url", None),
    }


def _seal_session_from_auth_response(
    *,
    access_token: str,
    refresh_token: str,
    user: dict[str, Any],
    impersonator: dict[str, Any] | None = None,
    cookie_password: str,
) -> str:
    """Seal auth tokens into a cookie.

    WorkOS 6+ ships ``seal_session_from_auth_response``; 5.x exposes the
    same Fernet helper as ``Session.seal_data``.
    """
    payload: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user,
    }
    if impersonator is not None:
        payload["impersonator"] = impersonator
    return Session.seal_data(payload, cookie_password)


def seal_auth_response(auth_response: Any) -> str:
    """Seal a WorkOS authenticate_with_code response into a cookie value."""
    user = _workos_user_to_dict(auth_response.user)
    impersonator = None
    if getattr(auth_response, "impersonator", None) is not None:
        imp = auth_response.impersonator
        impersonator = imp.to_dict() if hasattr(imp, "to_dict") else imp
    return _seal_session_from_auth_response(
        access_token=auth_response.access_token,
        refresh_token=auth_response.refresh_token,
        user=user,
        impersonator=impersonator,
        cookie_password=_cookie_password(),
    )


def set_wos_session_cookie(response: Response, sealed_session: str) -> None:
    settings = get_settings()
    response.set_cookie(
        WOS_SESSION_COOKIE,
        sealed_session,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=30 * 24 * 3600,
    )


def clear_wos_session_cookie(response: Response) -> None:
    response.delete_cookie(WOS_SESSION_COOKIE)


def get_authkit_authorization_url(*, state: str | None = None) -> str:
    settings = get_settings()
    client = get_workos_client()
    kwargs: dict[str, Any] = {
        "provider": "authkit",
        "redirect_uri": settings.workos_callback_uri,
    }
    if state:
        kwargs["state"] = state
    return client.user_management.get_authorization_url(**kwargs)


def _load_session(sealed_session: str) -> Session:
    return get_workos_client().user_management.load_sealed_session(
        session_data=sealed_session,
        cookie_password=_cookie_password(),
    )


class WorkOSAuthResult:
    """Result of authenticating a WorkOS sealed session."""

    __slots__ = ("authenticated", "user", "sealed_session", "reason")

    def __init__(
        self,
        *,
        authenticated: bool,
        user: Any = None,
        sealed_session: str | None = None,
        reason: str | None = None,
    ):
        self.authenticated = authenticated
        self.user = user
        self.sealed_session = sealed_session
        self.reason = reason


def authenticate_wos_session(sealed_session: str | None) -> WorkOSAuthResult:
    if not sealed_session:
        return WorkOSAuthResult(
            authenticated=False,
            reason="no_session_cookie_provided",
        )
    try:
        session = _load_session(sealed_session)
        auth_response = session.authenticate()
        if auth_response.authenticated and auth_response.user:
            return WorkOSAuthResult(
                authenticated=True,
                user=auth_response.user,
                sealed_session=sealed_session,
            )
        if auth_response.reason == AuthenticateWithSessionCookieFailureReason.NO_SESSION_COOKIE_PROVIDED:
            return WorkOSAuthResult(authenticated=False, reason=str(auth_response.reason))

        refresh_result = session.refresh()
        if refresh_result.authenticated and refresh_result.user:
            return WorkOSAuthResult(
                authenticated=True,
                user=refresh_result.user,
                sealed_session=refresh_result.sealed_session,
            )
        return WorkOSAuthResult(authenticated=False, reason="session_expired")
    except Exception as exc:
        logger.warning("WorkOS session authenticate failed: %s", exc)
        return WorkOSAuthResult(authenticated=False, reason="invalid_session")


def get_workos_user_from_request(request: Request) -> WorkOSAuthResult:
    return authenticate_wos_session(request.cookies.get(WOS_SESSION_COOKIE))


def get_logout_url(sealed_session: str | None) -> str | None:
    if not sealed_session:
        return None
    try:
        session = _load_session(sealed_session)
        return session.get_logout_url()
    except Exception as exc:
        logger.warning("WorkOS get_logout_url failed: %s", exc)
        return None


def _sign_pending_oauth(payload: dict) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    claims = {
        **payload,
        "aud": PENDING_OAUTH_AUD,
        "iss": settings.app_url,
        "iat": now,
        "exp": now + timedelta(seconds=PENDING_OAUTH_LIFETIME_SECONDS),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_pending_oauth(token: str) -> dict | None:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=PENDING_OAUTH_AUD,
        )
    except JWTError:
        return None


def set_pending_oauth_cookie(response: Response, params: dict[str, str]) -> None:
    settings = get_settings()
    token = _sign_pending_oauth(params)
    response.set_cookie(
        PENDING_OAUTH_COOKIE,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=PENDING_OAUTH_LIFETIME_SECONDS,
    )


def get_pending_oauth_params(request: Request) -> dict[str, str] | None:
    token = request.cookies.get(PENDING_OAUTH_COOKIE)
    if not token:
        return None
    claims = _decode_pending_oauth(token)
    if not claims:
        return None
    return {
        k: claims[k]
        for k in ("client_id", "redirect_uri", "state", "code_challenge")
        if k in claims and claims[k]
    }


def clear_pending_oauth_cookie(response: Response) -> None:
    response.delete_cookie(PENDING_OAUTH_COOKIE)


def pending_oauth_query(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    query: dict[str, str] = {
        "response_type": "code",
        "code_challenge_method": "S256",
        **params,
    }
    return urlencode(query)


async def upsert_user_from_workos(
    session: AsyncSession,
    workos_user: Any,
) -> User:
    """Find or create a Distill User from a WorkOS AuthKit profile."""
    profile = _workos_user_to_dict(workos_user)
    email = profile.get("email")
    if not email:
        raise ValueError("WorkOS user has no email")

    user = (
        await session.execute(select(User).where(User.email == email))
    ).unique().scalar_one_or_none()

    first = profile.get("first_name") or ""
    last = profile.get("last_name") or ""
    display_name = f"{first} {last}".strip()
    avatar_url = profile.get("profile_picture_url")

    if user:
        if display_name:
            user.display_name = display_name
        if avatar_url:
            user.avatar_url = avatar_url
        user.last_login_at = datetime.utcnow()
        await session.commit()
        await session.refresh(user)
        return user

    user = User(
        email=email,
        hashed_password=_unusable_password_hash(),
        is_active=True,
        is_verified=True,
        is_superuser=False,
        display_name=display_name or None,
        avatar_url=avatar_url,
        last_login_at=datetime.utcnow(),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    logger.info("Created Distill user from WorkOS AuthKit: %s", email)
    return user


async def resolve_distill_user(
    session: AsyncSession,
    workos_user: Any,
) -> User:
    return await upsert_user_from_workos(session, workos_user)
