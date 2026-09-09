"""Minimal OAuth 2.1 authorization server for remote MCP clients.

MCP clients run authorization-code + PKCE against these endpoints, using
Dynamic Client Registration (RFC 7591) or Client ID Metadata Documents.
The human signs in via WorkOS AuthKit (Google social login); a successful
login auto-approves the MCP connection (no separate consent screen). MCP
access tokens carry the Distill user id in `sub` so tools scope search to
that account.
"""

import base64
import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_async_session
from app.mcp_server.clients import (
    CimdError,
    DcrError,
    LEGACY_REDIRECT_URIS,
    fetch_cimd_client,
    is_allowed_redirect_uri,
    issue_dcr_client,
    resolve_oauth_client,
    verify_client_secret,
)
from app.mcp_server.workos_auth import (
    clear_pending_oauth_cookie,
    clear_wos_session_cookie,
    get_authkit_authorization_url,
    get_logout_url,
    get_pending_oauth_params,
    get_workos_user_from_request,
    get_workos_client,
    pending_oauth_query,
    resolve_distill_user,
    seal_auth_response,
    set_pending_oauth_cookie,
    set_wos_session_cookie,
    workos_configured,
    WOS_SESSION_COOKIE,
)

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(
    directory=str(Path(__file__).parent.parent / "templates")
)

ACCESS_TOKEN_AUD = "clipfinder-mcp"
CODE_AUD = "clipfinder-mcp-code"
REFRESH_AUD = "clipfinder-mcp-refresh"

CODE_LIFETIME_SECONDS = 300
ACCESS_TOKEN_LIFETIME_SECONDS = 24 * 3600
REFRESH_TOKEN_LIFETIME_SECONDS = 30 * 24 * 3600


def _client_credentials_misconfigured() -> bool:
    settings = get_settings()
    has_id = bool(settings.mcp_oauth_client_id)
    has_secret = bool(settings.mcp_oauth_client_secret)
    return has_id != has_secret


def _mcp_auth_configured() -> bool:
    return workos_configured() and not _client_credentials_misconfigured()


def _mcp_misconfigured_message() -> str:
    if _client_credentials_misconfigured():
        return (
            "MCP OAuth misconfigured: set both MCP_OAUTH_CLIENT_ID and "
            "MCP_OAUTH_CLIENT_SECRET, or leave both empty."
        )
    return (
        "MCP OAuth is not configured. Set WORKOS_API_KEY, WORKOS_CLIENT_ID, "
        "and WORKOS_COOKIE_PASSWORD."
    )


def _sign(claims: dict, aud: str, lifetime_seconds: int) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        **claims,
        "aud": aud,
        "iss": settings.app_url,
        "iat": now,
        "exp": now + timedelta(seconds=lifetime_seconds),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode(token: str, aud: str) -> dict | None:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=aud,
        )
    except JWTError:
        return None


def _token_error(error: str, description: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "error_description": description},
        headers={"Cache-Control": "no-store"},
    )


def _registration_error(error: str, description: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": error, "error_description": description},
        headers={"Cache-Control": "no-store"},
    )


def _oauth_cors_path(path: str) -> bool:
    return path.startswith("/.well-known/") or path in (
        "/mcp-oauth/register",
        "/mcp-oauth/token",
    )


class McpOAuthCorsMiddleware:
    """Allow any origin on discovery, DCR, and token (no cookies)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _oauth_cors_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        if scope.get("method") == "OPTIONS":
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", b"*"),
                        (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                        (b"access-control-allow-headers", b"Authorization, Content-Type"),
                        (b"access-control-max-age", b"86400"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in (message.get("headers") or [])
                    if k.lower()
                    not in (
                        b"access-control-allow-origin",
                        b"access-control-allow-credentials",
                    )
                ]
                headers.append((b"access-control-allow-origin", b"*"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cors)


# ---------------------------------------------------------------------------
# Discovery metadata
# ---------------------------------------------------------------------------

def _authorization_server_doc() -> dict:
    settings = get_settings()
    base = settings.app_url.rstrip("/")
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/mcp-oauth/authorize",
        "token_endpoint": f"{base}/mcp-oauth/token",
        "registration_endpoint": f"{base}/mcp-oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": [
            "none",
            "client_secret_post",
            "client_secret_basic",
        ],
        "scopes_supported": ["clipfinder"],
        "client_id_metadata_document_supported": True,
    }


@router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
@router.get("/.well-known/openid-configuration", include_in_schema=False)
async def authorization_server_metadata():
    """RFC 8414 authorization server metadata (also served as OIDC discovery)."""
    return _authorization_server_doc()


def _protected_resource_doc() -> dict:
    settings = get_settings()
    base = settings.app_url.rstrip("/")
    return {
        "resource": f"{base}/mcp/",
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
        "scopes_supported": ["clipfinder"],
    }


@router.get("/.well-known/oauth-protected-resource", include_in_schema=False)
@router.get("/.well-known/oauth-protected-resource/mcp", include_in_schema=False)
@router.get("/.well-known/oauth-protected-resource/mcp/", include_in_schema=False)
async def protected_resource_metadata():
    return _protected_resource_doc()


# ---------------------------------------------------------------------------
# Dynamic Client Registration (RFC 7591)
# ---------------------------------------------------------------------------

@router.post("/mcp-oauth/register", include_in_schema=False)
async def register_client(request: Request):
    """RFC 7591 dynamic client registration."""
    if not _mcp_auth_configured():
        return _registration_error("invalid_client_metadata", _mcp_misconfigured_message())
    try:
        body = await request.json()
    except Exception:
        return _registration_error(
            "invalid_client_metadata", "Request body must be JSON."
        )
    if not isinstance(body, dict):
        return _registration_error(
            "invalid_client_metadata", "Request body must be a JSON object."
        )
    try:
        registered = issue_dcr_client(body)
    except DcrError as exc:
        return _registration_error(exc.error, exc.description)
    return JSONResponse(
        status_code=201,
        content=registered,
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# WorkOS AuthKit login (Initiate login URL)
# ---------------------------------------------------------------------------

def _authorize_params_from_request(request: Request) -> dict[str, str]:
    return {
        k: request.query_params.get(k, "")
        for k in (
            "client_id",
            "redirect_uri",
            "state",
            "code_challenge",
        )
    }


@router.get("/mcp-oauth/login", include_in_schema=False)
async def mcp_login(request: Request):
    """Redirect to WorkOS AuthKit (Google social login)."""
    if not workos_configured():
        return HTMLResponse(
            "WorkOS is not configured (WORKOS_API_KEY/CLIENT_ID/COOKIE_PASSWORD).",
            status_code=503,
        )

    params = _authorize_params_from_request(request)
    if params.get("client_id") and params.get("code_challenge"):
        response = RedirectResponse(
            get_authkit_authorization_url(),
            status_code=302,
        )
        set_pending_oauth_cookie(response, params)
        return response

    pending = get_pending_oauth_params(request)
    if pending:
        response = RedirectResponse(
            get_authkit_authorization_url(),
            status_code=302,
        )
        set_pending_oauth_cookie(response, pending)
        return response

    return RedirectResponse(get_authkit_authorization_url(), status_code=302)


@router.get("/mcp-oauth/workos/callback", include_in_schema=False)
async def workos_callback(
    request: Request,
    code: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    """Exchange WorkOS AuthKit code, seal session, auto-approve MCP OAuth."""
    if not code:
        return RedirectResponse("/mcp-oauth/login", status_code=302)

    try:
        client = get_workos_client()
        auth_response = client.user_management.authenticate_with_code(code=code)
    except Exception as exc:
        logger.error("WorkOS authenticate_with_code failed: %s", exc)
        return HTMLResponse(f"Authentication failed: {exc}", status_code=401)

    sealed = seal_auth_response(auth_response)
    pending = get_pending_oauth_params(request)
    if pending:
        error = await _validate_authorize_params(
            pending.get("client_id", ""),
            pending.get("redirect_uri", ""),
            "code",
            pending.get("code_challenge", ""),
            "S256",
            fetch_cimd=True,
        )
        if error:
            response = templates.TemplateResponse(
                "mcp_consent.html",
                _consent_context(request, error=error),
                status_code=400,
            )
            set_wos_session_cookie(response, sealed)
            clear_pending_oauth_cookie(response)
            return response
        return await _auto_approve_redirect(
            session,
            workos_user=auth_response.user,
            client_id=pending["client_id"],
            redirect_uri=pending["redirect_uri"],
            state=pending.get("state", ""),
            code_challenge=pending["code_challenge"],
            sealed_session=sealed,
        )

    response = HTMLResponse(
        "Signed in successfully. Start the MCP connection from Claude to continue.",
        status_code=200,
    )
    set_wos_session_cookie(response, sealed)
    return response


@router.post("/mcp-oauth/logout", include_in_schema=False)
async def mcp_logout(request: Request):
    """Sign out of WorkOS AuthKit and clear the sealed session cookie."""
    sealed = request.cookies.get(WOS_SESSION_COOKIE)
    logout_url = get_logout_url(sealed)
    settings = get_settings()
    response = RedirectResponse(
        logout_url or f"{settings.app_url.rstrip('/')}/mcp-oauth/authorize",
        status_code=302,
    )
    clear_wos_session_cookie(response)
    return response


# ---------------------------------------------------------------------------
# Authorization endpoint (auto-approve after AuthKit login)
# ---------------------------------------------------------------------------

async def _auto_approve_redirect(
    session: AsyncSession,
    *,
    workos_user: Any,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    sealed_session: str | None = None,
) -> RedirectResponse:
    """Issue an authorization code and redirect back to the MCP client."""
    distill_user = await resolve_distill_user(session, workos_user)
    auth_code = _sign(
        {
            "token_use": "mcp_code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "sub": str(distill_user.id),
        },
        aud=CODE_AUD,
        lifetime_seconds=CODE_LIFETIME_SECONDS,
    )
    params: dict[str, str] = {"code": auth_code}
    if state:
        params["state"] = state
    response = RedirectResponse(
        f"{redirect_uri}?{urlencode(params)}", status_code=302
    )
    if sealed_session:
        set_wos_session_cookie(response, sealed_session)
    clear_pending_oauth_cookie(response)
    logger.info("MCP OAuth: auto-approved for user %s", distill_user.id)
    return response


async def _validate_authorize_params(
    client_id: str,
    redirect_uri: str,
    response_type: str,
    code_challenge: str,
    code_challenge_method: str,
    *,
    fetch_cimd: bool,
) -> str | None:
    if not _mcp_auth_configured():
        return _mcp_misconfigured_message()
    if response_type != "code":
        return "Only response_type=code is supported."
    if not code_challenge or code_challenge_method != "S256":
        return "PKCE with code_challenge_method=S256 is required."

    # Postman's token helper often sends client_id= and relies on the
    # allowlisted callback. Treat that the same as an unknown public client.
    if not client_id:
        if redirect_uri in LEGACY_REDIRECT_URIS:
            return None
        return "client_id is required."

    client = resolve_oauth_client(client_id)
    if client is None:
        if redirect_uri in LEGACY_REDIRECT_URIS:
            return None
        return "Unknown client_id."

    if client.kind == "cimd":
        if not is_allowed_redirect_uri(redirect_uri):
            return "redirect_uri is not allowed."
        if not fetch_cimd:
            return None
        try:
            fetched = await fetch_cimd_client(client_id)
        except CimdError as exc:
            return exc.description
        if redirect_uri not in fetched.redirect_uris:
            return "redirect_uri is not allowed."
        return None

    if redirect_uri not in client.redirect_uris:
        return "redirect_uri is not allowed."
    return None


def _consent_context(
    request: Request,
    *,
    error: str | None = None,
    client_id: str = "",
    redirect_uri: str = "",
    state: str = "",
    code_challenge: str = "",
    user_email: str | None = None,
) -> dict:
    return {
        "request": request,
        "error": error,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "user_email": user_email,
    }


@router.get("/mcp-oauth/authorize", include_in_schema=False)
async def authorize_page(
    request: Request,
    client_id: str = "",
    redirect_uri: str = "",
    response_type: str = "",
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "",
    session: AsyncSession = Depends(get_async_session),
):
    wos = get_workos_user_from_request(request)
    authenticated = bool(wos.authenticated and wos.user)
    error = await _validate_authorize_params(
        client_id,
        redirect_uri,
        response_type,
        code_challenge,
        code_challenge_method,
        fetch_cimd=authenticated,
    )
    if error:
        return templates.TemplateResponse(
            "mcp_consent.html",
            _consent_context(request, error=error),
            status_code=400,
        )

    if not authenticated:
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
        }
        login_url = f"{get_settings().app_url.rstrip('/')}/mcp-oauth/login?{pending_oauth_query(params)}"
        response = RedirectResponse(login_url, status_code=302)
        set_pending_oauth_cookie(response, params)
        return response

    sealed = (
        wos.sealed_session
        if wos.sealed_session and wos.sealed_session != request.cookies.get(WOS_SESSION_COOKIE)
        else None
    )
    return await _auto_approve_redirect(
        session,
        workos_user=wos.user,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        sealed_session=sealed,
    )


@router.post("/mcp-oauth/authorize", include_in_schema=False)
async def authorize_approve(
    request: Request,
    client_id: str = Form(""),
    redirect_uri: str = Form(""),
    state: str = Form(""),
    code_challenge: str = Form(""),
    session: AsyncSession = Depends(get_async_session),
):
    wos = get_workos_user_from_request(request)
    authenticated = bool(wos.authenticated and wos.user)
    error = await _validate_authorize_params(
        client_id,
        redirect_uri,
        "code",
        code_challenge,
        "S256",
        fetch_cimd=authenticated,
    )
    if error:
        return templates.TemplateResponse(
            "mcp_consent.html",
            _consent_context(request, error=error),
            status_code=400,
        )

    if not authenticated:
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
        }
        login_url = f"{get_settings().app_url.rstrip('/')}/mcp-oauth/login?{pending_oauth_query(params)}"
        response = RedirectResponse(login_url, status_code=302)
        set_pending_oauth_cookie(response, params)
        return response

    return await _auto_approve_redirect(
        session,
        workos_user=wos.user,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------

def _client_secret_from_basic_auth(request: Request) -> tuple[str, str] | None:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
        client_id, _, client_secret = decoded.partition(":")
        return client_id, client_secret
    except Exception:
        return None


def _verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(computed, code_challenge)


def _issue_token_pair(user_id: str, client_id: str = "") -> dict:
    access_token = _sign(
        {"token_use": "mcp_access", "sub": user_id},
        aud=ACCESS_TOKEN_AUD,
        lifetime_seconds=ACCESS_TOKEN_LIFETIME_SECONDS,
    )
    refresh_claims: dict[str, str] = {"token_use": "mcp_refresh", "sub": user_id}
    if client_id:
        refresh_claims["client_id"] = client_id
    refresh_token = _sign(
        refresh_claims,
        aud=REFRESH_AUD,
        lifetime_seconds=REFRESH_TOKEN_LIFETIME_SECONDS,
    )
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_LIFETIME_SECONDS,
        "refresh_token": refresh_token,
        "scope": "clipfinder",
    }


def _authenticate_token_client(
    client_id: str, client_secret: str
) -> JSONResponse | None:
    """Return an error response if confidential client auth fails."""
    client = resolve_oauth_client(client_id)
    if client is None or not client.confidential:
        return None
    if not verify_client_secret(client, client_secret):
        return _token_error("invalid_client", "Client authentication failed", 401)
    return None


@router.post("/mcp-oauth/token", include_in_schema=False)
async def token_endpoint(
    request: Request,
    grant_type: str = Form(""),
    code: str = Form(""),
    redirect_uri: str = Form(""),
    code_verifier: str = Form(""),
    refresh_token: str = Form(""),
    client_id: str = Form(""),
    client_secret: str = Form(""),
    resource: str = Form(""),  # RFC 8707; accepted so extra form fields do not 422.
):
    if not _mcp_auth_configured():
        return _token_error(
            "invalid_client",
            _mcp_misconfigured_message(),
            401,
        )

    basic = _client_secret_from_basic_auth(request)
    if basic:
        client_id, client_secret = basic

    if grant_type == "authorization_code":
        claims = _decode(code, aud=CODE_AUD)
        if not claims or claims.get("token_use") != "mcp_code":
            return _token_error("invalid_grant", "Invalid or expired authorization code")
        code_client_id = claims.get("client_id") or ""
        if client_id and code_client_id and client_id != code_client_id:
            return _token_error("invalid_grant", "client_id mismatch")
        if redirect_uri and redirect_uri != claims.get("redirect_uri"):
            return _token_error("invalid_grant", "redirect_uri mismatch")
        if not code_verifier or not _verify_pkce(
            code_verifier, claims.get("code_challenge", "")
        ):
            return _token_error("invalid_grant", "PKCE verification failed")
        user_id = claims.get("sub")
        if not user_id:
            return _token_error("invalid_grant", "Authorization code missing user")
        effective_client_id = client_id or code_client_id
        auth_error = _authenticate_token_client(effective_client_id, client_secret)
        if auth_error:
            return auth_error
        logger.info("MCP OAuth: issued token pair via authorization_code for %s", user_id)
        return JSONResponse(
            _issue_token_pair(user_id, effective_client_id),
            headers={"Cache-Control": "no-store"},
        )

    if grant_type == "refresh_token":
        claims = _decode(refresh_token, aud=REFRESH_AUD)
        if not claims or claims.get("token_use") != "mcp_refresh":
            return _token_error("invalid_grant", "Invalid or expired refresh token")
        refresh_client_id = claims.get("client_id") or ""
        if client_id and refresh_client_id and client_id != refresh_client_id:
            return _token_error("invalid_grant", "client_id mismatch")
        user_id = claims.get("sub")
        if not user_id:
            return _token_error("invalid_grant", "Refresh token missing user")
        effective_client_id = client_id or refresh_client_id
        auth_error = _authenticate_token_client(effective_client_id, client_secret)
        if auth_error:
            return auth_error
        logger.info("MCP OAuth: rotated token pair via refresh_token for %s", user_id)
        return JSONResponse(
            _issue_token_pair(user_id, effective_client_id),
            headers={"Cache-Control": "no-store"},
        )

    return _token_error("unsupported_grant_type", f"Unsupported grant_type '{grant_type}'")


def decode_mcp_access_token(token: str) -> dict | None:
    """Return JWT claims when token is a valid MCP access token."""
    claims = _decode(token, aud=ACCESS_TOKEN_AUD)
    if not claims or claims.get("token_use") != "mcp_access":
        return None
    return claims


def verify_mcp_access_token(token: str) -> bool:
    return decode_mcp_access_token(token) is not None
