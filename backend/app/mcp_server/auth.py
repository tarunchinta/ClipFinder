"""Bearer auth for the mounted MCP app: OAuth access tokens only."""

import json
from uuid import UUID

from app.config import get_settings
from app.mcp_server.context import mcp_user_id
from app.mcp_server.oauth import (
    _mcp_auth_configured,
    _mcp_misconfigured_message,
    decode_mcp_access_token,
)


class StaticBearerAuthMiddleware:
    """
    Pure ASGI middleware wrapping only the MCP sub-app. Accepts
    "Authorization: Bearer <OAuth access token>" issued by /mcp-oauth/token
    after WorkOS AuthKit login + consent.

    401 responses carry a WWW-Authenticate pointer to the protected-resource
    metadata so OAuth-capable clients can discover the authorization server.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not _mcp_auth_configured():
            await self._reject(send, 503, _mcp_misconfigured_message())
            return

        auth_header = ""
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                auth_header = value.decode("latin-1")
                break

        token_ctx = mcp_user_id.set(None)
        try:
            user_uuid = self._user_id_from_header(auth_header)
            if user_uuid is None:
                await self._reject(send, 401, "Invalid or missing bearer token")
                return
            mcp_user_id.set(user_uuid)
            await self.app(scope, receive, send)
        finally:
            mcp_user_id.reset(token_ctx)

    @staticmethod
    def _user_id_from_header(auth_header: str) -> UUID | None:
        if not auth_header.lower().startswith("bearer "):
            return None
        token = auth_header[7:].strip()
        claims = decode_mcp_access_token(token)
        if not claims:
            return None
        sub = claims.get("sub")
        if not sub:
            return None
        try:
            return UUID(str(sub))
        except ValueError:
            return None

    @staticmethod
    async def _reject(send, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("latin-1")),
        ]
        if status_code == 401:
            base = get_settings().app_url.rstrip("/")
            www = f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'
            headers.append((b"www-authenticate", www.encode("latin-1")))
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})
