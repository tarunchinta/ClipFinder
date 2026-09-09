"""OAuth client identity for MCP: pre-registered, DCR (RFC 7591), and CIMD."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import secrets
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from jose import JWTError, jwt

from app.config import get_settings

logger = logging.getLogger(__name__)

LEGACY_REDIRECT_URIS = frozenset(
    {
        "https://claude.ai/api/mcp/auth_callback",
        "https://oauth.pstmn.io/v1/callback",
    }
)

DCR_AUD = "clipfinder-mcp-dcr-client"
DCR_TOKEN_USE = "mcp_dcr_client"

MAX_REDIRECT_URIS = 10
MAX_REDIRECT_URI_LENGTH = 2048
MAX_CLIENT_NAME_LENGTH = 256

CIMD_TIMEOUT_SECONDS = 5.0
CIMD_MAX_BODY_BYTES = 64 * 1024
CIMD_DEFAULT_TTL_SECONDS = 3600
CIMD_MAX_TTL_SECONDS = 24 * 3600

_ALLOWED_TOKEN_AUTH_METHODS = frozenset(
    {"none", "client_secret_post", "client_secret_basic"}
)
_ALLOWED_GRANT_TYPES = frozenset({"authorization_code", "refresh_token"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# hostname -> (expires_at_monotonic-ish epoch, RegisteredClient)
_cimd_cache: dict[str, tuple[float, "RegisteredClient"]] = {}


class DcrError(Exception):
    def __init__(self, error: str, description: str):
        super().__init__(description)
        self.error = error
        self.description = description


class CimdError(Exception):
    def __init__(self, description: str):
        super().__init__(description)
        self.description = description


@dataclass(frozen=True)
class RegisteredClient:
    client_id: str
    redirect_uris: tuple[str, ...]
    kind: str  # "env" | "dcr" | "cimd"
    client_name: str = ""
    token_endpoint_auth_method: str = "none"
    client_secret_hash: str | None = None

    @property
    def confidential(self) -> bool:
        return self.token_endpoint_auth_method != "none"


def hash_client_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def is_allowed_redirect_uri(uri: str) -> bool:
    """HTTPS or loopback HTTP, absolute, no fragment."""
    if not uri or not isinstance(uri, str) or len(uri) > MAX_REDIRECT_URI_LENGTH:
        return False
    parsed = urlparse(uri)
    if parsed.fragment or not parsed.scheme or not parsed.netloc:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme == "http":
        return host in _LOOPBACK_HOSTS
    return False


def is_cimd_client_id(client_id: str) -> bool:
    if not client_id or not isinstance(client_id, str):
        return False
    parsed = urlparse(client_id)
    if parsed.scheme != "https" or not parsed.hostname or parsed.fragment:
        return False
    path = parsed.path or ""
    return path not in ("", "/")


def cimd_url_precheck(url: str) -> str | None:
    """Return an error if the URL must not be fetched (no DNS). None if shape is OK."""
    if not is_cimd_client_id(url):
        return "client_id is not a valid Client ID Metadata Document URL."
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        if hostname.lower() in _LOOPBACK_HOSTS:
            return "client_id host is not allowed."
        return None
    if _is_blocked_ip(ip):
        return "client_id host is not allowed."
    return None


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if getattr(addr, "ipv4_mapped", None) is not None and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _blocked_ip_for_host(hostname: str) -> str | None:
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            return "client_id host is not allowed."
        return None
    if hostname.lower() in _LOOPBACK_HOSTS:
        return "client_id host is not allowed."
    try:
        infos = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return "Unable to resolve client_id host."
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if _is_blocked_ip(addr):
            return "client_id host is not allowed."
    return None


def parse_cimd_document(url: str, data: Any) -> RegisteredClient:
    if not isinstance(data, dict):
        raise CimdError("Client metadata document must be a JSON object.")
    if data.get("client_id") != url:
        raise CimdError("Metadata client_id must match the document URL.")
    client_name = data.get("client_name")
    if not isinstance(client_name, str) or not client_name.strip():
        raise CimdError("Metadata is missing client_name.")
    uris = data.get("redirect_uris")
    if not isinstance(uris, list) or not uris:
        raise CimdError("Metadata is missing redirect_uris.")
    cleaned: list[str] = []
    for uri in uris:
        if not isinstance(uri, str) or not is_allowed_redirect_uri(uri):
            raise CimdError("Metadata contains an invalid redirect_uri.")
        cleaned.append(uri)
    auth_method = data.get("token_endpoint_auth_method") or "none"
    if auth_method not in _ALLOWED_TOKEN_AUTH_METHODS:
        raise CimdError("Unsupported token_endpoint_auth_method in metadata.")
    return RegisteredClient(
        client_id=url,
        redirect_uris=tuple(cleaned),
        kind="cimd",
        client_name=client_name.strip()[:MAX_CLIENT_NAME_LENGTH],
        token_endpoint_auth_method=str(auth_method),
    )


def _cache_ttl_seconds(headers: httpx.Headers) -> int:
    cache_control = headers.get("cache-control") or ""
    for part in cache_control.split(","):
        directive = part.strip().lower()
        if directive.startswith("max-age="):
            try:
                max_age = int(directive.split("=", 1)[1].strip())
            except ValueError:
                break
            return max(0, min(max_age, CIMD_MAX_TTL_SECONDS))
        if directive in ("no-store", "no-cache"):
            return 0
    expires = headers.get("expires")
    if expires:
        try:
            exp = parsedate_to_datetime(expires)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            ttl = int((exp - datetime.now(timezone.utc)).total_seconds())
            return max(0, min(ttl, CIMD_MAX_TTL_SECONDS))
        except (TypeError, ValueError, OverflowError):
            pass
    return CIMD_DEFAULT_TTL_SECONDS


def _cached_cimd(url: str) -> RegisteredClient | None:
    entry = _cimd_cache.get(url)
    if not entry:
        return None
    expires_at, client = entry
    if expires_at <= datetime.now(timezone.utc).timestamp():
        _cimd_cache.pop(url, None)
        return None
    return client


def clear_cimd_cache() -> None:
    _cimd_cache.clear()


async def fetch_cimd_client(url: str) -> RegisteredClient:
    cached = _cached_cimd(url)
    if cached:
        return cached
    precheck = cimd_url_precheck(url)
    if precheck:
        raise CimdError(precheck)
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    blocked = await asyncio.to_thread(_blocked_ip_for_host, hostname)
    if blocked:
        raise CimdError(blocked)
    try:
        async with httpx.AsyncClient(
            timeout=CIMD_TIMEOUT_SECONDS,
            follow_redirects=False,
            verify=True,
        ) as client:
            response = await client.get(
                url,
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        logger.warning("CIMD fetch failed for %s: %s", url, exc)
        raise CimdError("Failed to fetch client metadata document.") from exc
    if response.status_code != 200:
        raise CimdError("Failed to fetch client metadata document.")
    if len(response.content) > CIMD_MAX_BODY_BYTES:
        raise CimdError("Client metadata document is too large.")
    try:
        data = response.json()
    except ValueError as exc:
        raise CimdError("Client metadata document is not valid JSON.") from exc
    registered = parse_cimd_document(url, data)
    ttl = _cache_ttl_seconds(response.headers)
    if ttl > 0:
        _cimd_cache[url] = (
            datetime.now(timezone.utc).timestamp() + ttl,
            registered,
        )
    return registered


def issue_dcr_client(metadata: dict[str, Any]) -> dict[str, Any]:
    redirect_uris = metadata.get("redirect_uris")
    if not isinstance(redirect_uris, list) or not redirect_uris:
        raise DcrError("invalid_redirect_uri", "redirect_uris is required.")
    if len(redirect_uris) > MAX_REDIRECT_URIS:
        raise DcrError(
            "invalid_redirect_uri",
            f"At most {MAX_REDIRECT_URIS} redirect_uris are allowed.",
        )
    cleaned: list[str] = []
    for uri in redirect_uris:
        if not isinstance(uri, str) or not is_allowed_redirect_uri(uri):
            raise DcrError("invalid_redirect_uri", "redirect_uri is not allowed.")
        cleaned.append(uri)

    auth_method = metadata.get("token_endpoint_auth_method") or "none"
    if auth_method not in _ALLOWED_TOKEN_AUTH_METHODS:
        raise DcrError(
            "invalid_client_metadata",
            "Unsupported token_endpoint_auth_method.",
        )

    requested_grants = metadata.get("grant_types")
    if requested_grants is None:
        grant_types = ["authorization_code", "refresh_token"]
    else:
        if not isinstance(requested_grants, list) or not requested_grants:
            raise DcrError("invalid_client_metadata", "grant_types is invalid.")
        for grant in requested_grants:
            if grant not in _ALLOWED_GRANT_TYPES:
                raise DcrError(
                    "invalid_client_metadata",
                    f"Unsupported grant_type '{grant}'.",
                )
        if "authorization_code" not in requested_grants:
            raise DcrError(
                "invalid_client_metadata",
                "grant_types must include authorization_code.",
            )
        grant_types = list(dict.fromkeys(requested_grants))

    requested_response = metadata.get("response_types")
    if requested_response is None:
        response_types = ["code"]
    else:
        if not isinstance(requested_response, list) or requested_response != ["code"]:
            raise DcrError(
                "invalid_client_metadata",
                "Only response_types=['code'] is supported.",
            )
        response_types = ["code"]

    client_name = metadata.get("client_name") or ""
    if not isinstance(client_name, str):
        raise DcrError("invalid_client_metadata", "client_name must be a string.")
    client_name = client_name.strip()[:MAX_CLIENT_NAME_LENGTH]

    secret: str | None = None
    secret_hash: str | None = None
    if auth_method != "none":
        secret = secrets.token_urlsafe(32)
        secret_hash = hash_client_secret(secret)

    settings = get_settings()
    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "token_use": DCR_TOKEN_USE,
        "redirect_uris": cleaned,
        "client_name": client_name,
        "token_endpoint_auth_method": auth_method,
        "aud": DCR_AUD,
        "iss": settings.app_url,
        "iat": now,
        "jti": str(uuid.uuid4()),
    }
    if secret_hash:
        claims["client_secret_hash"] = secret_hash
    client_id = jwt.encode(
        claims, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    if isinstance(client_id, bytes):
        client_id = client_id.decode("ascii")

    body: dict[str, Any] = {
        "client_id": client_id,
        "client_id_issued_at": int(now.timestamp()),
        "redirect_uris": cleaned,
        "grant_types": grant_types,
        "response_types": response_types,
        "token_endpoint_auth_method": auth_method,
    }
    if client_name:
        body["client_name"] = client_name
    if secret:
        body["client_secret"] = secret
    return body


def decode_dcr_client(client_id: str) -> RegisteredClient | None:
    settings = get_settings()
    try:
        claims = jwt.decode(
            client_id,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=DCR_AUD,
            options={"verify_exp": False},
        )
    except JWTError:
        return None
    if claims.get("token_use") != DCR_TOKEN_USE:
        return None
    uris = claims.get("redirect_uris") or []
    if not isinstance(uris, list) or not all(
        isinstance(u, str) and is_allowed_redirect_uri(u) for u in uris
    ):
        return None
    auth_method = claims.get("token_endpoint_auth_method") or "none"
    if auth_method not in _ALLOWED_TOKEN_AUTH_METHODS:
        return None
    name = claims.get("client_name") or ""
    secret_hash = claims.get("client_secret_hash")
    return RegisteredClient(
        client_id=client_id,
        redirect_uris=tuple(uris),
        kind="dcr",
        client_name=str(name),
        token_endpoint_auth_method=str(auth_method),
        client_secret_hash=str(secret_hash) if secret_hash else None,
    )


def _env_client() -> RegisteredClient | None:
    settings = get_settings()
    client_id = settings.mcp_oauth_client_id
    secret = settings.mcp_oauth_client_secret
    if not (client_id and secret):
        return None
    return RegisteredClient(
        client_id=client_id,
        redirect_uris=tuple(LEGACY_REDIRECT_URIS),
        kind="env",
        client_name="Pre-registered MCP client",
        token_endpoint_auth_method="client_secret_post",
    )


def resolve_oauth_client(client_id: str) -> RegisteredClient | None:
    """Resolve a client without fetching CIMD documents."""
    if not client_id:
        return None
    env = _env_client()
    if (
        env
        and len(client_id) == len(env.client_id)
        and secrets.compare_digest(client_id, env.client_id)
    ):
        return env
    dcr = decode_dcr_client(client_id)
    if dcr:
        return dcr
    if is_cimd_client_id(client_id):
        return RegisteredClient(
            client_id=client_id,
            redirect_uris=(),
            kind="cimd",
            token_endpoint_auth_method="none",
        )
    return None


def verify_client_secret(client: RegisteredClient, client_secret: str) -> bool:
    if client.kind == "env":
        expected = get_settings().mcp_oauth_client_secret
        return bool(
            client_secret
            and expected
            and len(client_secret) == len(expected)
            and secrets.compare_digest(client_secret, expected)
        )
    if client.kind == "dcr" and client.client_secret_hash:
        if not client_secret:
            return False
        digest = hash_client_secret(client_secret)
        return secrets.compare_digest(digest, client.client_secret_hash)
    return not client.confidential
