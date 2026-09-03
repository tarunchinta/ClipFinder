"""PostgREST transport for the indexing workers.

Each Service Bus trigger used to open a SQLAlchemy engine pool and then a
*separate* AsyncSession per concurrent frame task (`async_session_maker()` inside
the per-frame, thumbnail, colour and transcript coroutines), so a single queue
message could hold a dozen Postgres connections at once.

This module replaces that with Supabase's PostgREST endpoint reached through one
pooled `httpx.AsyncClient`. `postgrest_session()` is opened once per trigger and
`POSTGREST_MAX_CONNECTIONS` (default 1) caps the pool, so however many frames are
embedded in parallel, the invocation holds exactly one connection and httpx
queues the requests over it. Postgres connection pooling itself now lives on the
Supabase side, where PostgREST owns a long-lived pool shared by every worker.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Iterable, Sequence
from uuid import UUID

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


class PostgrestError(RuntimeError):
    """A PostgREST request came back non-2xx."""

    def __init__(self, method: str, path: str, status_code: int, body: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body
        super().__init__(f"{method} {path} -> {status_code}: {body[:500]}")


class PostgrestNotConfiguredError(RuntimeError):
    """SUPABASE_URL / service-role key are missing, so there is no data path."""


def to_vector(values: Sequence[float] | None) -> str | None:
    """Format an embedding as a pgvector literal.

    Vectors go to Postgres as text (`[0.1,0.2,...]`) through RPC arguments that
    cast explicitly, rather than as JSON arrays, because PostgREST has no way to
    know the target column is `vector` rather than a numeric array.
    """
    if values is None:
        return None
    return "[" + ",".join(format(float(v), ".7g") for v in values) + "]"


def to_iso(value: datetime | None) -> str | None:
    """Serialise a naive UTC datetime the way the existing columns store it."""
    if value is None:
        return None
    return value.isoformat()


def _stringify(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class PostgrestClient:
    """Thin async wrapper over the PostgREST REST interface.

    Safe to share across concurrent tasks: httpx serialises requests over the
    bounded connection pool, which is the whole point of the single connection.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: JsonDict | None = None,
        json: Any = None,
        headers: JsonDict | None = None,
    ) -> httpx.Response:
        response = await self._client.request(
            method, path, params=params, json=json, headers=headers
        )
        if response.status_code >= 400:
            raise PostgrestError(method, path, response.status_code, response.text)
        return response

    @staticmethod
    def _filters(match: JsonDict) -> JsonDict:
        """Turn {"id": uuid} into PostgREST's {"id": "eq.<uuid>"} filter syntax."""
        return {column: f"eq.{_stringify(value)}" for column, value in match.items()}

    async def select(
        self,
        table: str,
        *,
        match: JsonDict,
        columns: str = "*",
        limit: int | None = None,
    ) -> list[JsonDict]:
        params = self._filters(match)
        params["select"] = columns
        if limit is not None:
            params["limit"] = str(limit)
        response = await self._request("GET", f"/{table}", params=params)
        return response.json()

    async def select_one(
        self,
        table: str,
        *,
        match: JsonDict,
        columns: str = "*",
    ) -> JsonDict | None:
        rows = await self.select(table, match=match, columns=columns, limit=1)
        return rows[0] if rows else None

    async def update(
        self,
        table: str,
        *,
        match: JsonDict,
        values: JsonDict,
        returning: bool = False,
    ) -> list[JsonDict]:
        """PATCH rows. Skips the round trip entirely when there is nothing to set."""
        if not values:
            return []
        headers = {
            "Prefer": "return=representation" if returning else "return=minimal"
        }
        response = await self._request(
            "PATCH",
            f"/{table}",
            params=self._filters(match),
            json={k: _json_safe(v) for k, v in values.items()},
            headers=headers,
        )
        return response.json() if returning else []

    async def insert(
        self,
        table: str,
        rows: Iterable[JsonDict],
        *,
        returning: bool = False,
    ) -> list[JsonDict]:
        payload = [{k: _json_safe(v) for k, v in row.items()} for row in rows]
        if not payload:
            return []
        headers = {
            "Prefer": "return=representation" if returning else "return=minimal"
        }
        response = await self._request("POST", f"/{table}", json=payload, headers=headers)
        return response.json() if returning else []

    async def delete(self, table: str, *, match: JsonDict) -> None:
        await self._request(
            "DELETE",
            f"/{table}",
            params=self._filters(match),
            headers={"Prefer": "return=minimal"},
        )

    async def rpc(self, function_name: str, args: JsonDict) -> Any:
        """Call a Postgres function exposed at /rpc/<name>.

        Everything that must be atomic (the frame counters) or that writes a
        `vector` column goes through here rather than through PATCH.
        """
        response = await self._request(
            "POST",
            f"/rpc/{function_name}",
            json={k: _json_safe(v) for k, v in args.items()},
        )
        if not response.content:
            return None
        return response.json()


def _json_safe(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@asynccontextmanager
async def postgrest_session() -> AsyncIterator[PostgrestClient]:
    """Open the one pooled connection a single trigger invocation is allowed.

    Opened per invocation on purpose: the Functions host calls `asyncio.run()` per
    message, so a client cached across invocations would be bound to a closed
    event loop.
    """
    settings = get_settings()
    base_url = settings.postgrest_url
    key = settings.postgrest_key
    if not base_url or not key:
        raise PostgrestNotConfiguredError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for the "
            "PostgREST data path"
        )

    max_connections = max(1, settings.postgrest_max_connections)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        limits=limits,
        timeout=settings.postgrest_timeout_seconds,
    ) as client:
        logger.debug(
            "PostgREST session open (max_connections=%d, base_url=%s)",
            max_connections,
            base_url,
        )
        yield PostgrestClient(client)
