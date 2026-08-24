"""Request-scoped MCP user identity (from OAuth access token sub claim)."""

from __future__ import annotations

import contextvars
from uuid import UUID

mcp_user_id: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "mcp_user_id",
    default=None,
)
