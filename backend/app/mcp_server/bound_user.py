"""Resolve the Distill user for MCP tools and HTTP reel ingest."""

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_async_session
from app.mcp_server.context import mcp_user_id
from app.models.user import User


async def get_bound_user(session: AsyncSession) -> User:
    """Resolve the Distill user from the MCP OAuth access token (sub claim)."""
    user_id = mcp_user_id.get()
    if user_id is None:
        raise ValueError("No MCP user identity in request context")
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).unique().scalar_one_or_none()
    if not user:
        raise ValueError(f"MCP token user id '{user_id}' does not match any Distill user")
    return user


async def get_ingest_user(
    session: AsyncSession = Depends(get_async_session),
) -> User:
    """User identity for POST /reels/ingest (env-bound, not MCP OAuth)."""
    email = get_settings().mcp_user_email
    if not email:
        raise HTTPException(status_code=503, detail="MCP_USER_EMAIL is not configured")
    user = (
        await session.execute(select(User).where(User.email == email))
    ).unique().scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=503,
            detail=f"MCP_USER_EMAIL '{email}' does not match any Distill user",
        )
    return user
