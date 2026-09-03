"""Google OAuth token management utilities."""

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import httpx
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()


async def refresh_google_token(refresh_token: str) -> tuple[str, int] | None:
    """
    Refresh a Google OAuth access token using the refresh token.
    
    Args:
        refresh_token: The user's Google refresh token
        
    Returns:
        Tuple of (new_access_token, expires_in_seconds) or None if failed
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["access_token"], data.get("expires_in", 3600)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error refreshing Google token: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error refreshing Google token: {e}")
        return None


def _needs_refresh(access_token, expires_at) -> bool:
    """True when the token is missing, expired, or expires within 5 minutes."""
    if expires_at:
        if datetime.utcnow() >= (expires_at - timedelta(minutes=5)):
            logger.info(f"Token expired or expiring soon (expires_at: {expires_at})")
            return True
    return not access_token


async def get_valid_access_token_via_postgrest(
    db,
    user_id: UUID,
    access_token: Optional[str],
    refresh_token: Optional[str],
    expires_at: Optional[datetime],
) -> Optional[str]:
    """get_valid_access_token for the worker path, persisting over PostgREST.

    Same refresh rules as the ORM version; only the write differs, so the Service
    Bus triggers never need a Postgres connection to top up a Google token.
    """
    from app.services.indexing_store import update_user_google_token

    if not _needs_refresh(access_token, expires_at):
        return access_token

    if not refresh_token:
        logger.warning("No refresh token available for token refresh")
        return None

    logger.info("Refreshing Google access token...")
    result = await refresh_google_token(refresh_token)
    if not result:
        logger.error("Failed to refresh Google access token")
        return None

    new_access_token, expires_in = result
    new_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    try:
        await update_user_google_token(
            db,
            user_id,
            access_token=new_access_token,
            expires_at=new_expires_at,
        )
        logger.info(f"Token refreshed successfully, expires at {new_expires_at}")
    except Exception as e:
        # Mirrors the ORM path: a failed write must not cost us a usable token.
        logger.error(f"Error updating token via PostgREST: {e}")

    return new_access_token


async def get_valid_access_token(
    user_id: UUID,
    access_token: Optional[str],
    refresh_token: Optional[str],
    expires_at: Optional[datetime],
    session: AsyncSession
) -> Optional[str]:
    """
    Get a valid Google access token, refreshing if expired.
    
    Checks if the current token is expired and refreshes it using the
    refresh token if necessary. Updates the database with the new token.
    
    Args:
        user_id: User UUID
        access_token: Current access token (may be expired)
        refresh_token: Refresh token for getting new access tokens
        expires_at: When the current access token expires
        session: Database session for updating tokens
        
    Returns:
        Valid access token or None if unable to get one
    """
    # Check if token is expired or about to expire (within 5 minutes)
    token_expired = False
    if expires_at:
        buffer = timedelta(minutes=5)
        if datetime.utcnow() >= (expires_at - buffer):
            token_expired = True
            logger.info(f"Token expired or expiring soon (expires_at: {expires_at})")
    
    # If token is valid, return it
    if not token_expired and access_token:
        return access_token
    
    # Need to refresh the token
    if not refresh_token:
        logger.warning("No refresh token available for token refresh")
        return None
    
    logger.info("Refreshing Google access token...")
    result = await refresh_google_token(refresh_token)
    if not result:
        logger.error("Failed to refresh Google access token")
        return None
    
    new_access_token, expires_in = result
    new_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    
    # Update the token in the database
    try:
        stmt = update(User).where(User.id == user_id).values(
            google_access_token=new_access_token,
            google_token_expires_at=new_expires_at
        )
        await session.execute(stmt)
        await session.commit()
        logger.info(f"Token refreshed successfully, expires at {new_expires_at}")
    except Exception as e:
        logger.error(f"Error updating token in database: {e}")
        # Still return the token even if DB update fails
    
    return new_access_token
