"""Instagram reel HTTP ingest (Distill-compatible JSON surface)."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_session
from app.mcp_server.bound_user import get_ingest_user
from app.models.user import User
from app.services.instagram_ingest import InstagramIngestError, ingest_instagram_reel

router = APIRouter(tags=["reels"])


class ReelIngestRequest(BaseModel):
    url: HttpUrl


async def _ingest_reel_handler(
    request: ReelIngestRequest,
    user: User,
    session: AsyncSession,
) -> dict:
    try:
        return await ingest_instagram_reel(str(request.url), user, session)
    except InstagramIngestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/reels/ingest")
async def ingest_reel(
    request: ReelIngestRequest,
    user: User = Depends(get_ingest_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    return await _ingest_reel_handler(request, user, session)


@router.post("/reels/download")
async def download_reel(
    request: ReelIngestRequest,
    user: User = Depends(get_ingest_user),
    session: AsyncSession = Depends(get_async_session),
) -> dict:
    """Distill-compatible alias for POST /reels/ingest."""
    return await _ingest_reel_handler(request, user, session)
