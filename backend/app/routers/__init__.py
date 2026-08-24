"""API routers."""

from app.routers.auth import router as auth_router
from app.routers.pages import router as pages_router
from app.routers.drive import router as drive_router
from app.routers.reels import router as reels_router

__all__ = ["auth_router", "pages_router", "drive_router", "reels_router"]



