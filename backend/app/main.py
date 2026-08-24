"""
ClipFinder MVP - FastAPI Application

Semantic video clip search for creators.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging

from app.config import get_settings
from app.database import create_db_and_tables
from app.mcp_server.auth import StaticBearerAuthMiddleware
from app.mcp_server.oauth import router as mcp_oauth_router
from app.mcp_server.server import mcp
from app.routers import auth_router, pages_router, drive_router, reels_router

# Configure logging so app and router logs are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("app").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


class McpPathNormalizerMiddleware:
    """Handle the slashless MCP endpoint without an HTTP redirect."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Creates database tables on startup.
    """
    logger.info("Starting ClipFinder API...")
    
    # Create database tables
    await create_db_and_tables()
    logger.info("Database tables created/verified")

    # Starlette does not run mounted sub-app lifespans; without this every
    # MCP request fails with "Task group is not initialized"
    async with mcp.session_manager.run():
        yield

    logger.info("Shutting down ClipFinder API...")


# Create FastAPI application
app = FastAPI(
    title="ClipFinder API",
    description="Semantic video clip search for creators",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Claude sends requests to /mcp even when discovery advertises /mcp/.
# Rewrite internally so auth headers and POST bodies survive unchanged.
app.add_middleware(McpPathNormalizerMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        settings.app_url,
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(auth_router, prefix="/auth")
app.include_router(pages_router)
app.include_router(drive_router)
app.include_router(reels_router)

# OAuth endpoints + discovery metadata for claude.ai custom connectors
app.include_router(mcp_oauth_router)

# Remote MCP server (streamable HTTP), guarded by bearer auth (static key or OAuth)
app.mount("/mcp", StaticBearerAuthMiddleware(mcp.streamable_http_app()))


# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "0.1.0"}


# API info endpoint
@app.get("/api", tags=["system"])
async def api_info():
    """API information endpoint."""
    return {
        "name": "ClipFinder API",
        "version": "0.1.0",
        "description": "Semantic video clip search for creators",
        "docs": "/docs",
        "endpoints": {
            "auth": "/auth",
            "search": "/api/search (coming soon)",
            "embed": "/api/embed (coming soon)",
            "index": "/api/index (coming soon)",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )



