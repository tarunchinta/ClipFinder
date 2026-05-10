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
from app.routers import auth_router, pages_router, drive_router

# Configure logging so app and router logs are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("app").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


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



