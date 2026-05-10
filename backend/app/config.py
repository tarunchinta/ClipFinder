"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the directory where this config file lives (backend/app/)
# Then go up one level to get backend/ where .env is located
BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env file into environment variables BEFORE pydantic-settings
# This ensures the subprocess also has the variables loaded
load_dotenv(BASE_DIR / ".env", override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Use pydantic-settings v2 syntax with absolute path to .env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/clipfinder"
    
    # JWT
    jwt_secret: str = "change-me-in-production-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    jwt_lifetime_seconds: int = 3600  # 1 hour
    
    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_api_key: str = ""  # API key for Google Picker (create in Cloud Console)
    
    # App URLs (set APP_URL/FRONTEND_URL to dev or prod values from .env)
    app_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"

    # OAuth callback: Google redirects here after login. Must match app_url and be in Google Cloud Console.
    @property
    def google_redirect_uri(self) -> str:
        return f"{self.app_url}/auth/google/callback"
    
    # Stripe
    stripe_test_key: str = ""
    
    # Redis (optional; video frame indexing runs in-process, no Redis required)
    redis_url: str = "redis://localhost:6379"
    
    # PostHog
    posthog_api_key: str = ""

    # Langfuse (optional - tracing embeddings and retrieval)
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    
    # Azure OpenAI (for text embeddings)
    azure_openai_endpoint_sample_full: str = ""  # Full endpoint URL with deployment and API version
    azure_openai_api_key: str = ""
    
    # Azure AI Vision (for CLIP/image embeddings)
    azure_ai_vision_endpoint: str = ""  # Azure ML endpoint for CLIP model
    azure_ai_vision_key: str = ""
    azure_ai_vision_deployment_name: str = "openai-clip-image-text-embedd"  # Azure ML deployment name

    # Supabase (Storage for video frame images; DB may also be Supabase/Postgres)
    supabase_url: str = ""
    supabase_key: str = ""  # service_role or anon key with storage write
    supabase_storage_bucket: str = "video-frames"  # bucket name for frame images

    # Azure Blob Storage (for video frame thumbnails; preferred when set)
    azure_blob_connection_string: str = ""
    azure_blob_container_name: str = "video-frames"

    # Azure Service Bus (for video frame indexing queue)
    service_bus_connection_string: str = Field(
        default="",
        validation_alias="SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING",
    )
    video_indexing_queue_name: str = Field(
        default="frame-indexing",
        validation_alias="VIDEO_INDEXING_QUEUE",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()



