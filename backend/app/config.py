"""Application configuration using pydantic-settings."""



from functools import lru_cache

from pathlib import Path



from dotenv import load_dotenv

from pydantic import AliasChoices, Field

from pydantic_settings import BaseSettings, SettingsConfigDict



# Get the directory where this config file lives (backend/app/)

# Then go up one level to get backend/ where .env is located

BASE_DIR = Path(__file__).resolve().parent.parent



# Load local development defaults without replacing environment variables
# supplied by the deployment platform (for example, Azure Container Apps).

load_dotenv(BASE_DIR / ".env", override=False)





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

    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )

    

    # Gemini Embedding 2 (for vision/image embeddings via Google AI Studio)

    gemini_api_key: str = Field(

        default="",

        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_AI_VISION_API_KEY"),

    )

    gemini_embedding_model: str = Field(

        default="gemini-embedding-2",

        validation_alias=AliasChoices("GEMINI_EMBEDDING_MODEL", "GOOGLE_AI_VISION_EMBEDDING_MODEL"),

    )

    gemini_embedding_dimension: int = Field(

        default=768,

        validation_alias=AliasChoices("GEMINI_EMBEDDING_DIMENSION", "GOOGLE_AI_VISION_EMBEDDING_DIMENSION"),

    )



    # Supabase (Storage for video frame images; DB may also be Supabase/Postgres)

    supabase_url: str = ""

    supabase_key: str = ""  # service_role or anon key with storage write

    supabase_storage_bucket: str = "video-frames"  # bucket name for frame images



    # Azure Blob Storage (for video frame thumbnails; preferred when set)

    azure_blob_connection_string: str = ""

    azure_blob_container_name: str = Field(
        default="video-frames",
        validation_alias=AliasChoices(
            "AZURE_BLOB_CONTAINER_NAME",
            "AZURE_BLOB_STORAGE_CONTAINER_NAME",
        ),
    )



    # Azure Service Bus (required for indexing job enqueue)

    service_bus_connection_string: str = Field(

        default="",

        validation_alias="SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING",

    )

    video_indexing_queue_name: str = Field(

        default="frame-indexing",

        validation_alias="VIDEO_INDEXING_QUEUE",

    )

    image_indexing_queue_name: str = Field(

        default="image-indexing",

        validation_alias="IMAGE_INDEXING_QUEUE",

    )



    # Max concurrent embed+DB tasks per video during in-process frame indexing

    frame_index_parallelism: int = Field(

        default=8,

        validation_alias="FRAME_INDEX_PARALLELISM",

    )



    # Whisper transcription (WhisperX, word-level timestamps, runs locally on CPU during video indexing)

    transcription_enabled: bool = Field(

        default=True,

        validation_alias="TRANSCRIPTION_ENABLED",

    )

    whisper_model_size: str = Field(

        default="tiny",

        validation_alias="WHISPER_MODEL_SIZE",

    )



    # Remote MCP server (streamable HTTP at /mcp)

    mcp_user_email: str = Field(

        default="",

        validation_alias="MCP_USER_EMAIL",

    )

    azure_blob_videos_container_name: str = Field(

        default="instagram-videos",

        validation_alias="AZURE_BLOB_VIDEOS_CONTAINER",

    )

    mcp_max_reel_seconds: int = Field(

        default=180,

        validation_alias="MCP_MAX_REEL_SECONDS",

    )

    # OAuth client credentials for claude.ai custom connectors (Advanced settings).

    mcp_oauth_client_id: str = Field(

        default="",

        validation_alias="MCP_OAUTH_CLIENT_ID",

    )

    mcp_oauth_client_secret: str = Field(

        default="",

        validation_alias="MCP_OAUTH_CLIENT_SECRET",

    )

    # WorkOS AuthKit (Google social login before MCP consent)

    workos_api_key: str = Field(

        default="",

        validation_alias="WORKOS_API_KEY",

    )

    workos_client_id: str = Field(

        default="",

        validation_alias="WORKOS_CLIENT_ID",

    )

    workos_redirect_uri: str = Field(

        default="",

        validation_alias="WORKOS_REDIRECT_URI",

    )

    workos_cookie_password: str = Field(

        default="",

        validation_alias="WORKOS_COOKIE_PASSWORD",

    )

    @property

    def workos_callback_uri(self) -> str:

        if self.workos_redirect_uri:

            return self.workos_redirect_uri

        return f"{self.app_url.rstrip('/')}/mcp-oauth/workos/callback"

    @property

    def cookie_secure(self) -> bool:

        return self.app_url.startswith("https://")





@lru_cache

def get_settings() -> Settings:

    """Get cached settings instance."""

    return Settings()





