"""Google Drive API service for file operations."""

from typing import Optional
from datetime import datetime, timedelta
import logging

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Video MIME types we support
VIDEO_MIME_TYPES = [
    "video/mp4",
    "video/quicktime",  # .mov
    "video/x-msvideo",  # .avi
    "video/webm",
    "video/x-matroska",  # .mkv
    "video/mpeg",
    "video/3gpp",
    "video/x-m4v",
]

# Image MIME types we support
IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/heic",
    "image/heif",
]

# Combined media types
SUPPORTED_MIME_TYPES = VIDEO_MIME_TYPES + IMAGE_MIME_TYPES

# File size limit: 100 MB
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Duration limit: 30 seconds
MAX_DURATION_SECONDS = 30

# Max clips per folder
MAX_CLIPS_PER_FOLDER = 100


class GoogleDriveService:
    """Service for interacting with Google Drive API."""
    
    def __init__(self, access_token: str, refresh_token: Optional[str] = None):
        """
        Initialize the Drive service with OAuth credentials.
        
        Args:
            access_token: User's Google OAuth access token
            refresh_token: Optional refresh token for token renewal
        """
        self.credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
        )
        self.service = build("drive", "v3", credentials=self.credentials)
    
    def get_folder_info(self, folder_id: str) -> Optional[dict]:
        """
        Get information about a folder.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            Folder metadata or None if not found/accessible
        """
        try:
            folder = self.service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType"
            ).execute()
            
            if folder.get("mimeType") != "application/vnd.google-apps.folder":
                logger.warning(f"Item {folder_id} is not a folder")
                return None
                
            return folder
        except HttpError as e:
            logger.error(f"Error getting folder {folder_id}: {e}")
            return None
    
    def list_media_files(self, folder_id: str) -> list[dict]:
        """
        List media files (videos and images) in a folder with validation metadata.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            List of file objects with validation status
        """
        try:
            # Build query for media files (videos and images) in the specified folder
            mime_query = " or ".join([f"mimeType='{mt}'" for mt in SUPPORTED_MIME_TYPES])
            query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"
            
            # Request file metadata including video and image-specific properties
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, modifiedTime, videoMediaMetadata, imageMediaMetadata, thumbnailLink, webViewLink, webContentLink)",
                pageSize=MAX_CLIPS_PER_FOLDER,
                orderBy="name"
            ).execute()
            
            files = results.get("files", [])
            
            # Validate each file
            validated_files = []
            for file in files:
                validated_file = self._validate_file(file)
                validated_files.append(validated_file)
            
            return validated_files
            
        except HttpError as e:
            logger.error(f"Error listing files in folder {folder_id}: {e}")
            return []
    
    def _validate_file(self, file: dict) -> dict:
        """
        Validate a file against MVP constraints.
        
        Checks:
        - File size ≤ 100 MB
        - Video duration ≤ 30 seconds (videos only)
        
        Args:
            file: File metadata from Drive API
            
        Returns:
            File dict with added validation fields
        """
        errors = []
        mime_type = file.get("mimeType", "")
        is_video = mime_type in VIDEO_MIME_TYPES
        is_image = mime_type in IMAGE_MIME_TYPES
        media_type = "video" if is_video else "image" if is_image else "unknown"
        
        # Check file size
        size_bytes = int(file.get("size", 0))
        size_mb = size_bytes / (1024 * 1024)
        
        if size_bytes > MAX_FILE_SIZE_BYTES:
            errors.append(f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_BYTES // (1024*1024)} MB)")
        
        # Get dimensions and duration based on media type
        width = None
        height = None
        duration_seconds = None
        
        if is_video:
            # Check video duration (if available)
            video_metadata = file.get("videoMediaMetadata", {})
            duration_ms = video_metadata.get("durationMillis")
            width = video_metadata.get("width")
            height = video_metadata.get("height")
            
            if duration_ms:
                duration_seconds = int(duration_ms) / 1000
                if duration_seconds > MAX_DURATION_SECONDS:
                    errors.append(f"Video too long: {duration_seconds:.1f}s (max {MAX_DURATION_SECONDS}s)")
        elif is_image:
            # Get image dimensions
            image_metadata = file.get("imageMediaMetadata", {})
            width = image_metadata.get("width")
            height = image_metadata.get("height")
            # Images have no duration constraint
        
        # Build validated file object
        return {
            "id": file.get("id"),
            "name": file.get("name"),
            "mimeType": mime_type,
            "mediaType": media_type,
            "size": size_bytes,
            "sizeMB": round(size_mb, 2),
            "durationSeconds": duration_seconds,
            "modifiedTime": file.get("modifiedTime"),
            "thumbnailUrl": file.get("thumbnailLink"),
            "driveUrl": file.get("webViewLink"),
            "downloadUrl": file.get("webContentLink"),
            "isValid": len(errors) == 0,
            "errors": errors,
            "width": width,
            "height": height,
        }
    
    def get_file_stream_url(self, file_id: str) -> Optional[str]:
        """
        Get a streaming URL for a video file.
        
        Note: For Drive files, we use the webContentLink with alt=media
        or embed URL for preview.
        
        Args:
            file_id: Google Drive file ID
            
        Returns:
            Streaming URL or None
        """
        try:
            file = self.service.files().get(
                fileId=file_id,
                fields="webContentLink, webViewLink"
            ).execute()
            
            # For video preview, we can use the embed URL format
            # https://drive.google.com/file/d/{fileId}/preview
            return f"https://drive.google.com/file/d/{file_id}/preview"
            
        except HttpError as e:
            logger.error(f"Error getting stream URL for {file_id}: {e}")
            return None

