"""Vision embedding service for generating image embeddings using Azure AI Vision CLIP.

VERIFIED WORKING API FORMAT (from test_vision_api.py):
- Endpoint URL: https://<name>.inference.ml.azure.com/score
- Auth Header: "Authorization: Bearer <API_KEY>"
- Deployment Header: "azureml-model-deployment: <DEPLOYMENT_NAME>"  # REQUIRED!

REQUEST FORMAT:
{
    "input_data": {
        "columns": ["image", "text"],      # BOTH columns always required
        "index": [0, 1, ...],              # Row indices (0-based)
        "data": [
            [image_base64, ""],            # Image-only: text must be empty string ""
            ["", "description text"],      # Text-only: image must be empty string ""
        ]
    }
}

IMAGE ENCODING:
- Use base64.encodebytes(image_bytes).decode("utf-8")
- This adds newlines every 76 chars (required by the model)

RESPONSE FORMAT:
- For images: [{"image_features": [768 floats...]}]
- For text:   [{"text_features": [768 floats...]}]
"""

import base64
import logging
from typing import Optional, Literal

import httpx

from app.config import get_settings
from app.observability import trace_embedding_generation

logger = logging.getLogger(__name__)

# Embedding dimension for CLIP model
VISION_EMBEDDING_DIMENSION = 768


class VisionEmbeddingService:
    """Service for generating image embeddings using Azure AI Vision CLIP model."""
    
    def __init__(self):
        """Initialize the vision embedding service with Azure AI Vision credentials."""
        settings = get_settings()
        self.endpoint = settings.azure_ai_vision_endpoint
        self.api_key = settings.azure_ai_vision_key
        self.deployment_name = settings.azure_ai_vision_deployment_name
        
        if not self.endpoint or not self.api_key:
            logger.warning(
                "Azure AI Vision credentials not configured. "
                "Vision embedding generation will be disabled."
            )
    
    @property
    def is_configured(self) -> bool:
        """Check if Azure AI Vision is properly configured."""
        return bool(self.endpoint and self.api_key and self.deployment_name)
    
    def _get_headers(self) -> dict[str, str]:
        """Get the required headers for Azure ML API calls."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "azureml-model-deployment": self.deployment_name,  # REQUIRED!
        }
    
    async def _get_fresh_thumbnail_url(
        self, 
        drive_file_id: str, 
        google_access_token: str
    ) -> Optional[str]:
        """
        Get a fresh thumbnail URL from Google Drive API.
        
        Args:
            drive_file_id: Google Drive file ID
            google_access_token: User's Google OAuth access token
            
        Returns:
            Fresh thumbnail URL or None if failed
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"https://www.googleapis.com/drive/v3/files/{drive_file_id}",
                    params={"fields": "thumbnailLink"},
                    headers={"Authorization": f"Bearer {google_access_token}"}
                )
                response.raise_for_status()
                data = response.json()
                thumbnail_url = data.get("thumbnailLink")
                
                if thumbnail_url:
                    # Increase thumbnail size for better embeddings
                    if "=s" in thumbnail_url:
                        thumbnail_url = thumbnail_url.rsplit("=s", 1)[0] + "=s400"
                    return thumbnail_url
                return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Drive API error getting thumbnail: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting fresh thumbnail URL: {e}")
            return None
    
    async def _download_image(
        self, 
        image_url: str,
        google_access_token: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Download an image from a URL.
        
        Args:
            image_url: URL of the image to download
            google_access_token: Optional Google access token (not typically needed for thumbnailLink)
            
        Returns:
            Image bytes or None if download fails
        """
        try:
            headers = {}
            # Google's lh3.googleusercontent.com URLs don't need auth once we have the link
            # But some URLs might need it, so include if provided
            if google_access_token and "googleapis.com" in image_url:
                headers["Authorization"] = f"Bearer {google_access_token}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    image_url, 
                    follow_redirects=True,
                    headers=headers if headers else None
                )
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image: {e.response.status_code} - URL: {image_url[:100]}...")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image: {e}")
            return None
    
    def _parse_response(
        self, 
        data: any, 
        mode: Literal["image", "text"]
    ) -> Optional[list[float]]:
        """
        Parse the API response to extract embedding features.
        
        Args:
            data: Response JSON data
            mode: "image" to extract image_features, "text" to extract text_features
            
        Returns:
            Embedding vector or None if parsing fails
        """
        feature_key = "image_features" if mode == "image" else "text_features"
        
        # Response format: [{"image_features": [...]} or {"text_features": [...]}]
        if isinstance(data, list) and len(data) > 0:
            result = data[0]
            if isinstance(result, dict) and feature_key in result:
                return result[feature_key]
        
        logger.error(f"Unexpected response format from Azure AI Vision: {data}")
        return None
    
    def _parse_batch_response(
        self, 
        data: any, 
        mode: Literal["image", "text"]
    ) -> list[Optional[list[float]]]:
        """
        Parse the batch API response to extract embedding features.
        
        Args:
            data: Response JSON data (list of results)
            mode: "image" to extract image_features, "text" to extract text_features
            
        Returns:
            List of embedding vectors (or None for failed items)
        """
        feature_key = "image_features" if mode == "image" else "text_features"
        embeddings = []
        
        # Response format: [{"image_features": [...]}, {"image_features": [...]}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and feature_key in item:
                    embeddings.append(item[feature_key])
                else:
                    embeddings.append(None)
        
        return embeddings
    
    async def generate_embedding(
        self, 
        image_url_or_text: str,
        mode: Literal["image", "text"] = "image",
        drive_file_id: Optional[str] = None,
        google_access_token: Optional[str] = None
    ) -> Optional[list[float]]:
        """
        Generate an embedding vector for a single image URL or text query.
        
        Args:
            image_url_or_text: URL of the image (for mode="image") or text query (for mode="text")
            mode: "image" for image embedding, "text" for text embedding (CLIP text encoder)
            drive_file_id: Optional Google Drive file ID to fetch fresh thumbnail (only for image mode)
            google_access_token: Optional Google OAuth token for Drive API access
            
        Returns:
            A list of floats representing the embedding vector,
            or None if generation fails
        """
        if not self.is_configured:
            logger.warning("Azure AI Vision not configured, skipping embedding generation")
            return None
        
        if not image_url_or_text or not image_url_or_text.strip():
            logger.warning(f"Empty {mode} input provided for embedding generation")
            return None
        
        # Prepare the request data based on mode
        if mode == "text":
            # Text embedding: empty image, text filled
            request_body = {
                "input_data": {
                    "columns": ["image", "text"],
                    "index": [0],
                    "data": [["", image_url_or_text]]  # [empty_image, text]
                }
            }
        else:
            # Image embedding: download and encode image
            actual_url = image_url_or_text
            if drive_file_id and google_access_token:
                fresh_url = await self._get_fresh_thumbnail_url(drive_file_id, google_access_token)
                if fresh_url:
                    actual_url = fresh_url
                    logger.debug(f"Using fresh thumbnail URL for file {drive_file_id}")
            
            # Download the image
            image_bytes = await self._download_image(actual_url, google_access_token)
            if not image_bytes:
                logger.warning(f"Failed to download image from URL: {actual_url[:100]}...")
                return None
            
            # Convert to base64 using encodebytes (adds newlines every 76 chars - REQUIRED!)
            image_base64 = base64.encodebytes(image_bytes).decode('utf-8')
            
            request_body = {
                "input_data": {
                    "columns": ["image", "text"],
                    "index": [0],
                    "data": [[image_base64, ""]]  # [image, empty_text]
                }
            }

        input_summary = image_url_or_text[:200] + ("..." if len(image_url_or_text) > 200 else "") if mode == "text" else "image"
        with trace_embedding_generation(
            name="vision_embedding",
            model="azure-clip",
            input_summary=input_summary,
        ):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.endpoint,
                        headers=self._get_headers(),
                        json=request_body
                    )
                    response.raise_for_status()

                    data = response.json()
                    embedding = self._parse_response(data, mode)

                    if embedding:
                        logger.debug(f"Generated {mode} embedding (dim={len(embedding)})")
                        return embedding

                    return None

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Azure AI Vision API error: {e.response.status_code} - {e.response.text}"
                )
                return None
            except httpx.RequestError as e:
                logger.error(f"Azure AI Vision request error: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error generating vision embedding: {e}")
                return None

    async def generate_embedding_from_image_bytes(self, image_bytes: bytes) -> Optional[list[float]]:
        """
        Generate an embedding for image bytes (e.g. extracted video frame).
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG etc.)
            
        Returns:
            A list of floats (768-dim CLIP embedding), or None if generation fails
        """
        if not self.is_configured:
            logger.warning("Azure AI Vision not configured, skipping embedding generation")
            return None
        if not image_bytes:
            return None
        image_base64 = base64.encodebytes(image_bytes).decode("utf-8")
        request_body = {
            "input_data": {
                "columns": ["image", "text"],
                "index": [0],
                "data": [[image_base64, ""]],
            }
        }
        with trace_embedding_generation(
            name="vision_embedding_from_bytes",
            model="azure-clip",
            input_summary="image_bytes",
        ):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.endpoint,
                        headers=self._get_headers(),
                        json=request_body,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return self._parse_response(data, "image")
            except Exception as e:
                logger.error(f"Vision embedding from bytes failed: {e}")
                return None

    async def generate_text_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate an embedding for a text query using CLIP's text encoder.
        
        This is useful for searching images by text description.
        The resulting embedding can be compared with image embeddings using cosine similarity.
        
        Args:
            text: The text query to embed
            
        Returns:
            A list of floats representing the embedding vector, or None if generation fails
        """
        return await self.generate_embedding(text, mode="text")
    
    async def generate_embeddings_batch(
        self, 
        image_urls: list[str],
        drive_file_ids: Optional[list[str]] = None,
        google_access_token: Optional[str] = None,
        batch_size: int = 1
    ) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple images.
        
        NOTE: batch_size=1 is required because the Azure ML CLIP endpoint
        does not reliably support batching multiple images in a single request.
        Each image is processed in a separate API call for reliability.
        
        Args:
            image_urls: List of image URLs to generate embeddings for
            drive_file_ids: Optional list of Google Drive file IDs (parallel to image_urls)
            google_access_token: Optional Google OAuth token for Drive API access
            batch_size: Number of images to process in each API call (default 1 for reliability)
            
        Returns:
            List of embedding vectors (or None for failed images),
            in the same order as input URLs
        """
        if not self.is_configured:
            logger.warning("Azure AI Vision not configured, skipping batch embedding generation")
            return [None] * len(image_urls)
        
        if not image_urls:
            return []
        
        results: list[Optional[list[float]]] = [None] * len(image_urls)
        
        # Process in batches
        for batch_start in range(0, len(image_urls), batch_size):
            batch_end = min(batch_start + batch_size, len(image_urls))
            batch_urls = image_urls[batch_start:batch_end]
            batch_file_ids = (
                drive_file_ids[batch_start:batch_end] 
                if drive_file_ids else [None] * len(batch_urls)
            )
            
            # Filter out empty URLs but track their positions
            non_empty_indices = []
            non_empty_urls = []
            non_empty_file_ids = []
            for i, (url, file_id) in enumerate(zip(batch_urls, batch_file_ids)):
                if url and url.strip():
                    non_empty_indices.append(batch_start + i)
                    non_empty_urls.append(url)
                    non_empty_file_ids.append(file_id)
            
            if not non_empty_urls:
                continue
            
            # Download all images in this batch and encode to base64
            images_base64 = []
            valid_indices = []
            
            for i, (url, file_id) in enumerate(zip(non_empty_urls, non_empty_file_ids)):
                # Try to get fresh thumbnail URL if we have credentials
                actual_url = url
                if file_id and google_access_token:
                    fresh_url = await self._get_fresh_thumbnail_url(file_id, google_access_token)
                    if fresh_url:
                        actual_url = fresh_url
                
                image_bytes = await self._download_image(actual_url, google_access_token)
                if image_bytes:
                    # Use encodebytes (adds newlines every 76 chars - REQUIRED!)
                    images_base64.append(base64.encodebytes(image_bytes).decode('utf-8'))
                    valid_indices.append(non_empty_indices[i])
                else:
                    logger.warning(f"Failed to download image at index {non_empty_indices[i]}")
            
            if not images_base64:
                continue
            
            # Build the batch request with both columns and 2D data array
            request_body = {
                "input_data": {
                    "columns": ["image", "text"],
                    "index": list(range(len(images_base64))),
                    "data": [[img_b64, ""] for img_b64 in images_base64]  # [image, empty_text] for each
                }
            }

            with trace_embedding_generation(
                name="vision_embedding_batch",
                model="azure-clip",
                input_summary=f"batch of {len(images_base64)} images",
            ):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            self.endpoint,
                            headers=self._get_headers(),
                            json=request_body
                        )
                        response.raise_for_status()

                        data = response.json()

                        # Parse response to extract image_features
                        embeddings = self._parse_batch_response(data, mode="image")

                        # Map embeddings back to original positions
                        for i, embedding in enumerate(embeddings):
                            if i < len(valid_indices) and embedding:
                                results[valid_indices[i]] = embedding

                        logger.info(
                            f"Generated {len(embeddings)} vision embeddings in batch "
                            f"({batch_start}-{batch_end} of {len(image_urls)})"
                        )

                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"Azure AI Vision API error in batch: {e.response.status_code} - {e.response.text}"
                    )
                except httpx.RequestError as e:
                    logger.error(f"Azure AI Vision request error in batch: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in batch vision embedding generation: {e}")
        
        return results


# Singleton instance for convenience
_vision_embedding_service: Optional[VisionEmbeddingService] = None


def get_vision_embedding_service() -> VisionEmbeddingService:
    """Get the singleton VisionEmbeddingService instance."""
    global _vision_embedding_service
    if _vision_embedding_service is None:
        _vision_embedding_service = VisionEmbeddingService()
    return _vision_embedding_service
