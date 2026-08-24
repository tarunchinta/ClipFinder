"""Vision embedding service using Gemini Embedding 2 (Google AI Studio).

API reference: https://ai.google.dev/gemini-api/docs/embeddings

REST endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent
    Header: x-goog-api-key

Text query (asymmetric retrieval):
    {"content": {"parts": [{"text": "task: search result | query: ..."}]}, "output_dimensionality": 768}

Image:
    {"content": {"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": "<base64>"}}]}, "output_dimensionality": 768}

Response:
    {"embedding": {"values": [768 floats]}}
"""

import asyncio
import base64
import logging
from typing import Any, Literal, Optional

import httpx

from app.config import get_settings
from app.observability import trace_embedding_generation

logger = logging.getLogger(__name__)

GEMINI_EMBED_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
SEARCH_QUERY_PREFIX = "task: search result | query: "

VISION_EMBEDDING_DIMENSION = 768

DRIVE_THUMBNAIL_MAX_PX = 336


def normalize_drive_thumbnail_url(url: str, max_px: int = DRIVE_THUMBNAIL_MAX_PX) -> str:
    """Rewrite Google Drive thumbnail URL size param (e.g. =s220 -> =s336)."""
    if "=s" in url:
        return url.rsplit("=s", 1)[0] + f"=s{max_px}"
    return url


def _detect_image_mime_type(image_bytes: bytes) -> str:
    """Detect JPEG or PNG from magic bytes; default to JPEG for ffmpeg frames."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    return "image/jpeg"


def _format_search_query(text: str) -> str:
    """Format a search query for asymmetric retrieval per Gemini Embedding 2 docs."""
    stripped = text.strip()
    if stripped.startswith("task:"):
        return stripped
    return f"{SEARCH_QUERY_PREFIX}{stripped}"


class VisionEmbeddingService:
    """Service for generating multimodal embeddings using Gemini Embedding 2."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_embedding_model
        self.output_dimensionality = settings.gemini_embedding_dimension
        self._embed_url = f"{GEMINI_EMBED_BASE_URL}/{self.model}:embedContent"

        if not self.api_key:
            logger.warning(
                "Gemini API key not configured (GEMINI_API_KEY or GOOGLE_AI_VISION_API_KEY). "
                "Vision embedding generation will be disabled."
            )

    @property
    def is_configured(self) -> bool:
        """Check if Gemini Embedding 2 is properly configured."""
        return bool(self.api_key and self.model)

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _parse_response(self, data: Any) -> Optional[list[float]]:
        """Extract embedding values from Gemini embedContent response."""
        if not isinstance(data, dict):
            logger.error(f"Unexpected Gemini response type: {type(data)}")
            return None

        embedding = data.get("embedding")
        if isinstance(embedding, dict):
            values = embedding.get("values")
            if isinstance(values, list) and values:
                return values

        logger.error(f"Unexpected Gemini response format: {data}")
        return None

    async def _call_gemini_embed(
        self,
        parts: list[dict[str, Any]],
        trace_name: str,
        input_summary: str,
    ) -> Optional[list[float]]:
        """Call Gemini embedContent with retry on rate limits."""
        request_body = {
            "content": {"parts": parts},
            "output_dimensionality": self.output_dimensionality,
        }

        with trace_embedding_generation(
            name=trace_name,
            model=self.model,
            input_summary=input_summary,
        ):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            self._embed_url,
                            headers=self._get_headers(),
                            json=request_body,
                        )
                        response.raise_for_status()
                        embedding = self._parse_response(response.json())
                        if embedding:
                            logger.debug(
                                f"Generated embedding (dim={len(embedding)}, "
                                f"expected={self.output_dimensionality})"
                            )
                        return embedding

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(f"Gemini rate limited, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    logger.error(
                        f"Gemini API error: {e.response.status_code} - {e.response.text}"
                    )
                    return None
                except httpx.RequestError as e:
                    logger.error(f"Gemini request error: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error generating vision embedding: {e}")
                    return None

        return None

    async def _get_fresh_thumbnail_url(
        self,
        drive_file_id: str,
        google_access_token: str,
    ) -> Optional[str]:
        """Get a fresh thumbnail URL from Google Drive API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"https://www.googleapis.com/drive/v3/files/{drive_file_id}",
                    params={"fields": "thumbnailLink"},
                    headers={"Authorization": f"Bearer {google_access_token}"},
                )
                response.raise_for_status()
                data = response.json()
                thumbnail_url = data.get("thumbnailLink")

                if thumbnail_url:
                    return normalize_drive_thumbnail_url(thumbnail_url)
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
        google_access_token: Optional[str] = None,
    ) -> Optional[bytes]:
        """Download an image from a URL."""
        try:
            headers = {}
            if google_access_token and "googleapis.com" in image_url:
                headers["Authorization"] = f"Bearer {google_access_token}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    image_url,
                    follow_redirects=True,
                    headers=headers if headers else None,
                )
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error downloading image: {e.response.status_code} - "
                f"URL: {image_url[:100]}..."
            )
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image: {e}")
            return None

    async def generate_embedding(
        self,
        image_url_or_text: str,
        mode: Literal["image", "text"] = "image",
        drive_file_id: Optional[str] = None,
        google_access_token: Optional[str] = None,
    ) -> Optional[list[float]]:
        """
        Generate an embedding vector for a single image URL or text query.

        Args:
            image_url_or_text: Image URL (mode="image") or text query (mode="text")
            mode: "image" for image embedding, "text" for search query embedding
            drive_file_id: Optional Drive file ID to fetch fresh thumbnail (image mode)
            google_access_token: Optional Google OAuth token for Drive API access

        Returns:
            Embedding vector or None if generation fails
        """
        if not self.is_configured:
            logger.warning("Gemini Embedding 2 not configured, skipping embedding generation")
            return None

        if not image_url_or_text or not image_url_or_text.strip():
            logger.warning(f"Empty {mode} input provided for embedding generation")
            return None

        if mode == "text":
            formatted_text = _format_search_query(image_url_or_text)
            input_summary = formatted_text[:200] + (
                "..." if len(formatted_text) > 200 else ""
            )
            return await self._call_gemini_embed(
                parts=[{"text": formatted_text}],
                trace_name="vision_embedding",
                input_summary=input_summary,
            )

        actual_url = image_url_or_text
        if drive_file_id and google_access_token:
            fresh_url = await self._get_fresh_thumbnail_url(drive_file_id, google_access_token)
            if fresh_url:
                actual_url = fresh_url
                logger.debug(f"Using fresh thumbnail URL for file {drive_file_id}")

        image_bytes = await self._download_image(actual_url, google_access_token)
        if not image_bytes:
            logger.warning(f"Failed to download image from URL: {actual_url[:100]}...")
            return None

        return await self.generate_embedding_from_image_bytes(image_bytes)

    async def generate_embedding_from_image_bytes(
        self, image_bytes: bytes
    ) -> Optional[list[float]]:
        """
        Generate an embedding for image bytes (e.g. extracted video frame).

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            Embedding vector (768-dim by default), or None if generation fails
        """
        if not self.is_configured:
            logger.warning("Gemini Embedding 2 not configured, skipping embedding generation")
            return None
        if not image_bytes:
            return None

        mime_type = _detect_image_mime_type(image_bytes)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return await self._call_gemini_embed(
            parts=[{"inline_data": {"mime_type": mime_type, "data": image_base64}}],
            trace_name="vision_embedding_from_bytes",
            input_summary="image_bytes",
        )

    async def generate_text_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate an embedding for a text search query.

        Uses asymmetric retrieval formatting so text queries can be compared
        against image/frame embeddings in the shared Gemini embedding space.
        """
        return await self.generate_embedding(text, mode="text")

    async def generate_document_text_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate an embedding for document-side text (e.g. a transcript segment).

        Unlike generate_text_embedding, no query prefix is applied: stored text is
        embedded as-is so asymmetric query embeddings can retrieve it.
        """
        if not self.is_configured:
            logger.warning("Gemini Embedding 2 not configured, skipping embedding generation")
            return None
        stripped = (text or "").strip()
        if not stripped:
            return None
        input_summary = stripped[:200] + ("..." if len(stripped) > 200 else "")
        return await self._call_gemini_embed(
            parts=[{"text": stripped}],
            trace_name="transcript_text_embedding",
            input_summary=input_summary,
        )

    async def generate_embeddings_batch(
        self,
        image_urls: list[str],
        drive_file_ids: Optional[list[str]] = None,
        google_access_token: Optional[str] = None,
        batch_size: int = 1,
    ) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple images (one API call per image).

        Args:
            image_urls: List of image URLs
            drive_file_ids: Optional Drive file IDs parallel to image_urls
            google_access_token: Optional Google OAuth token
            batch_size: Ignored; kept for API compatibility (Gemini embeds one image per call)

        Returns:
            List of embedding vectors in the same order as input URLs
        """
        if not self.is_configured:
            logger.warning("Gemini Embedding 2 not configured, skipping batch embedding generation")
            return [None] * len(image_urls)

        if not image_urls:
            return []

        results: list[Optional[list[float]]] = [None] * len(image_urls)

        for i, url in enumerate(image_urls):
            if not url or not url.strip():
                continue
            file_id = drive_file_ids[i] if drive_file_ids and i < len(drive_file_ids) else None
            results[i] = await self.generate_embedding(
                url,
                mode="image",
                drive_file_id=file_id,
                google_access_token=google_access_token,
            )

        generated = sum(1 for r in results if r is not None)
        logger.info(f"Generated {generated}/{len(image_urls)} vision embeddings in batch")
        return results


_vision_embedding_service: Optional[VisionEmbeddingService] = None


def get_vision_embedding_service() -> VisionEmbeddingService:
    """Get the singleton VisionEmbeddingService instance."""
    global _vision_embedding_service
    if _vision_embedding_service is None:
        _vision_embedding_service = VisionEmbeddingService()
    return _vision_embedding_service
