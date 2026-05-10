"""Embedding service for generating text embeddings using Azure OpenAI."""

import logging
from typing import Optional

import httpx

from app.config import get_settings
from app.observability import trace_embedding_generation

logger = logging.getLogger(__name__)

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIMENSION = 1536


class EmbeddingService:
    """Service for generating text embeddings using Azure OpenAI."""
    
    def __init__(self):
        """Initialize the embedding service with Azure OpenAI credentials."""
        settings = get_settings()
        self.endpoint = settings.azure_openai_endpoint_sample_full
        self.api_key = settings.azure_openai_api_key
        
        if not self.endpoint or not self.api_key:
            logger.warning(
                "Azure OpenAI credentials not configured. "
                "Embedding generation will be disabled."
            )
    
    @property
    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(self.endpoint and self.api_key)
    
    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        """
        Generate an embedding vector for a single text string.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding vector,
            or None if generation fails
        """
        if not self.is_configured:
            logger.warning("Azure OpenAI not configured, skipping embedding generation")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None

        input_summary = text[:200] + ("..." if len(text) > 200 else "")
        with trace_embedding_generation(
            name="text_embedding",
            model="text-embedding-3-small",
            input_summary=input_summary,
        ):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.endpoint,
                        headers={
                            "Content-Type": "application/json",
                            "api-key": self.api_key,
                        },
                        json={
                            "input": text,
                        }
                    )
                    response.raise_for_status()

                    data = response.json()
                    embedding = data["data"][0]["embedding"]

                    logger.debug(f"Generated embedding for text: {text[:50]}...")
                    return embedding
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Azure OpenAI API error: {e.response.status_code} - {e.response.text}"
                )
                return None
            except httpx.RequestError as e:
                logger.error(f"Azure OpenAI request error: {e}")
                return None
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected response format from Azure OpenAI: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                return None
    
    async def generate_embeddings_batch(
        self, 
        texts: list[str],
        batch_size: int = 16
    ) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Azure OpenAI supports batch input, which is more efficient than
        individual requests.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Number of texts to process in each API call
            
        Returns:
            List of embedding vectors (or None for failed texts),
            in the same order as input texts
        """
        if not self.is_configured:
            logger.warning("Azure OpenAI not configured, skipping batch embedding generation")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        results: list[Optional[list[float]]] = [None] * len(texts)
        
        # Process in batches
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Filter out empty texts but track their positions
            non_empty_indices = []
            non_empty_texts = []
            for i, text in enumerate(batch_texts):
                if text and text.strip():
                    non_empty_indices.append(batch_start + i)
                    non_empty_texts.append(text)
            
            if not non_empty_texts:
                continue

            input_summary = f"batch of {len(non_empty_texts)} texts"
            with trace_embedding_generation(
                name="text_embedding_batch",
                model="text-embedding-3-small",
                input_summary=input_summary,
            ):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            self.endpoint,
                            headers={
                                "Content-Type": "application/json",
                                "api-key": self.api_key,
                            },
                            json={
                                "input": non_empty_texts,
                            }
                        )
                        response.raise_for_status()

                        data = response.json()
                        embeddings = data["data"]

                        # Map embeddings back to their original positions
                        # Azure OpenAI returns embeddings in the same order as input
                        for i, emb_data in enumerate(embeddings):
                            original_index = non_empty_indices[i]
                            results[original_index] = emb_data["embedding"]

                        logger.info(
                            f"Generated {len(non_empty_texts)} embeddings in batch "
                            f"({batch_start}-{batch_end} of {len(texts)})"
                        )
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"Azure OpenAI API error in batch: {e.response.status_code} - {e.response.text}"
                    )
                except httpx.RequestError as e:
                    logger.error(f"Azure OpenAI request error in batch: {e}")
                except (KeyError, IndexError) as e:
                    logger.error(f"Unexpected response format from Azure OpenAI batch: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in batch embedding generation: {e}")
        
        return results


# Singleton instance for convenience
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
