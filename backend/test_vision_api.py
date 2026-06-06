"""
Gemini Embedding 2 API Test Script
==================================

API reference: https://ai.google.dev/gemini-api/docs/embeddings

Tests text and image embedding via Google AI Studio embedContent endpoint.

Run with: python test_vision_api.py
"""

import asyncio
import base64
import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GEMINI_EMBED_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
OUTPUT_DIM = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_VISION_API_KEY", "")
EMBED_URL = f"{GEMINI_EMBED_BASE_URL}/{MODEL}:embedContent"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def embed(parts: list[dict], label: str) -> list[float] | None:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }
    body = {
        "content": {"parts": parts},
        "output_dimensionality": OUTPUT_DIM,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(EMBED_URL, headers=headers, json=body)

    print(f"\n{label} — Status: {response.status_code}")
    if response.status_code != 200:
        print(f"  FAILED: {response.text[:300]}")
        return None

    data = response.json()
    values = data.get("embedding", {}).get("values", [])
    if not values:
        print(f"  FAILED: unexpected response {data}")
        return None

    print(f"  OK — dimension: {len(values)}, first 5: {values[:5]}")
    return values


async def test_text_embedding() -> tuple[bool, list[float] | None]:
    print("\n" + "=" * 60)
    print("  TEST: Text Embedding (search query)")
    print("=" * 60)

    values = await embed(
        [{"text": "task: search result | query: a photo of a cat"}],
        "Text query",
    )
    return (values is not None and len(values) == OUTPUT_DIM, values)


async def test_image_embedding() -> tuple[bool, list[float] | None]:
    print("\n" + "=" * 60)
    print("  TEST: Image Embedding")
    print("=" * 60)

    print("Downloading test image...")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        img_response = await client.get("https://picsum.photos/200/200")
        if img_response.status_code != 200:
            print(f"  FAILED to download test image: {img_response.status_code}")
            return False, None
        image_bytes = img_response.content

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    print(f"Image size: {len(image_bytes)} bytes")

    values = await embed(
        [{"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}],
        "Image",
    )
    return (values is not None and len(values) == OUTPUT_DIM, values)


async def test_cross_modal(text_vec: list[float] | None, image_vec: list[float] | None) -> bool:
    print("\n" + "=" * 60)
    print("  TEST: Cross-modal similarity sanity check")
    print("=" * 60)

    if not text_vec or not image_vec:
        print("  SKIPPED (text or image test failed)")
        return False

    sim = cosine_similarity(text_vec, image_vec)
    print(f"  Cosine similarity (cat query vs random image): {sim:.4f}")
    print("  (Random image won't match 'cat' strongly; this confirms vectors are valid.)")
    return True


async def main():
    print("\n" + "=" * 60)
    print("  GEMINI EMBEDDING 2 TEST")
    print("=" * 60)
    print(f"\nModel: {MODEL}")
    print(f"Output dimension: {OUTPUT_DIM}")
    print(f"API Key: {'*' * 10 + API_KEY[-4:] if API_KEY else '(not set)'}")

    if not API_KEY:
        print("\nERROR: Set GEMINI_API_KEY or GOOGLE_AI_VISION_API_KEY in backend/.env")
        return

    text_ok, text_vec = await test_text_embedding()
    image_ok, image_vec = await test_image_embedding()
    cross_ok = await test_cross_modal(text_vec, image_vec)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Text embedding:  {'OK' if text_ok else 'FAILED'}")
    print(f"  Image embedding: {'OK' if image_ok else 'FAILED'}")
    print(f"  Cross-modal:     {'OK' if cross_ok else 'SKIPPED/FAILED'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
