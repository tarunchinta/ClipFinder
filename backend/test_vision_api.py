"""
Azure AI Vision CLIP Embedding API Test Script
==============================================

VERIFIED WORKING - January 2026

ENDPOINT CONFIGURATION:
=======================
- Endpoint URL: https://<name>.inference.ml.azure.com/score
- Auth Header: "Authorization: Bearer <API_KEY>"
- Deployment Header: "azureml-model-deployment: <DEPLOYMENT_NAME>"  # REQUIRED!

REQUEST FORMAT (VERIFIED WORKING):
==================================
{
    "input_data": {
        "columns": ["image", "text"],      # BOTH columns always required
        "index": [0, 1, ...],              # Row indices (0-based)
        "data": [
            [image_base64, ""],            # Image-only: text must be empty string ""
            ["", "description text"],      # Text-only: image must be empty string ""
            [image_base64, "description"]  # Both: provide both values
        ]
    }
}

IMAGE ENCODING:
===============
- Use base64.encodebytes(image_bytes).decode("utf-8")
- This adds newlines every 76 chars (required by the model)
- Do NOT use base64.b64encode() - it may not work

RESPONSE FORMAT:
================
Returns a list of dicts, one per input row:
- For images: [{"image_features": [768 floats...]}]
- For text:   [{"text_features": [768 floats...]}]
- For both:   [{"image_features": [...], "text_features": [...]}]

Embedding dimension: 768

Run with: python test_vision_api.py
"""

import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ============ CONFIGURATION ============
ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT", "")
API_KEY = os.getenv("AZURE_AI_VISION_KEY", "")
DEPLOYMENT_NAME = "openai-clip-image-text-embedd"


async def test_text_embedding():
    """Test text-only embedding (verified working)."""
    print("\n" + "=" * 60)
    print("  TEST: Text Embedding")
    print("=" * 60)
    
    request_body = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [
                ["", "a photo of a cat"]  # Empty image, text only
            ]
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "azureml-model-deployment": DEPLOYMENT_NAME,
    }
    
    print(f"Request: {json.dumps(request_body, indent=2)}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ENDPOINT, headers=headers, json=request_body)
        
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✓ SUCCESS!")
        if isinstance(data, list) and len(data) > 0:
            features = data[0].get("text_features", [])
            print(f"  Embedding dimension: {len(features)}")
            print(f"  First 5 values: {features[:5]}")
        return True
    else:
        print(f"✗ FAILED: {response.text[:200]}")
        return False


async def test_image_embedding():
    """Test image embedding by downloading a real test image."""
    print("\n" + "=" * 60)
    print("  TEST: Image Embedding")
    print("=" * 60)
    
    # Download a small test image (placeholder image service)
    print("Downloading test image...")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        img_response = await client.get("https://picsum.photos/200/200")
        if img_response.status_code != 200:
            print(f"✗ Failed to download test image: {img_response.status_code}")
            return False
        image_bytes = img_response.content
    
    # Encode to base64 (use encodebytes like the notebook does)
    image_base64 = base64.encodebytes(image_bytes).decode("utf-8")
    print(f"Image size: {len(image_bytes)} bytes, base64 length: {len(image_base64)}")
    
    request_body = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [
                [image_base64, ""]  # Image only, empty text
            ]
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "azureml-model-deployment": DEPLOYMENT_NAME,
    }
    
    print("Sending request...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ENDPOINT, headers=headers, json=request_body)
        
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✓ SUCCESS!")
        if isinstance(data, list) and len(data) > 0:
            features = data[0].get("image_features", [])
            print(f"  Embedding dimension: {len(features)}")
            print(f"  First 5 values: {features[:5]}")
        return True
    else:
        print(f"✗ FAILED: {response.text[:300]}")
        return False


async def main():
    print("\n" + "=" * 60)
    print("  AZURE AI VISION CLIP EMBEDDING TEST")
    print("=" * 60)
    
    print(f"\nEndpoint: {ENDPOINT}")
    print(f"Deployment: {DEPLOYMENT_NAME}")
    print(f"API Key: {'*' * 10 + API_KEY[-4:] if API_KEY else '(not set)'}")
    
    if not ENDPOINT or not API_KEY:
        print("\n✗ ERROR: Missing AZURE_AI_VISION_ENDPOINT or AZURE_AI_VISION_KEY in .env")
        return
    
    # Run tests
    text_ok = await test_text_embedding()
    image_ok = await test_image_embedding()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Text embedding:  {'✓ Working' if text_ok else '✗ Failed'}")
    print(f"  Image embedding: {'✓ Working' if image_ok else '✗ Failed'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
