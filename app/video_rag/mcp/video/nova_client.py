"""
Amazon Bedrock / Nova client helpers.

All calls to AWS are centralised here so the rest of the codebase stays
clean.  Three responsibilities:

1. Multimodal embeddings  — Amazon Titan Multimodal Embeddings V1.
   Used for BOTH text queries and images, giving a unified vector space.
   This replaces the three separate indexes in the reference implementation
   (CLIP for frames, text-embedding-3-small for transcripts and captions).

2. Scene captioning       — Amazon Nova Pro (vision).
   Accepts a batch of consecutive frames and returns a single scene
   description, which is cheaper and more coherent than one call per frame.

3. RAG answer generation  — Amazon Nova Pro (text).
   Given retrieved context chunks, generates a grounded answer.
"""

import base64
import io
import json
import time
from typing import List

import boto3
import numpy as np
from PIL import Image

from config import get_settings
from logging_config import get_logger, log_performance

settings = get_settings()
logger = get_logger("NovaClient")


# ── Boto3 client (lazy singleton) ──────────────────────────────────────────────

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _invoke(model_id: str, body: dict) -> dict:
    """Call Bedrock InvokeModel and return the parsed response body."""
    start_time = time.perf_counter()
    client = _get_client()
    
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    
    result = json.loads(response["body"].read())
    duration = (time.perf_counter() - start_time) * 1000
    
    logger.debug(
        "Bedrock invocation completed",
        model_id=model_id,
        duration_ms=duration,
        input_tokens=result.get("usage", {}).get("inputTokens"),
        output_tokens=result.get("usage", {}).get("outputTokens"),
    )
    
    return result


# ── 1. Multimodal embeddings ───────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Embed a text string using Amazon Titan Multimodal Embeddings.

    The resulting vector lives in the same space as embed_image() vectors,
    enabling cross-modal similarity search (text query ↔ image index).
    """
    start_time = time.perf_counter()
    
    body = {"inputText": text}
    response = _invoke(settings.NOVA_MULTIMODAL_EMBED_MODEL_ID, body)
    embedding = np.array(response["embedding"], dtype=np.float32)
    
    duration = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Text embedded",
        text_length=len(text),
        embedding_dim=len(embedding),
        duration_ms=duration,
    )
    
    return embedding


def embed_image(image: Image.Image) -> np.ndarray:
    """
    Embed a PIL Image using Amazon Titan Multimodal Embeddings.

    Produces a vector in the same space as embed_text() vectors.
    """
    start_time = time.perf_counter()
    
    body = {"inputImage": _pil_to_b64(image)}
    response = _invoke(settings.NOVA_MULTIMODAL_EMBED_MODEL_ID, body)
    embedding = np.array(response["embedding"], dtype=np.float32)
    
    duration = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Image embedded",
        image_size=f"{image.width}x{image.height}",
        embedding_dim=len(embedding),
        duration_ms=duration,
    )
    
    return embedding


def embed_image_b64(image_b64: str) -> List[float]:
    """Embed a base64-encoded image string (convenience wrapper)."""
    body = {"inputImage": image_b64}
    response = _invoke(settings.NOVA_MULTIMODAL_EMBED_MODEL_ID, body)
    return response["embedding"]


# ── 2. Scene captioning ────────────────────────────────────────────────────────

def caption_scene(frames: List[Image.Image], prompt: str | None = None) -> str:
    """
    Generate a scene caption for a batch of consecutive video frames.

    Sending multiple frames in one call gives Nova Pro temporal context,
    producing more coherent descriptions than per-frame captioning.

    Args:
        frames:  List of PIL Images (consecutive frames from the same scene).
        prompt:  Optional override; falls back to settings.CAPTION_MODEL_PROMPT.

    Returns:
        Scene description string.
    """
    start_time = time.perf_counter()
    prompt = prompt or settings.CAPTION_MODEL_PROMPT

    # Build the multi-image message content.
    content: List[dict] = []
    for frame in frames:
        b64 = _pil_to_b64(frame)
        content.append({
            "image": {
                "format": "jpeg",
                "source": {"bytes": b64},  # base64 string — JSON-serialisable
            }
        })
    content.append({"text": prompt})

    body = {
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": 256, "temperature": 0.2},
    }

    try:
        response = _invoke(settings.NOVA_PRO_MODEL_ID, body)
        caption = response["output"]["message"]["content"][0]["text"].strip()
        
        duration = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "Scene captioned",
            frames_count=len(frames),
            caption_length=len(caption),
            duration_ms=duration,
        )
        
        return caption
    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Scene captioning failed",
            frames_count=len(frames),
            error=str(exc),
            duration_ms=duration,
        )
        return ""


# ── 3. RAG answer generation ───────────────────────────────────────────────────

def generate_rag_answer(question: str, context: str) -> str:
    """
    Use Amazon Nova Pro to answer *question* given retrieved *context*.

    The context is injected as a system-level block so the model treats it
    as ground truth and avoids hallucinating beyond what the video contains.

    Args:
        question: The user's question.
        context:  Retrieved transcript / caption chunks (pre-formatted).

    Returns:
        Grounded natural-language answer.
    """
    start_time = time.perf_counter()
    
    logger.info(
        "Generating RAG answer",
        question=question[:80] + "..." if len(question) > 80 else question,
        context_length=len(context),
    )
    
    system_prompt = (
        "You are a precise video assistant. Answer the user's question "
        "using ONLY the provided video context. If the context does not "
        "contain enough information to answer, say so clearly. "
        "Do not invent details.\n\n"
        f"VIDEO CONTEXT:\n{context}"
    )

    body = {
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": [{"text": question}]}],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.1},
    }

    try:
        response = _invoke(settings.NOVA_PRO_MODEL_ID, body)
        answer = response["output"]["message"]["content"][0]["text"].strip()
        
        duration = (time.perf_counter() - start_time) * 1000
        logger.info(
            "RAG answer generated",
            question_length=len(question),
            context_length=len(context),
            answer_length=len(answer),
            duration_ms=duration,
        )
        
        return answer
    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000
        logger.error(
            "RAG answer generation failed",
            question=question[:50],
            error=str(exc),
            duration_ms=duration,
        )
        return "I encountered an error generating the answer. Please try again."