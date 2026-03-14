"""
Audio transcription backends.

Primary:  AWS Transcribe  (boto3 — uses the same AWS credentials).
Fallback: OpenAI Whisper  (openai SDK — set USE_AWS_TRANSCRIBE=False in .env).

Both return a dict ``{"text": "…"}`` so the downstream UDF
(extract_transcript_text) works identically with either backend.
"""

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import boto3
from loguru import logger

from config import get_settings

settings = get_settings()
logger = logger.bind(name="Transcription")


# ── AWS Transcribe ─────────────────────────────────────────────────────────────

def transcribe_with_aws(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe *audio_path* using AWS Transcribe (async job).

    Uploads the audio to a temporary S3 location, starts a transcription job,
    polls until completion, then returns ``{"text": "…"}``.

    Note: requires an S3 bucket.  The bucket name is derived from the AWS
    account via the ``AWS_S3_TRANSCRIBE_BUCKET`` env var (see config).
    """
    s3 = boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    transcribe = boto3.client(
        "transcribe",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    bucket = getattr(settings, "AWS_S3_TRANSCRIBE_BUCKET", None)
    if not bucket:
        logger.warning("AWS_S3_TRANSCRIBE_BUCKET not set — falling back to Whisper.")
        return transcribe_with_whisper(audio_path)

    audio_path = Path(audio_path)
    s3_key = f"transcribe-tmp/{uuid.uuid4().hex}/{audio_path.name}"
    job_name = f"vr-{uuid.uuid4().hex[:12]}"

    try:
        # Upload audio chunk to S3.
        s3.upload_file(str(audio_path), bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"

        # Start transcription job.
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat=audio_path.suffix.lstrip(".").lower() or "mp3",
            LanguageCode="en-US",
        )

        # Poll until done.
        for _ in range(120):  # max ~4 minutes
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            state = status["TranscriptionJob"]["TranscriptionJobStatus"]
            if state == "COMPLETED":
                break
            if state == "FAILED":
                reason = status["TranscriptionJob"].get("FailureReason", "unknown")
                logger.error(f"AWS Transcribe job '{job_name}' failed: {reason}")
                return {"text": ""}
            time.sleep(2)
        else:
            logger.error(f"Transcription job '{job_name}' timed out.")
            return {"text": ""}

        # Download and parse transcript.
        transcript_uri = (
            status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        )
        import urllib.request
        with urllib.request.urlopen(transcript_uri) as resp:
            transcript_data = json.loads(resp.read().decode())

        text = transcript_data["results"]["transcripts"][0]["transcript"]
        return {"text": text}

    finally:
        # Clean up S3 object.
        try:
            s3.delete_object(Bucket=bucket, Key=s3_key)
        except Exception:
            pass


# ── OpenAI Whisper fallback ────────────────────────────────────────────────────

def transcribe_with_whisper(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe *audio_path* using the OpenAI Whisper API.

    Returns ``{"text": "…"}`` to match the AWS Transcribe return shape.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=settings.WHISPER_MODEL,
            file=audio_file,
            response_format="json",
        )

    return {"text": response.text}