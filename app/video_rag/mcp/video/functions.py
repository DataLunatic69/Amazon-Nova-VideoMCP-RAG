"""
Pixeltable UDFs (user-defined functions).

These are decorated with @pxt.udf so Pixeltable can call them inside
computed-column expressions and handle serialisation automatically.
"""

import pixeltable as pxt
from PIL import Image


@pxt.udf
def extract_transcript_text(transcript: pxt.Json) -> str:
    """
    Pull the plain text out of a Whisper / AWS Transcribe response dict.

    Both services return ``{"text": "…", …}`` at the top level.
    """
    return str(transcript.get("text", ""))


@pxt.udf
def resize_image(image: pxt.Image, width: int, height: int) -> pxt.Image:
    """
    Resize *image* to fit within *width* × *height* (maintains aspect ratio).

    PIL.Image.thumbnail() modifies in-place, so we work on a copy.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    copy = image.copy()
    copy.thumbnail((width, height), Image.LANCZOS)
    return copy