"""
src/ingestion/image_processor.py

FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 2.0

Enhanced Production Image Processor

Capabilities
------------
1. Embedded image extraction from PDFs
2. OCR scanned pages
3. OCR embedded charts/tables/screenshots
4. Gemma4 multimodal vision extraction
5. Direct image-file support (.png/.jpg/.jpeg/.bmp/.webp/.tiff)
6. Memory-safe temp image handling
7. Multiprocess OCR
8. Chart/table extraction
9. Decorative image handling
10. Metadata-aware extraction

Outputs merge into:
    state.raw_text
    state.table_cells

Constraints
-----------
- fully local
- Ollama multimodal
- pytesseract OCR
- memory-safe
- production-grade fault tolerance
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

OCR_DPI = 200

MAX_IMAGES_PER_DOC = 250

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
}

VISION_PROMPT_TEMPLATE = """
You are analyzing an image from a financial document.

Tasks:
1. Extract ALL visible text.
2. Extract ALL visible numbers.
3. Detect charts/tables/financial graphics.
4. Detect labels and values.
5. Return valid JSON only.

Format:

{
  "type": "chart|table|screenshot|document|logo|other",
  "title": "<title>",
  "text": "<all visible text>",
  "data": [
    {
      "label": "<label>",
      "value": "<value>",
      "unit": "<unit>"
    }
  ]
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# OCR Worker
# ─────────────────────────────────────────────────────────────────────────────


def _ocr_worker(image_path: str) -> str:

    try:

        import pytesseract
        from PIL import Image

        img = Image.open(image_path)

        text = pytesseract.image_to_string(img)

        return text.strip()

    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedImage
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedImage:

    __slots__ = (
        "page_number",
        "image_index",
        "width",
        "height",
        "format",
        "source",
        "ocr_text",
        "vision_data",
        "temp_image_path",
    )

    def __init__(
        self,
        page_number: int,
        image_index: int,
        width: int = 0,
        height: int = 0,
        format: str = "png",
        source: str = "",
        ocr_text: str = "",
        vision_data: Optional[Dict] = None,
        temp_image_path: str = "",
    ):

        self.page_number = page_number
        self.image_index = image_index
        self.width = width
        self.height = height
        self.format = format
        self.source = source
        self.ocr_text = ocr_text
        self.vision_data = vision_data or {}
        self.temp_image_path = temp_image_path

    @property
    def has_data(self):

        return bool(
            self.ocr_text.strip()
        ) or bool(
            self.vision_data.get(
                "data",
                [],
            )
        )

# ─────────────────────────────────────────────────────────────────────────────
# ImageProcessor
# ─────────────────────────────────────────────────────────────────────────────


class ImageProcessor:

    def __init__(
        self,