"""
src/ingestion/image_processor.py

Production-Grade Financial Image Processor

Optimized for:
- FinanceBench
- SEC filings
- OCR
- Embedded charts
- Tables
- Screenshots
- PDF images
- Windows
- Colab
- T4 GPUs
- Ollama multimodal
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
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OCR_DPI = 200

MAX_IMAGES_PER_DOC = 250

MAX_IMAGE_SIZE_MB = 15

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
}

VISION_PROMPT = """
You are analyzing a financial document image.

Extract:
1. ALL visible text
2. ALL visible numbers
3. Tables/charts if present
4. Labels and values

Return valid JSON only.

{
  "type": "chart|table|screenshot|document|other",
  "title": "",
  "text": "",
  "data": [
    {
      "label": "",
      "value": "",
      "unit": ""
    }
  ]
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# OCR Worker
# ─────────────────────────────────────────────────────────────────────────────


def _ocr_worker(
    image_path: str,
) -> str:

    try:

        import pytesseract

        from PIL import Image

        image = Image.open(
            image_path
        )

        text = (
            pytesseract.image_to_string(
                image
            )
        )

        return text.strip()

    except Exception:

        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Extracted Image
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
        vision_data: Optional[
            Dict
        ] = None,
        temp_image_path: str = "",
    ):

        self.page_number = (
            page_number
        )

        self.image_index = (
            image_index
        )

        self.width = width

        self.height = height

        self.format = format

        self.source = source

        self.ocr_text = ocr_text

        self.vision_data = (
            vision_data or {}
        )

        self.temp_image_path = (
            temp_image_path
        )

    @property
    def has_data(
        self,
    ) -> bool:

        return bool(
            self.ocr_text.strip()
        ) or bool(
            self.vision_data
        )

# ─────────────────────────────────────────────────────────────────────────────
# Image Processor
# ─────────────────────────────────────────────────────────────────────────────


class ImageProcessor:

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_vision: bool = False,
        llm_client=None,
        max_workers: int = 2,
    ):

        self.enable_ocr = (
            enable_ocr
        )

        self.enable_vision = (
            enable_vision
        )

        self._llm = llm_client

        self.max_workers = max(
            1,
            int(max_workers),
        )

    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        state,
    ):

        document_path = getattr(
            state,
            "document_path",
            "",
        ) or ""

        if not document_path:

            logger.warning(
                "[IMG] Missing document path"
            )

            return state

        extracted = (
            self.process_document(
                document_path
            )
        )

        if not extracted:

            return state

        image_texts = []

        table_cells = list(
            getattr(
                state,
                "table_cells",
                [],
            )
            or []
        )

        for item in extracted:

            if item.ocr_text:

                image_texts.append(
                    item.ocr_text
                )

            if item.vision_data:

                text = (
                    item.vision_data.get(
                        "text",
                        "",
                    )
                )

                if text:

                    image_texts.append(
                        text
                    )

                for idx, row in enumerate(
                    item.vision_data.get(
                        "data",
                        [],
                    )
                ):

                    table_cells.append(
                        {
                            "row_header": row.get(
                                "label",
                                "",
                            ),
                            "col_header": row.get(
                                "unit",
                                "",
                            ),
                            "value": row.get(
                                "value",
                                "",
                            ),
                            "page": item.page_number,
                            "table_number": idx,
                            "section": "IMAGE",
                            "company": getattr(
                                state,
                                "company_name",
                                "",
                            ),
                            "doc_type": getattr(
                                state,
                                "doc_type",
                                "",
                            ),
                            "fiscal_year": getattr(
                                state,
                                "fiscal_year",
                                "",
                            ),
                        }
                    )

        if image_texts:

            existing = getattr(
                state,
                "raw_text",
                "",
            ) or ""

            merged = (
                existing
                + "\n\n"
                + "\n\n".join(
                    image_texts
                )
            )

            state.raw_text = (
                merged
            )

        state.table_cells = (
            table_cells
        )

        return state

    # ─────────────────────────────────────────────────────────────────────

    def process_document(
        self,
        document_path: str,
    ) -> List[ExtractedImage]:

        ext = Path(
            document_path
        ).suffix.lower()

        if (
            ext
            in SUPPORTED_IMAGE_EXTENSIONS
        ):

            return self._process_single_image(
                document_path
            )

        if ext == ".pdf":

            return self._process_pdf(
                document_path
            )

        return []

    # ─────────────────────────────────────────────────────────────────────
    # PDF Processing
    # ─────────────────────────────────────────────────────────────────────

    def _process_pdf(
        self,
        pdf_path: str,
    ) -> List[ExtractedImage]:

        images = []

        try:

            import fitz

            from PIL import Image

            doc = fitz.open(
                pdf_path
            )

            image_count = 0

            for page_idx, page in enumerate(
                doc,
                start=1,
            ):

                if (
                    image_count
                    >= MAX_IMAGES_PER_DOC
                ):

                    break

                # Embedded images

                embedded = (
                    page.get_images(
                        full=True
                    )
                )

                for img_idx, img in enumerate(
                    embedded,
                    start=1,
                ):

                    if (
                        image_count
                        >= MAX_IMAGES_PER_DOC
                    ):

                        break

                    try:

                        xref = img[0]

                        base = (
                            doc.extract_image(
                                xref
                            )
                        )

                        image_bytes = (
                            base[
                                "image"
                            ]
                        )

                        image_ext = (
                            base.get(
                                "ext",
                                "png",
                            )
                        )

                        with tempfile.NamedTemporaryFile(
                            suffix=f".{image_ext}",
                            delete=False,
                        ) as tmp:

                            tmp.write(
                                image_bytes
                            )

                            temp_path = (
                                tmp.name
                            )

                        pil_image = (
                            Image.open(
                                io.BytesIO(
                                    image_bytes
                                )
                            )
                        )

                        extracted = (
                            ExtractedImage(
                                page_number=page_idx,
                                image_index=img_idx,
                                width=pil_image.width,
                                height=pil_image.height,
                                format=image_ext,
                                source="pdf_embedded",
                                temp_image_path=temp_path,
                            )
                        )

                        images.append(
                            extracted
                        )

                        image_count += 1

                    except Exception:

                        logger.exception(
                            "[IMG] Embedded extraction failed"
                        )

                # Full page render OCR fallback

                if (
                    image_count
                    >= MAX_IMAGES_PER_DOC
                ):

                    break

                try:

                    pix = (
                        page.get_pixmap(
                            dpi=OCR_DPI
                        )
                    )

                    image_bytes = (
                        pix.tobytes(
                            "png"
                        )
                    )

                    with tempfile.NamedTemporaryFile(
                        suffix=".png",
                        delete=False,
                    ) as tmp:

                        tmp.write(
                            image_bytes
                        )

                        temp_path = (
                            tmp.name
                        )

                    images.append(
                        ExtractedImage(
                            page_number=page_idx,
                            image_index=image_count,
                            width=pix.width,
                            height=pix.height,
                            format="png",
                            source="page_render",
                            temp_image_path=temp_path,
                        )
                    )

                    image_count += 1

                except Exception:

                    logger.exception(
                        "[IMG] Page render failed"
                    )

            doc.close()

        except Exception:

            logger.exception(
                "[IMG] PDF processing failed"
            )

        self._run_ocr(
            images
        )

        self._run_vision(
            images
        )

        return [
            img
            for img in images
            if img.has_data
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Single Image
    # ─────────────────────────────────────────────────────────────────────

    def _process_single_image(
        self,
        image_path: str,
    ) -> List[ExtractedImage]:

        try:

            from PIL import Image

            image = Image.open(
                image_path
            )

            extracted = (
                ExtractedImage(
                    page_number=1,
                    image_index=1,
                    width=image.width,
                    height=image.height,
                    format=Path(
                        image_path
                    ).suffix.replace(
                        ".",
                        "",
                    ),
                    source="direct_image",
                    temp_image_path=image_path,
                )
            )

            images = [extracted]

            self._run_ocr(
                images
            )

            self._run_vision(
                images
            )

            return images

        except Exception:

            logger.exception(
                "[IMG] Image load failed"
            )

            return []

    # ─────────────────────────────────────────────────────────────────────
    # OCR
    # ─────────────────────────────────────────────────────────────────────

    def _run_ocr(
        self,
        images: List[
            ExtractedImage
        ],
    ):

        if not self.enable_ocr:

            return

        valid_paths = [
            img.temp_image_path
            for img in images
            if img.temp_image_path
        ]

        if not valid_paths:
            return

        try:

            with ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:

                results = list(
                    executor.map(
                        _ocr_worker,
                        valid_paths,
                    )
                )

            for img, text in zip(
                images,
                results,
            ):

                img.ocr_text = (
                    text or ""
                )

        except Exception:

            logger.exception(
                "[IMG] OCR pipeline failed"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Vision
    # ─────────────────────────────────────────────────────────────────────

    def _run_vision(
        self,
        images: List[
            ExtractedImage
        ],
    ):

        if (
            not self.enable_vision
            or self._llm is None
        ):

            return

        for img in images:

            try:

                if not os.path.exists(
                    img.temp_image_path
                ):

                    continue

                size_mb = (
                    os.path.getsize(
                        img.temp_image_path
                    )
                    / (
                        1024 * 1024
                    )
                )

                if (
                    size_mb
                    > MAX_IMAGE_SIZE_MB
                ):

                    continue

                with open(
                    img.temp_image_path,
                    "rb",
                ) as f:

                    image_bytes = (
                        f.read()
                    )

                encoded = (
                    base64.b64encode(
                        image_bytes
                    ).decode(
                        "utf-8"
                    )
                )

                payload = (
                    f"{VISION_PROMPT}\n\n"
                    f"IMAGE_BASE64:\n{encoded[:2000]}"
                )

                response = (
                    self._llm.chat(
                        prompt=payload,
                        temperature=0.1,
                        max_tokens=1200,
                    )
                )

                parsed = (
                    self._safe_json(
                        response
                    )
                )

                if parsed:

                    img.vision_data = (
                        parsed
                    )

            except Exception:

                logger.exception(
                    "[IMG] Vision failed"
                )

    # ─────────────────────────────────────────────────────────────────────
    # JSON Parsing
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_json(
        text: str,
    ) -> Dict:

        if not text:
            return {}

        cleaned = text.strip()

        if cleaned.startswith(
            "```"
        ):

            cleaned = re.sub(
                r"^```(?:json)?",
                "",
                cleaned,
                flags=re.I,
            )

            cleaned = re.sub(
                r"```$",
                "",
                cleaned,
            )

        try:

            return json.loads(
                cleaned
            )

        except Exception:
            pass

        try:

            start = cleaned.find(
                "{"
            )

            end = cleaned.rfind(
                "}"
            )

            if (
                start >= 0
                and end > start
            ):

                return json.loads(
                    cleaned[
                        start:end + 1
                    ]
                )

        except Exception:
            pass

        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_image_processor(
    state,
    enable_ocr: bool = True,
    enable_vision: bool = False,
    llm_client=None,
):

    processor = ImageProcessor(
        enable_ocr=enable_ocr,
        enable_vision=enable_vision,
        llm_client=llm_client,
    )

    return processor.run(
        state
    )