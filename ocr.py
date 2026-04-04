from pathlib import Path

from PIL import Image, UnidentifiedImageError
import pytesseract


def extract_text_from_image(file_path: Path) -> str:
    try:
        with Image.open(file_path) as image:
            return pytesseract.image_to_string(image)
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is not installed on this system") from exc
