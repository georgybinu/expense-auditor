from pathlib import Path

import fitz


def extract_text_from_pdf(file_path: Path) -> str:
    text_parts = []

    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
    except RuntimeError as exc:
        raise ValueError("Uploaded file is not a valid PDF") from exc

    return "\n".join(text_parts)
