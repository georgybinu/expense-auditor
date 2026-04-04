from pathlib import Path
from html import escape
import re
import shutil
from typing import Dict, Optional

from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import pytesseract

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


def extract_receipt_details(ocr_text: str) -> Dict[str, Optional[str]]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    amount_pattern = r"(?:[$€£]\s?)?\d{1,3}(?:,\d{3})*(?:\.\d{2})"
    merchant_ignore_words = (
        "receipt",
        "invoice",
        "tax",
        "cashier",
        "server",
        "table",
        "order",
        "transaction",
        "approval",
        "auth",
        "card",
        "visa",
        "mastercard",
        "subtotal",
        "total",
        "amount",
        "change",
        "balance",
        "date",
        "time",
        "www",
        "http",
    )

    merchant_name = None
    for line in lines[:8]:
        normalized_line = line.lower()
        if re.search(amount_pattern, line):
            continue
        if any(word in normalized_line for word in merchant_ignore_words):
            continue
        if re.search(r"\d{3,}", line):
            continue
        if len(line) < 3:
            continue
        merchant_name = line
        break

    if merchant_name is None and lines:
        merchant_name = lines[0]

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
    ]
    date_match = None
    for pattern in date_patterns:
        date_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if date_match:
            break

    total_amount = None
    total_keywords = [
        "grand total",
        "amount due",
        "net total",
        "total due",
        "total",
    ]
    excluded_total_words = ("subtotal", "tax", "discount", "change", "balance", "tip")

    for line in lines:
        normalized_line = line.lower()
        if any(word in normalized_line for word in excluded_total_words):
            continue
        if any(keyword in normalized_line for keyword in total_keywords):
            amounts = re.findall(amount_pattern, line)
            if amounts:
                total_amount = amounts[-1].replace(" ", "")
                break

    if total_amount is None:
        amount_matches = []
        for line in lines:
            normalized_line = line.lower()
            if any(word in normalized_line for word in ("subtotal", "tax", "discount", "change")):
                continue
            amount_matches.extend(re.findall(amount_pattern, line))
        if amount_matches:
            total_amount = amount_matches[-1].replace(" ", "")

    return {
        "merchant_name": merchant_name,
        "date": date_match.group(0) if date_match else None,
        "total_amount": total_amount,
    }


@app.get("/", response_class=HTMLResponse)
def read_root() -> str:
    return """
    <html>
        <head>
            <title>Upload File</title>
        </head>
        <body>
            <h1>Upload File</h1>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input name="file" type="file" />
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload", response_class=HTMLResponse)
def upload_file(file: UploadFile = File(...)) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        with Image.open(file_path) as image:
            extracted_text = pytesseract.image_to_string(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Tesseract OCR is not installed on this system",
        ) from exc

    text = extracted_text.strip() or "No text detected."
    receipt_details = extract_receipt_details(text)

    return f"""
    <html>
        <head>
            <title>OCR Result</title>
        </head>
        <body>
            <h1>OCR Result</h1>
            <p><strong>File:</strong> {escape(safe_name)}</p>
            <p><strong>Saved to:</strong> {escape(str(file_path))}</p>
            <h2>Parsed Details</h2>
            <p><strong>Merchant:</strong> {escape(receipt_details["merchant_name"] or "Not found")}</p>
            <p><strong>Date:</strong> {escape(receipt_details["date"] or "Not found")}</p>
            <p><strong>Total:</strong> {escape(receipt_details["total_amount"] or "Not found")}</p>
            <h2>Extracted Text</h2>
            <pre>{escape(text)}</pre>
            <p><a href="/">Upload another image</a></p>
        </body>
    </html>
    """
