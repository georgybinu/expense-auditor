from pathlib import Path
import logging
import shutil
from typing import Optional

from fastapi import HTTPException
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from ocr import extract_text_from_image
from parser import extract_receipt_details
from pdf import extract_text_from_pdf
from quality import get_blur_score
from rules import evaluate_expense_rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
BLUR_THRESHOLD = 40.0


def parse_amount(amount_text: Optional[str]) -> Optional[float]:
    if not amount_text:
        return None

    cleaned_amount = (
        amount_text.replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .replace(",", "")
        .strip()
    )

    try:
        return float(cleaned_amount)
    except ValueError:
        return None


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
                <input name="justification" type="text" placeholder="Enter justification" />
                <input name="file" type="file" />
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload")
def upload_file(
    justification: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name
    blur_score = None

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file_path.suffix.lower() == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            try:
                blur_score = get_blur_score(file_path)
            except RuntimeError:
                logger.warning("OpenCV not installed; skipping blur check for file=%s", safe_name)
            else:
                if blur_score < BLUR_THRESHOLD:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded image is too blurry for OCR (blur score: {blur_score:.2f})",
                    )
            extracted_text = extract_text_from_image(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = extracted_text.strip() or "No text detected."
    receipt_details = extract_receipt_details(text)
    total_value = parse_amount(receipt_details["total_amount"])
    decision = evaluate_expense_rule(total_value)
    logger.info(
        "Processed upload file=%s extracted_amount=%s",
        safe_name,
        receipt_details["total_amount"],
    )

    return {
        "filename": safe_name,
        "justification": justification,
        "extracted_data": {
            "path": str(file_path),
            "blur_score": blur_score,
            "ocr_text": text,
            "merchant": receipt_details["merchant_name"],
            "date": receipt_details["date"],
            "amount": receipt_details["total_amount"],
        },
        "decision": {
            "status": decision["status"],
            "explanation": decision["reason"],
        },
    }
