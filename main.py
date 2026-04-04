from pathlib import Path
import shutil
from typing import Optional

from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

from ocr import extract_text_from_image
from parser import extract_receipt_details
from pdf import extract_text_from_pdf
from rules import evaluate_expense_rule

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


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
                <input name="file" type="file" />
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """


@app.post("/upload")
def upload_file(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file_path.suffix.lower() == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            extracted_text = extract_text_from_image(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = extracted_text.strip() or "No text detected."
    receipt_details = extract_receipt_details(text)
    total_value = parse_amount(receipt_details["total_amount"])
    decision = evaluate_expense_rule(total_value)

    return {
        "filename": safe_name,
        "path": str(file_path),
        "ocr_text": text,
        "merchant": receipt_details["merchant_name"],
        "date": receipt_details["date"],
        "amount": receipt_details["total_amount"],
        "status": decision["status"],
        "reason": decision["reason"],
    }
