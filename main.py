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
from prompt_utils import build_auditor_prompt
from quality import get_blur_score
from rules import evaluate_expense_rule
from text_utils import chunk_text, retrieve_relevant_chunks
from llm import extract_llm_json, query_llama3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
BLUR_THRESHOLD = 40.0
POLICY_CHUNKS = []


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
                <input name="city" type="text" placeholder="Enter city" />
                <input name="employee_role" type="text" placeholder="Enter employee role" />
                <input name="claimed_date" type="text" placeholder="Enter claimed date" />
                <input name="file" type="file" />
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    """


@app.get("/policy-chunks")
def get_policy_chunks() -> dict:
    return {
        "count": len(POLICY_CHUNKS),
        "chunks": POLICY_CHUNKS,
    }


@app.get("/policy-search")
def search_policy_chunks(
    expense_category: Optional[str] = None,
    city: Optional[str] = None,
    employee_role: Optional[str] = None,
) -> dict:
    relevant_chunks = retrieve_relevant_chunks(
        chunks=POLICY_CHUNKS,
        expense_category=expense_category,
        city=city,
        employee_role=employee_role,
    )
    return {
        "query": {
            "expense_category": expense_category,
            "city": city,
            "employee_role": employee_role,
        },
        "count": len(relevant_chunks),
        "chunks": relevant_chunks,
    }


@app.post("/upload")
def upload_file(
    justification: str = Form(...),
    city: str = Form(...),
    employee_role: str = Form(...),
    claimed_date: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name
    blur_score = None
    text_chunks = None

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file_path.suffix.lower() == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
            text_chunks = chunk_text(extracted_text, 300)
            POLICY_CHUNKS.clear()
            POLICY_CHUNKS.extend(text_chunks)
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
    relevant_policy_chunks = retrieve_relevant_chunks(
        chunks=POLICY_CHUNKS,
        expense_category=justification,
        city=city,
        employee_role=employee_role,
    )
    policy_text = "\n\n".join(relevant_policy_chunks) if relevant_policy_chunks else "No relevant policy text found."
    auditor_prompt = build_auditor_prompt(
        receipt_data=receipt_details,
        justification=justification,
        policy_text=policy_text,
        city=city,
        role=employee_role,
    )
    try:
        llm_response = query_llama3(auditor_prompt)
    except Exception as exc:
        logger.warning("Failed to query llama3 for file=%s: %s", safe_name, exc)
        llm_response = f"LLM call failed: {exc}"
    llm_decision = extract_llm_json(llm_response)
    final_decision = {
        "status": decision["status"],
        "explanation": decision["reason"],
        "confidence": None,
        "source": "rule_engine",
    }
    if llm_decision["status"] and llm_decision["reason"]:
        final_decision = {
            "status": llm_decision["status"],
            "explanation": llm_decision["reason"],
            "confidence": llm_decision["confidence"],
            "source": "llm",
        }
    logger.info(
        "Processed upload file=%s extracted_amount=%s",
        safe_name,
        receipt_details["total_amount"],
    )

    return {
        "filename": safe_name,
        "submitted_data": {
            "justification": justification,
            "city": city,
            "employee_role": employee_role,
            "claimed_date": claimed_date,
        },
        "extracted_data": {
            "path": str(file_path),
            "blur_score": blur_score,
            "ocr_text": text,
            "text_chunks": text_chunks,
            "merchant": receipt_details["merchant_name"],
            "date": receipt_details["date"],
            "amount": receipt_details["total_amount"],
        },
        "decision": final_decision,
        "rule_decision": {
            "status": decision["status"],
            "explanation": decision["reason"],
        },
        "policy_match": {
            "count": len(relevant_policy_chunks),
            "chunks": relevant_policy_chunks,
        },
        "auditor_prompt": auditor_prompt,
        "llm_response": llm_response,
        "llm_decision": llm_decision,
    }
