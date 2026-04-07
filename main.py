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


def choose_final_decision(rule_decision: dict, llm_decision: dict) -> dict:
    if llm_decision["status"] and llm_decision["reason"]:
        return {
            "status": llm_decision["status"],
            "reason": llm_decision["reason"],
            "confidence": llm_decision["confidence"],
            "source": "llm",
        }

    return {
        "status": rule_decision["status"],
        "reason": rule_decision["reason"],
        "confidence": None,
        "source": "rule_engine",
    }


def get_status_color(status: str) -> str:
    status_colors = {
        "Approved": "Green",
        "Flagged": "Yellow",
        "Rejected": "Red",
    }
    return status_colors.get(status, "Yellow")


def validate_receipt_data(
    receipt_details: dict,
    blur_score: Optional[float],
) -> list:
    validation_errors = []

    if blur_score is not None and blur_score < BLUR_THRESHOLD:
        validation_errors.append(
            f"Uploaded image is too blurry for reliable OCR (blur score: {blur_score:.2f})"
        )

    required_fields = {
        "merchant_name": "merchant",
        "total_amount": "amount",
        "date": "date",
    }
    for field_key, field_name in required_fields.items():
        if not receipt_details.get(field_key):
            validation_errors.append(f"Missing required receipt data: {field_name}")

    return validation_errors


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

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Check blur for uploaded images.
        if file_path.suffix.lower() == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
            policy_chunks = chunk_text(extracted_text, 300)
            POLICY_CHUNKS.clear()
            POLICY_CHUNKS.extend(policy_chunks)
        else:
            policy_chunks = None
            try:
                blur_score = get_blur_score(file_path)
            except RuntimeError:
                logger.warning("OpenCV not installed; skipping blur check for file=%s", safe_name)

            # 2. Extract OCR from the image.
            extracted_text = extract_text_from_image(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 3. Parse receipt data from the extracted text.
    text = extracted_text.strip() or "No text detected."
    receipt_details = extract_receipt_details(text)
    total_value = parse_amount(receipt_details["total_amount"])
    validation_errors = validate_receipt_data(receipt_details, blur_score)

    # 4. Retrieve relevant policy chunks.
    relevant_policy_chunks = retrieve_relevant_chunks(
        chunks=POLICY_CHUNKS,
        expense_category=justification,
        city=city,
        employee_role=employee_role,
    )

    # 5. Build the LLM prompt.
    policy_text = "\n\n".join(relevant_policy_chunks) if relevant_policy_chunks else "No relevant policy text found."
    auditor_prompt = build_auditor_prompt(
        receipt_data=receipt_details,
        justification=justification,
        policy_text=policy_text,
        city=city,
        role=employee_role,
    )

    # 6. Call llama3 and parse its JSON response.
    try:
        llm_response = query_llama3(auditor_prompt)
    except Exception as exc:
        logger.warning("Failed to query llama3 for file=%s: %s", safe_name, exc)
        llm_response = f"LLM call failed: {exc}"

    llm_decision = extract_llm_json(llm_response)
    rule_decision = evaluate_expense_rule(total_value)

    # 7. Return the final decision JSON.
    final_decision = choose_final_decision(rule_decision, llm_decision)
    if validation_errors:
        final_decision = {
            "status": "Rejected",
            "reason": "; ".join(validation_errors),
            "confidence": "1.0",
            "source": "validation",
        }
    logger.info(
        "Processed upload file=%s extracted_amount=%s",
        safe_name,
        receipt_details["total_amount"],
    )

    return {
        "status": final_decision["status"],
        "color": get_status_color(final_decision["status"]),
        "reason": final_decision["reason"],
        "confidence": final_decision["confidence"],
        "source": final_decision["source"],
        "receipt": {
            "filename": safe_name,
            "merchant": receipt_details["merchant_name"],
            "amount": receipt_details["total_amount"],
            "date": receipt_details["date"],
            "claimed_date": claimed_date,
            "blur_score": blur_score,
        },
        "employee": {
            "city": city,
            "role": employee_role,
            "justification": justification,
        },
        "policy": {
            "matched_chunks": relevant_policy_chunks,
            "matched_count": len(relevant_policy_chunks),
        },
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
        },
        "debug": {
            "path": str(file_path),
            "ocr_text": text,
            "policy_chunks_stored": len(POLICY_CHUNKS),
            "auditor_prompt": auditor_prompt,
            "llm_response": llm_response,
            "rule_decision": rule_decision,
        },
    }
