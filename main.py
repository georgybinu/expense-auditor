from pathlib import Path
import logging
import shutil
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import create_access_token, decode_access_token, hash_password, verify_password
from database import SessionLocal
from models import Claim, Company, Policy, User
from ocr import extract_text_from_image
from parser import extract_receipt_details
from pdf import extract_text_from_pdf
from prompt_utils import build_auditor_prompt
from quality import get_blur_score
from rag_utils import generate_embeddings, semantic_search_chunks
from rules import evaluate_expense_rule
from text_utils import chunk_text, retrieve_relevant_chunks
from llm import extract_llm_json, query_llama3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
POLICIES_DIR = Path("policies")
POLICIES_DIR.mkdir(exist_ok=True)
BLUR_THRESHOLD = 40.0
POLICY_CHUNKS = []
POLICY_CHUNK_EMBEDDINGS = []
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


class SignupRequest(BaseModel):
    email: str
    password: str
    role: str
    company_name: Optional[str] = None
    city: Optional[str] = None
    designation: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user


def require_auditor(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "auditor":
        raise HTTPException(status_code=403, detail="Auditor access required")
    return current_user


def require_employee(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "employee":
        raise HTTPException(status_code=403, detail="Employee access required")
    return current_user


def get_accessible_claim(
    claim_id: int,
    current_user: User,
    db: Session,
) -> Claim:
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    if claim is None:
        raise HTTPException(status_code=404, detail="Claim not found")

    if current_user.role == "auditor" and current_user.company_id == claim.company_id:
        return claim

    if current_user.role == "employee" and current_user.id == claim.user_id:
        return claim

    raise HTTPException(status_code=403, detail="Not authorized to access this claim")


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


def get_notification_message(status: Optional[str]) -> str:
    notifications = {
        "Approved": "Approved",
        "Flagged": "Needs review",
        "Rejected": "Rejected",
    }
    return notifications.get(status or "", "Needs review")


def get_risk_priority(status: Optional[str]) -> int:
    priorities = {
        "Rejected": 0,
        "Flagged": 1,
        "Approved": 2,
    }
    return priorities.get(status or "", 3)


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


@app.post("/signup")
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> dict:
    role = payload.role.strip().lower()
    if role not in {"auditor", "employee"}:
        raise HTTPException(status_code=400, detail="Role must be either 'auditor' or 'employee'")

    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email is already registered")

    company = None
    company_name = payload.company_name.strip() if payload.company_name else None

    if role == "auditor":
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required for auditors")

        existing_company = db.query(Company).filter(Company.name == company_name).first()
        if existing_company:
            raise HTTPException(status_code=400, detail="Company already exists")

        company = Company(name=company_name)
        db.add(company)
        db.flush()

    if role == "employee":
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required for employees")
        if not payload.city or not payload.city.strip():
            raise HTTPException(status_code=400, detail="city is required for employees")
        if not payload.designation or not payload.designation.strip():
            raise HTTPException(status_code=400, detail="designation is required for employees")

        company = db.query(Company).filter(Company.name == company_name).first()
        if company is None:
            raise HTTPException(status_code=404, detail="Company not found")

    user = User(
        email=payload.email,
        password=hash_password(payload.password),
        role=role,
        city=payload.city.strip() if payload.city else None,
        designation=payload.designation.strip() if payload.designation else None,
        company_id=company.id if company else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "User created successfully",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "city": user.city,
            "designation": user.designation,
            "company_id": user.company_id,
            "company_name": company.name if company else None,
        },
    }


@app.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role,
            "company_id": user.company_id,
        }
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "city": user.city,
            "designation": user.designation,
            "company_id": user.company_id,
        },
    }


@app.get("/companies")
def list_companies(db: Session = Depends(get_db)) -> dict:
    companies = db.query(Company).order_by(Company.name.asc()).all()
    return {
        "count": len(companies),
        "companies": [
            {
                "id": company.id,
                "name": company.name,
            }
            for company in companies
        ],
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
def get_policy_chunks(current_user: User = Depends(require_auditor)) -> dict:
    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
            "company_id": current_user.company_id,
        },
        "count": len(POLICY_CHUNKS),
        "embedding_count": len(POLICY_CHUNK_EMBEDDINGS),
        "chunks": POLICY_CHUNKS,
    }


@app.get("/policy-search")
def search_policy_chunks(
    expense_category: Optional[str] = None,
    city: Optional[str] = None,
    employee_role: Optional[str] = None,
    current_user: User = Depends(require_auditor),
) -> dict:
    query_parts = [expense_category, city, employee_role]
    semantic_query = " ".join(part.strip() for part in query_parts if part and part.strip())
    semantic_results = semantic_search_chunks(
        query=semantic_query,
        chunks=POLICY_CHUNKS,
        embeddings=POLICY_CHUNK_EMBEDDINGS,
    )

    if semantic_results:
        relevant_chunks = [chunk for chunk, _ in semantic_results]
    else:
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
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
        },
        "search_type": "semantic" if semantic_results else "keyword",
        "count": len(relevant_chunks),
        "matches": [
            {
                "chunk": chunk,
                "score": score,
            }
            for chunk, score in semantic_results
        ] if semantic_results else [],
        "chunks": relevant_chunks,
    }


@app.get("/claims")
def list_company_claims(
    current_user: User = Depends(require_auditor),
    db: Session = Depends(get_db),
) -> dict:
    if current_user.company_id is None:
        raise HTTPException(status_code=400, detail="Auditor is not linked to a company")

    claims = db.query(Claim).filter(Claim.company_id == current_user.company_id).all()
    claims.sort(
        key=lambda claim: (
            get_risk_priority(claim.status),
            -(claim.created_at.timestamp() if claim.created_at else 0),
        )
    )

    return {
        "company_id": current_user.company_id,
        "count": len(claims),
        "sort_order": ["Rejected", "Flagged", "Approved"],
        "claims": [
            {
                "id": claim.id,
                "user_id": claim.user_id,
                "company_id": claim.company_id,
                "amount": claim.amount,
                "status": claim.status,
                "color": get_status_color(claim.status),
                "message": get_notification_message(claim.status),
                "reason": claim.reason,
                "receipt_path": claim.receipt_path,
                "justification": claim.justification,
                "created_at": claim.created_at,
            }
            for claim in claims
        ],
    }


@app.get("/claims/{claim_id}")
def get_claim_details(
    claim_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    claim = get_accessible_claim(claim_id, current_user, db)

    return {
        "claim": {
            "id": claim.id,
            "user_id": claim.user_id,
            "company_id": claim.company_id,
            "receipt_path": claim.receipt_path,
            "extracted_text": claim.extracted_text,
            "justification": claim.justification,
            "ai_decision": claim.status,
            "color": get_status_color(claim.status),
            "message": get_notification_message(claim.status),
            "reason": claim.reason,
            "policy_snippet": claim.policy_snippet,
            "created_at": claim.created_at,
        }
    }


@app.get("/my-claims")
def list_my_claims(
    current_user: User = Depends(require_employee),
    db: Session = Depends(get_db),
) -> dict:
    claims = (
        db.query(Claim)
        .filter(Claim.user_id == current_user.id)
        .order_by(Claim.created_at.desc())
        .all()
    )

    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
        },
        "count": len(claims),
        "claims": [
            {
                "id": claim.id,
                "status": claim.status,
                "color": get_status_color(claim.status),
                "message": get_notification_message(claim.status),
                "reason": claim.reason,
                "justification": claim.justification,
                "created_at": claim.created_at,
            }
            for claim in claims
        ],
    }


@app.post("/policy-upload")
def upload_policy(
    file: UploadFile = File(...),
    current_user: User = Depends(require_auditor),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Policy upload must be a PDF")

    if current_user.company_id is None:
        raise HTTPException(status_code=400, detail="Auditor is not linked to a company")

    safe_name = Path(file.filename).name
    file_path = POLICIES_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_text_from_pdf(file_path)
        policy_chunks = chunk_text(extracted_text, 300)
        policy_embeddings = generate_embeddings(policy_chunks) if policy_chunks else []
        POLICY_CHUNKS.clear()
        POLICY_CHUNKS.extend(policy_chunks)
        POLICY_CHUNK_EMBEDDINGS.clear()
        POLICY_CHUNK_EMBEDDINGS.extend(policy_embeddings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    policy = Policy(
        company_id=current_user.company_id,
        file_path=str(file_path),
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)

    logger.info("Processed policy upload file=%s chunks=%s", safe_name, len(policy_chunks))

    return {
        "message": "Policy uploaded successfully",
        "filename": safe_name,
        "uploaded_by": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role,
            "company_id": current_user.company_id,
        },
        "policy": {
            "id": policy.id,
            "company_id": policy.company_id,
            "path": str(file_path),
            "chunk_count": len(policy_chunks),
            "embedding_count": len(POLICY_CHUNK_EMBEDDINGS),
        },
    }


@app.post("/claims")
def submit_claim(
    justification: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(require_employee),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if current_user.company_id is None:
        raise HTTPException(status_code=400, detail="Employee is not linked to a company")

    if Path(file.filename).suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Claim submission requires a receipt image, not a PDF")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_text_from_image(file_path).strip()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    receipt_details = extract_receipt_details(extracted_text)
    latest_policy = (
        db.query(Policy)
        .filter(Policy.company_id == current_user.company_id)
        .order_by(Policy.id.desc())
        .first()
    )
    if latest_policy is None:
        raise HTTPException(status_code=404, detail="No policy found for employee company")

    try:
        policy_text = extract_text_from_pdf(Path(latest_policy.file_path)).strip()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    relevant_policy_chunks = retrieve_relevant_chunks(
        chunks=chunk_text(policy_text, 300),
        expense_category=justification,
        city=current_user.city,
        employee_role=current_user.designation or current_user.role,
    )
    policy_snippet = "\n\n".join(relevant_policy_chunks) if relevant_policy_chunks else policy_text

    auditor_prompt = build_auditor_prompt(
        receipt_data=receipt_details,
        justification=justification,
        policy_text=policy_snippet,
        city=current_user.city or "Unknown",
        role=current_user.designation or current_user.role,
    )

    try:
        llm_response = query_llama3(auditor_prompt)
    except Exception as exc:
        logger.warning("Failed to query llama3 for claim file=%s: %s", safe_name, exc)
        llm_response = f"LLM call failed: {exc}"

    llm_decision = extract_llm_json(llm_response)
    claim_status = llm_decision["status"] if llm_decision["status"] in {"Approved", "Flagged", "Rejected"} else "Flagged"
    claim_reason = llm_decision["reason"] or "Unable to determine decision from AI response"
    total_value = parse_amount(receipt_details.get("total_amount"))

    claim = Claim(
        user_id=current_user.id,
        company_id=current_user.company_id,
        amount=int(total_value) if total_value is not None else None,
        status=claim_status,
        reason=claim_reason,
        receipt_path=str(file_path),
        justification=justification,
        extracted_text=extracted_text,
        policy_snippet=policy_snippet,
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)

    return {
        "message": "Claim submitted successfully",
        "claim": {
            "id": claim.id,
            "user_id": claim.user_id,
            "company_id": claim.company_id,
            "amount": claim.amount,
            "status": claim.status,
            "color": get_status_color(claim.status),
            "message": get_notification_message(claim.status),
            "reason": claim.reason,
            "justification": claim.justification,
            "receipt_path": claim.receipt_path,
            "extracted_text": extracted_text,
            "policy_path": latest_policy.file_path,
            "policy_snippet": policy_snippet,
            "llm_decision": llm_decision,
            "created_at": claim.created_at,
        },
    }


@app.post("/upload")
def upload_file(
    justification: str = Form(...),
    city: str = Form(...),
    employee_role: str = Form(...),
    claimed_date: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(require_employee),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    if Path(file.filename).suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Claim submission does not accept PDF policy uploads")

    safe_name = Path(file.filename).name
    file_path = UPLOADS_DIR / safe_name
    blur_score = None

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Check blur for uploaded images.
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
            "id": current_user.id,
            "email": current_user.email,
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
