from pathlib import Path
from html import escape
import shutil

from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import pytesseract

app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


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

    return f"""
    <html>
        <head>
            <title>OCR Result</title>
        </head>
        <body>
            <h1>OCR Result</h1>
            <p><strong>File:</strong> {escape(safe_name)}</p>
            <p><strong>Saved to:</strong> {escape(str(file_path))}</p>
            <h2>Extracted Text</h2>
            <pre>{escape(text)}</pre>
            <p><a href="/">Upload another image</a></p>
        </body>
    </html>
    """
