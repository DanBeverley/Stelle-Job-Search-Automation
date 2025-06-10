from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from .. import schemas
from ..security import get_current_user
from ..services.ml_service import ml_service
import pypdf
import docx
import io

router = APIRouter()

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Extracts text from a PDF file stream."""
    try:
        reader = pypdf.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF file: {e}")

def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    """Extracts text from a DOCX file stream."""
    try:
        document = docx.Document(file_stream)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing DOCX file: {e}")


@router.post("/parse-cv", response_model=schemas.CVParsingResult)
async def parse_cv(
    file: UploadFile = File(...),
    current_user: schemas.User = Depends(get_current_user)
):
    """
    Uploads and parses a CV (PDF or DOCX) to predict its category.
    """
    content = await file.read()
    file_stream = io.BytesIO(content)
    text = ""
    
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file_stream)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_stream)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

    # Get prediction from the ML service
    predicted_category = ml_service.predict(text)
    
    return {
        "filename": file.filename,
        "predicted_category": predicted_category
    } 