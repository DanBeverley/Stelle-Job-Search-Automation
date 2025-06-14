from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from .. import schemas
from .auth import get_current_active_user
from ..services.ml_service import ml_service
from ..services.gemini_service import parse_cv_with_gemini
import pypdf
import docx
import io
import magic
from PIL import Image
import pytesseract

# Note: To use pytesseract, you need to have the Tesseract OCR engine installed on your system
# and configured in your system's PATH.

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

def extract_text_from_image(file_stream: io.BytesIO) -> str:
    """Extracts text from an image file stream using OCR."""
    try:
        image = Image.open(file_stream)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        # This could be a Pillow error or a Tesseract error
        raise HTTPException(status_code=400, detail=f"Error processing image file: {e}")


@router.post("/parse-cv", response_model=schemas.CVParsingResult)
async def parse_cv(
    file: UploadFile = File(...),
    current_user: schemas.User = Depends(get_current_active_user)
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

@router.post("/parse-cv/details", response_model=schemas.CVParsingDetailsResult)
async def parse_cv_details(
    file: UploadFile = File(...),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Uploads and parses a CV (PDF, DOCX, PNG, JPG) to extract structured details
    using an AI model.
    """
    content = await file.read()
    file_stream = io.BytesIO(content)
    text = ""

    # Use python-magic to detect the real file type
    file_type = magic.from_buffer(content, mime=True)
    
    if file_type == "application/pdf":
        text = extract_text_from_pdf(file_stream)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_stream)
    elif file_type in ["image/png", "image/jpeg", "image/tiff"]:
        text = extract_text_from_image(file_stream)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}. Please upload a PDF, DOCX, PNG, or JPG file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

    try:
        # Get structured data from Gemini
        parsed_data = parse_cv_with_gemini(text)
        
        # We need to add the filename to the dictionary before validating with Pydantic
        parsed_data['filename'] = file.filename

        # Validate and structure the data using our Pydantic model
        result = schemas.CVParsingDetailsResult(**parsed_data)
        
        return result

    except (ValueError, IOError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") 