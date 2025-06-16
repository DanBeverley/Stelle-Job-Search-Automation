from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from logging import getLogger
from .. import schemas
from .auth import get_current_active_user
from ..services.gemini_service import parse_cv_from_image, parse_cv_from_text
from ..utils.cv_utils import extract_text_from_pdf, extract_text_from_docx
import io

logger = getLogger(__name__)
router = APIRouter()

@router.post("/parse", response_model=schemas.CVParsingDetailsResult, summary="Parse a CV to extract structured data")
async def parse_cv(
    file: UploadFile = File(...),
    # current_user: schemas.User = Depends(get_current_active_user) # Temporarily disabled for easier testing
):
    """
    Uploads and parses a CV file (PDF, DOCX, PNG, JPG) and returns structured data.
    """
    contents = await file.read()
    content_type = file.content_type
    logger.info(f"Received file '{file.filename}' with content type '{content_type}'")

    try:
        if content_type.startswith("image/"):
            parsed_data = parse_cv_from_image(contents)
        elif content_type == "application/pdf":
            cv_text = extract_text_from_pdf(contents)
            parsed_data = parse_cv_from_text(cv_text)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            cv_text = extract_text_from_docx(io.BytesIO(contents))
            parsed_data = parse_cv_from_text(cv_text)
        elif content_type == "text/plain":
            cv_text = contents.decode('utf-8')
            parsed_data = parse_cv_from_text(cv_text)
        else:
            logger.warning(f"Unsupported file type uploaded: {content_type}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {content_type}. Please upload a PDF, DOCX, PNG, or JPG file."
            )

        if not parsed_data:
            logger.error("CV parsing returned no data.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse CV: Model returned no data."
            )

        # Add filename to the response
        parsed_data['filename'] = file.filename
        return parsed_data

    except ValueError as e:
        logger.error(f"ValueError during CV parsing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during CV parsing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during CV parsing."
        ) 