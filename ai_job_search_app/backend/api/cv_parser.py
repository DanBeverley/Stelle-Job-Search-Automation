from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from logging import getLogger
from sqlalchemy.orm import Session
from .. import schemas
from ..models.db.database import get_db
from .auth import get_current_active_user
from ..services.gemini_service import parse_cv_from_image, parse_cv_from_text
from ..utils.cv_utils import extract_text_from_pdf, extract_text_from_docx
import io

logger = getLogger(__name__)
router = APIRouter()

@router.post("/parse", response_model=schemas.CVParsingDetailsResult, summary="Parse a CV and save to user profile")
async def parse_cv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Uploads a CV file, parses it to extract structured data,
    and saves the result to the current user's profile.
    """
    contents = await file.read()
    content_type = file.content_type
    logger.info(f"Received file '{file.filename}' for user '{current_user.email}'")

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

        # Save the parsed data to the user's profile in the database
        try:
            current_user.parsed_cv_data = parsed_data
            db.commit()
            db.refresh(current_user)
            logger.info(f"Successfully saved parsed CV data for user '{current_user.email}'")
        except Exception as db_error:
            logger.error(f"Database error saving parsed CV data for user '{current_user.email}': {db_error}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save parsed CV data to user profile."
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