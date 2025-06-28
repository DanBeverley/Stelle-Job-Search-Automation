from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import schemas
from services.ml_service import MLService
from models.db.database import get_db
from api.auth import get_current_active_user
from utils.api_helpers import handle_service_error

router = APIRouter()

@router.post("/generate", response_model=schemas.CoverLetterResponse, summary="Generate a Cover Letter")
def generate_cover_letter(
    request: schemas.CoverLetterRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Generates a cover letter for a specific job application using a fine-tuned
    generative model based on the user's stored CV data.
    """
    try:
        # Use our working ML service instead of the problematic cover_letter_service
        ml_service = MLService()
        cover_letter = ml_service.generate_cover_letter(
            job_title=request.job_title,
            company=request.company,
            additional_info=getattr(request, 'additional_info', '')
        )
        
        # Return in expected schema format
        return schemas.CoverLetterResponse(
            cover_letter_text=cover_letter,
            prompt_used=f"Generated for {request.job_title} at {request.company}"
        )
    except Exception as e:
        raise handle_service_error(e, "Cover Letter Generation") 