from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import schemas
from services import cover_letter_service
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
        response = cover_letter_service.generate_cover_letter_with_finetuned_model(
            user=current_user, request=request
        )
        return response
    except Exception as e:
        raise handle_service_error(e, "Cover Letter Generation") 