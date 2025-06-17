from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas
from ..services import cover_letter_service
from ..models.db.database import get_db
from .auth import get_current_active_user

router = APIRouter()

@router.post("/generate", response_model=schemas.CoverLetterResponse, summary="Generate a Cover Letter")
def generate_cover_letter(
    request: schemas.CoverLetterRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Generates a cover letter for a specific job application using the user's
    stored CV data. This is currently a placeholder and will be connected
    to a fine-tuned model in the future.
    """
    if not current_user.parsed_cv_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No CV data found for this user. Please upload and parse a CV first."
        )

    response = cover_letter_service.generate_cover_letter_placeholder(
        user=current_user, request=request
    )
    return response 