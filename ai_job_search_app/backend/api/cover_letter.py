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
    Generates a cover letter for a specific job application using a fine-tuned
    generative model based on the user's stored CV data.
    """
    try:
        response = cover_letter_service.generate_cover_letter_with_finetuned_model(
            user=current_user, request=request
        )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        # Generic error handler for any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        ) 