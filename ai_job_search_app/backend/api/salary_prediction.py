from fastapi import APIRouter, Depends, HTTPException, Body, status
from ai_job_search_app.backend import schemas
from ..services import gemini_service
from .auth import get_current_active_user

router = APIRouter()

@router.post(
    "/predict-gemini",
    response_model=schemas.SalaryPredictionResponse,
    summary="Predict salary using Gemini based on detailed CV data"
)
async def predict_salary_advanced(
    request: schemas.SalaryPredictionRequest = Body(...),
    # current_user: schemas.User = Depends(get_current_active_user) # Temporarily disabled for easier testing
):
    """
    Predicts a salary range for a job title and location, personalized with
    the candidate's skills, experience, and education.
    """
    try:
        salary_data = gemini_service.get_salary_estimation_with_gemini(
            job_title=request.job_title,
            location=request.location,
            skills=request.skills,
            experience=request.experience,
            education=request.education
        )

        if not salary_data or "error" in salary_data:
            error_message = salary_data.get("error", "An unknown error occurred during prediction.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_message
            )
        
        return schemas.SalaryPredictionResponse(**salary_data)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        ) 