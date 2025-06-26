from fastapi import APIRouter, Depends, Body
from .. import schemas
from ..services import salary_prediction_service
from .auth import get_current_active_user
from ..models.db.database import get_db
from ..utils.api_helpers import get_user_with_cv_data, handle_service_error
from sqlalchemy.orm import Session

router = APIRouter()

@router.post(
    "/predict",
    response_model=schemas.SalaryPredictionResponse,
    summary="Predict salary using our in-house model"
)
def predict_salary(
    request: schemas.SalaryPredictionRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Predicts a salary range using our custom-trained XGBoost model.
    The prediction is based on the job title, location, and a description
    built from the user's skills and experience combined with the job's requirements.
    """
    db_user = get_user_with_cv_data(db, current_user, require_cv=True)

    # Combine user's skills and experiences with job info for a richer description
    user_skills = ", ".join(db_user.parsed_cv_data.get('skills', []))
    user_summary = db_user.parsed_cv_data.get('summary', '')
    
    # The training script used 'Job.Description'. We simulate that here.
    combined_description = f"{user_summary}. Required skills: {', '.join(request.skills)}. Candidate skills: {user_skills}."

    try:
        salary_data = salary_prediction_service.predict_salary_with_xgb(
            job_title=request.job_title,
            location=request.location,
            job_description=combined_description
        )
        return schemas.SalaryPredictionResponse(**salary_data)

    except Exception as e:
        raise handle_service_error(e, "Salary Prediction") 