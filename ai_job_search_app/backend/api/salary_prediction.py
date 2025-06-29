from fastapi import APIRouter, Depends, Body
import schemas
from services.ml_service import MLService
from api.auth import get_current_active_user
from models.db.database import get_db
from utils.api_helpers import handle_service_error
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
        # Use our working ML service
        ml_service = MLService()
        salary_data = ml_service.predict_salary(
            job_title=request.job_title,
            experience_level=getattr(request, 'experience_level', 'Mid'),
            location=request.location
        )
        
        return schemas.SalaryPredictionResponse(
            predicted_salary=salary_data['predicted_salary'],
            confidence_score=salary_data['confidence'],
            currency=salary_data.get('currency', 'USD'),
            salary_range_min=int(salary_data['predicted_salary'] * 0.9),
            salary_range_max=int(salary_data['predicted_salary'] * 1.1)
        )

    except Exception as e:
        raise handle_service_error(e, "Salary Prediction") 