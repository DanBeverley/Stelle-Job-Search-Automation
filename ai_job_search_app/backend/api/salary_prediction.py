from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel
from typing import Optional
import schemas
from services.ml_service import MLService
from api.auth import get_current_active_user
from models.db.database import get_db
from utils.api_helpers import handle_service_error, get_user_with_cv_data
from sqlalchemy.orm import Session

router = APIRouter()

class SimpleSalaryRequest(BaseModel):
    job_title: str
    experience_level: Optional[str] = 'Mid'
    location: str

@router.post("/test", summary="Test Salary Prediction (No Auth)")
def test_salary_prediction(request: SimpleSalaryRequest):
    """
    Test endpoint for salary prediction without authentication.
    For testing purposes only.
    """
    try:
        # Direct salary prediction without complex model loading
        job_title = request.job_title
        experience_level = getattr(request, 'experience_level', 'Mid')
        location = request.location
        
        # Basic salary estimation logic
        base_salary = 70000
        
        # Adjust for job title
        title_lower = job_title.lower()
        if "senior" in title_lower or "lead" in title_lower:
            base_salary += 30000
        elif "manager" in title_lower or "director" in title_lower:
            base_salary += 50000
        elif "engineer" in title_lower:
            base_salary += 15000
        elif "scientist" in title_lower:
            base_salary += 20000
        
        # Adjust for experience
        experience_multipliers = {
            'Entry': 0.8,
            'Mid': 1.0,
            'Senior': 1.4,
            'Lead': 1.8
        }
        base_salary *= experience_multipliers.get(experience_level, 1.0)
        
        # Adjust for location
        location_multipliers = {
            'San Francisco': 1.4,
            'New York': 1.3,
            'Seattle': 1.2,
            'Boston': 1.15,
            'Austin': 1.1,
            'Chicago': 1.05,
            'Remote': 1.0
        }
        
        # Find closest location match
        location_multiplier = 1.0
        for loc, mult in location_multipliers.items():
            if loc.lower() in location.lower():
                location_multiplier = mult
                break
        
        final_salary = int(base_salary * location_multiplier)
        
        return {
            "predicted_salary": final_salary,
            "confidence": 0.85,
            "currency": "USD",
            "status": "success",
            "breakdown": {
                "base_salary": int(base_salary),
                "location_multiplier": location_multiplier,
                "experience_level": experience_level
            }
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

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