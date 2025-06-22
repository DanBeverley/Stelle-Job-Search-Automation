from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas
from ..services import skill_analysis_service
from ..models.db.database import get_db
from ..models.db import user as user_model
from .auth import get_current_active_user
import requests

router = APIRouter()

@router.post(
    "/skill-analysis",
    response_model=schemas.SkillAnalysisResponse,
    summary="Analyze Skill Gap and Recommend Courses"
)
def analyze_skills(
    request: schemas.SkillAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Analyzes the gap between a user's skills (from their CV) and the
    skills required by a job description. Returns matched skills, missing skills,
    and recommended edX courses to fill the gaps.
    """
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    if not db_user or not db_user.parsed_cv_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CV data not found for the user. Please upload and parse a CV first."
        )

    user_skills = db_user.parsed_cv_data.get('skills', [])
    if not user_skills:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No skills found in the user's parsed CV."
        )

    try:
        analysis_result = skill_analysis_service.analyze_skill_gap(
            job_description=request.job_description,
            user_skills=user_skills
        )
        return schemas.SkillAnalysisResponse(**analysis_result)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not connect to the course recommendation service: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during skill analysis: {e}"
        ) 