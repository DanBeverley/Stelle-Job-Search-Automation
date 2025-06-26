from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import schemas
from ..services import skill_analysis_service
from ..models.db.database import get_db
from .auth import get_current_active_user
from ..utils.api_helpers import get_user_with_cv_data, handle_service_error

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
    db_user = get_user_with_cv_data(db, current_user, require_cv=True)

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
    except Exception as e:
        raise handle_service_error(e, "Skill Analysis") 