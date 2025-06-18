from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..api.auth import get_current_active_user
from .. import schemas
from ..models.db.database import get_db
from ..models.db import user as user_model
from ..services import interview_prep_service

router = APIRouter()

@router.post("/interview-prep/generate-questions", response_model=schemas.InterviewQuestions)
def get_interview_questions(
    request: schemas.InterviewPrepRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Takes a job description and the user's CV data from the database,
    and returns a list of tailored mock interview questions.
    """
    if not request.job_description:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    if not db_user or not db_user.parsed_cv_data:
        raise HTTPException(status_code=404, detail="CV data not found for the user. Please upload a CV first.")

    questions = interview_prep_service.generate_questions_from_cv(
        cv_data=db_user.parsed_cv_data, 
        job_description=request.job_description
    )
    
    return schemas.InterviewQuestions(questions=questions)

class AnswerRequest(schemas.BaseModel):
    answer: str

@router.post("/interview-prep/analyze-answer", response_model=schemas.STAResponse)
def analyze_user_answer(
    request: AnswerRequest,
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Analyzes a user's answer to an interview question using the STAR method
    and provides feedback.
    """
    if not request.answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
        
    feedback = interview_prep_service.analyze_answer_with_star(request.answer)
    
    return schemas.STAResponse(feedback=feedback) 