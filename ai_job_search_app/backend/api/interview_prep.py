from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.auth import get_current_active_user
import schemas
from models.db.database import get_db
from services.ml_service import MLService
from services import interview_prep_service
from utils.api_helpers import get_user_with_cv_data, validate_non_empty_string

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
    validate_non_empty_string(request.job_description, "Job description")
    db_user = get_user_with_cv_data(db, current_user, require_cv=True)

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
    validate_non_empty_string(request.answer, "Answer")
        
    feedback = interview_prep_service.analyze_answer_with_star(request.answer)
    
    return schemas.STAResponse(feedback=feedback)

@router.post("/generate-response")
def generate_interview_response(request: dict):
    """
    Generate interview response using our working ML service
    """
    try:
        ml_service = MLService()
        response = ml_service.generate_interview_response(request.get('question', ''))
        return {'response': response}
    except Exception as e:
        return {'error': str(e), 'response': 'Sorry, I could not generate a response at this time.'} 