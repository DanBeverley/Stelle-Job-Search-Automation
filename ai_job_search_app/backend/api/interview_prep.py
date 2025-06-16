from fastapi import APIRouter, Depends, HTTPException
from ..api.auth import get_current_active_user
from .. import schemas
from ..services.gemini_service import generate_interview_questions_with_gemini

router = APIRouter()

@router.post("/interview-prep/generate-questions", response_model=schemas.InterviewQuestions)
async def get_interview_questions(
    request: schemas.InterviewPrepRequest,
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Takes a job description and optional CV text, and returns a list of
    generated mock interview questions from Gemini to help the user prepare.
    """
    if not request.job_description:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    # In a future version, would fetch the user's latest parsed CV from the DB
    # if request.cv_text is not provided.
    cv_text = request.cv_text or "" # Pass an empty string if CV is not provided
    
    questions = generate_interview_questions_with_gemini(
        job_description=request.job_description, 
        cv_text=cv_text
    )
    
    return schemas.InterviewQuestions(questions=questions) 