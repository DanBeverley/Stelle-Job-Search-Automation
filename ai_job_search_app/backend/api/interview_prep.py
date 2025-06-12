from fastapi import APIRouter, Depends, HTTPException
from ..api.auth import get_current_active_user
from .. import schemas

router = APIRouter()

def generate_mock_questions(job_description: str, cv_text: str = None) -> list[str]:
    """
    Generates mock interview questions based on job description and CV.
    
    This is a placeholder and will be replaced with a call to a fine-tuned NLP model.
    """
    questions = [
        "Can you tell me about yourself and your background?",
        "What interests you about this particular role?",
        "What are your biggest strengths and weaknesses?",
        "Where do you see yourself in five years?"
    ]

    # Simple keyword-based logic for demonstration
    if "python" in job_description.lower():
        questions.append("This role mentions Python. Can you describe a challenging project you've completed using Python?")
        
    if "team" in job_description.lower() or "collaborate" in job_description.lower():
        questions.append("Can you provide an example of a time you successfully collaborated with a team?")

    if "lead" in job_description.lower():
        questions.append("This position involves leadership. What is your leadership style?")

    return questions

@router.post("/interview-prep/generate-questions", response_model=schemas.InterviewQuestions)
async def get_interview_questions(
    request: schemas.InterviewPrepRequest,
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Takes a job description and optional CV text, and returns a list of
    generated mock interview questions to help the user prepare.
    """
    if not request.job_description:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    
    # In a future version, we would fetch the user's latest parsed CV from the DB
    # if request.cv_text is not provided.
    
    questions = generate_mock_questions(request.job_description, request.cv_text)
    
    return schemas.InterviewQuestions(questions=questions) 