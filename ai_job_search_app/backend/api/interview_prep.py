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

@router.post("/test-questions")
def test_generate_interview_questions(request: dict):
    """
    Test endpoint for generating interview questions based on CV/job info.
    For testing purposes only - doesn't require authentication.
    """
    try:
        # Simulate CV data for testing
        mock_cv_data = {
            "skills": ["Python", "JavaScript", "React", "FastAPI", "Machine Learning"],
            "experience": [
                {
                    "title": "Software Developer",
                    "company": "TechCorp",
                    "duration": "2021-2023",
                    "description": "Developed web applications using Python and React"
                }
            ],
            "education": [
                {
                    "degree": "Computer Science",
                    "institution": "University",
                    "year": "2021"
                }
            ]
        }
        
        job_title = request.get('job_title', 'Software Engineer')
        company = request.get('company', 'Tech Company')
        
        # Generate professional/technical interview questions
        technical_questions = [
            # Algorithm and Data Structure Questions
            f"Implement a solution for finding the optimal path in a graph using {mock_cv_data['skills'][0]}. What algorithm would you choose and why?",
            f"Design a distributed caching system. How would you handle cache invalidation and consistency using {mock_cv_data['skills'][1]}?",
            f"Explain the time and space complexity of quicksort vs mergesort. When would you choose one over the other?",
            f"How would you design a rate limiting system for an API? What data structures and algorithms would you use?",
            
            # System Design Questions
            f"Design a scalable chat application that can handle millions of users. How would you implement it using {mock_cv_data['skills'][2]}?",
            f"How would you design a URL shortening service like bit.ly? Walk me through your database schema and API design.",
            f"Explain how you would implement a recommendation system for an e-commerce platform.",
            f"Design a logging system that can handle billions of log entries per day. How would you ensure performance and reliability?",
            
            # Programming Challenges
            f"Write a function that detects if a linked list has a cycle. What's the optimal approach?",
            f"Implement a thread-safe singleton pattern in {mock_cv_data['skills'][0]}. Explain potential pitfalls.",
            f"How would you reverse a binary tree? Code it step by step and explain your approach.",
            f"Find the kth largest element in an unsorted array. What's the most efficient solution?",
            
            # Technical Problem Solving
            f"A production system using {mock_cv_data['skills'][1]} is experiencing high latency. How would you debug and optimize it?",
            f"How would you implement database transactions with ACID properties? Explain with concrete examples.",
            f"Design a CI/CD pipeline for a {mock_cv_data['skills'][2]} application. What tools and strategies would you use?",
            f"Explain how garbage collection works in modern programming languages. How would you optimize memory usage?"
        ]
        
        # Behavioral questions with technical focus
        behavioral_questions = [
            f"Describe a time when you had to optimize a critical piece of code in {mock_cv_data['skills'][0]}. What was your methodology?",
            f"Tell me about the most challenging technical problem you've solved. How did you approach it?",
            f"How do you stay current with technology trends in {mock_cv_data['skills'][4]} and software engineering?",
            f"Describe a situation where you had to make a trade-off between code quality and delivery speed."
        ]
        
        # Select a mix of questions based on the skills
        questions = technical_questions[:8] + behavioral_questions
        
        return {
            "questions": questions,
            "cv_skills_found": mock_cv_data['skills'],
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

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