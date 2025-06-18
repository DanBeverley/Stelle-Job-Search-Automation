from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Resume Schemas
class ResumeDataBase(BaseModel):
    content: Dict[str, Any]

class ResumeData(ResumeDataBase):
    pass

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# User Schemas
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    encrypted_resume_data: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    parsed_cv_data: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

# Job Search Schemas
class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    source: str # e.g., 'Indeed', 'Adzuna'

class JobSearchResult(BaseModel):
    jobs: List[JobListing]

# CV Parser Schemas
class CVParsingResult(BaseModel):
    filename: str
    predicted_category: str

class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    years: Optional[str] = None

class Experience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    period: Optional[str] = None
    description: Optional[str] = None

class CVParsingDetailsResult(BaseModel):
    filename: str
    summary: Optional[str] = None
    skills: List[str] = []
    educations: List[Education] = []
    experiences: List[Experience] = []

# Interview Prep Schemas
class InterviewPrepRequest(BaseModel):
    job_description: str

class InterviewQuestions(BaseModel):
    questions: List[str]

class STAResponse(BaseModel):
    feedback: str

class SalaryPredictionRequest(BaseModel):
    job_title: str = Field(..., example="Software Developer")
    location: str = Field(..., example="California")
    skills: List[str] = Field(..., example=["Python", "FastAPI", "AWS"])
    experience: List[dict] = Field(..., example=[{"company": "Tech Inc.", "role": "Senior Developer", "period": "2018-2022"}])
    education: List[dict] = Field(..., example=[{"institution": "State University", "degree": "B.S. Computer Science", "years": "2014-2018"}])

class SalaryPredictionResponse(BaseModel):
    min_salary: int
    max_salary: int
    median_salary: int
    commentary: str
    currency: str = Field(..., example="USD")
    period: str = Field(..., example="annual")
    error: Optional[str] = None

# Application Tracker Schemas
class ApplicationBase(BaseModel):
    job_title: str
    company: str
    status: str = Field(..., example="Applied")
    date_applied: str # Consider using datetime in the future
    notes: Optional[str] = None
    cover_letter_text: Optional[str] = None

class ApplicationCreate(ApplicationBase):
    pass

class ApplicationUpdate(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    status: Optional[str] = None
    date_applied: Optional[str] = None
    notes: Optional[str] = None
    cover_letter_text: Optional[str] = None

class Application(ApplicationBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True

# Cover Letter Schemas
class CoverLetterRequest(BaseModel):
    job_title: str
    company: str
    job_description: str

class CoverLetterResponse(BaseModel):
    cover_letter_text: str
    prompt_used: str # For debugging and verification