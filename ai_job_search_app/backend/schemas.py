from pydantic import BaseModel
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
    cv_text: Optional[str] = None # CV text can be passed directly or fetched for the user

class InterviewQuestions(BaseModel):
    questions: List[str]