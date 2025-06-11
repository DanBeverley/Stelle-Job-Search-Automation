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