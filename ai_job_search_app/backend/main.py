from dotenv import load_dotenv

# Load environment variables from .env file before any other imports
load_dotenv()

from fastapi import FastAPI
from .api import auth, cv_parser, resume_builder, job_search, interview_prep, salary_prediction
from .models.db.database import engine, Base
from .models.db import user  

app = FastAPI()

# Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(cv_parser.router, prefix="/api/cv", tags=["CV Parser"])
app.include_router(resume_builder.router, prefix="/api/resume", tags=["Resume Builder"])
app.include_router(job_search.router, prefix="/api/jobs", tags=["Job Search"])
app.include_router(interview_prep.router, prefix="/api/interview", tags=["Interview Prep"])
app.include_router(salary_prediction.router, prefix="/api/salary", tags=["Salary Prediction"])

@app.on_event("startup")
def on_startup():
    # Create all database tables
    Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Job Search API V2"} 