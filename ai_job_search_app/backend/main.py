from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import auth, cv_parser, resume_builder, job_search, interview_prep, salary_prediction, application, cover_letter, skill_analysis, health
from .models.db.database import engine, Base
from .models.db import user as user_model
from .models.db import application as application_model
from .utils.logging_config import setup_logging, get_logger
from .config.settings import get_settings

# Initialize settings
settings = get_settings()

# Setup logging configuration
setup_logging(
    level=settings.log_level,
    log_file=settings.log_file
)
logger = get_logger(__name__)

# Validate configuration on startup
missing_settings = settings.validate_required_settings()
if missing_settings:
    for setting in missing_settings:
        logger.error("Configuration error: %s", setting)
    if settings.is_production():
        raise RuntimeError("Invalid configuration for production environment")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
)

# Add CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
app.include_router(health.router, prefix="/api", tags=["Health Check"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(application.router, prefix="/api/applications", tags=["Application Tracker"])
app.include_router(cover_letter.router, prefix="/api/cover-letter", tags=["Cover Letter Generator"])
app.include_router(cv_parser.router, prefix="/api/cv", tags=["CV Parser"])
app.include_router(resume_builder.router, prefix="/api/resume", tags=["Resume Builder"])
app.include_router(job_search.router, prefix="/api/jobs", tags=["Job Search"])
app.include_router(interview_prep.router, prefix="/api/interview", tags=["Interview Preparation"])
app.include_router(salary_prediction.router, prefix="/api/salary", tags=["Salary Prediction"])
app.include_router(skill_analysis.router, prefix="/api/skills", tags=["Skill Analysis"])

@app.on_event("startup")
def on_startup():
    """Initialize database tables and log application startup."""
    logger.info("Starting AI Job Search Application...")
    # Uses the Base metadata to create all tables.
    # Ensure table definition by importing 
    # Registered with the Base metadata before this call is made.
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Job Search API V2"} 