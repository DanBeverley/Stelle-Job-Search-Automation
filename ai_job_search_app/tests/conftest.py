"""
Pytest configuration and shared fixtures for the AI Job Search Application tests.
"""
import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

# Import application components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from backend.models.db.database import get_db, Base
from backend.models.db import user as user_model
from backend import schemas


# Test Database Setup
@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine using SQLite in memory."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="function")
def test_client(test_db_session):
    """Create a test client with overridden database dependency."""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# User Fixtures
@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture
def test_user(test_db_session, test_user_data):
    """Create a test user in the database."""
    from backend.models.db.crud import create_user
    from backend.security import get_password_hash
    
    hashed_password = get_password_hash(test_user_data["password"])
    user_schema = schemas.UserCreate(**test_user_data)
    db_user = create_user(test_db_session, user_schema, hashed_password)
    return db_user


@pytest.fixture
def auth_headers(test_client, test_user_data):
    """Get authentication headers for API requests."""
    # Register and login user
    response = test_client.post("/api/auth/register", json=test_user_data)
    assert response.status_code == 200
    
    # Login to get token
    login_data = {
        "username": test_user_data["email"],
        "password": test_user_data["password"]
    }
    response = test_client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# CV Test Data Fixtures
@pytest.fixture
def sample_cv_data():
    """Sample parsed CV data for testing."""
    return {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1234567890"
        },
        "skills": ["Python", "Machine Learning", "FastAPI", "SQL", "Docker"],
        "experience": [
            {
                "company": "Tech Corp",
                "position": "Software Engineer",
                "duration": "2020-2023",
                "description": "Developed ML models and REST APIs"
            }
        ],
        "education": [
            {
                "institution": "University of Technology",
                "degree": "Computer Science",
                "year": "2020"
            }
        ],
        "summary": "Experienced software engineer with ML expertise"
    }


@pytest.fixture
def test_user_with_cv(test_db_session, test_user, sample_cv_data):
    """Create a test user with parsed CV data."""
    test_user.parsed_cv_data = sample_cv_data
    test_db_session.commit()
    test_db_session.refresh(test_user)
    return test_user


# Job Search Test Data
@pytest.fixture
def sample_job_listing():
    """Sample job listing data."""
    return {
        "title": "Senior Python Developer",
        "company": "Tech Solutions Inc",
        "location": "San Francisco, CA",
        "description": "We are looking for a senior Python developer with ML experience...",
        "source": "TestAPI"
    }


@pytest.fixture
def mock_job_search_responses():
    """Mock responses for job search APIs."""
    return {
        "adzuna": [
            {
                "title": "Python Developer",
                "company": "Adzuna Corp",
                "location": "New York, NY",
                "description": "Python development role",
                "source": "Adzuna"
            }
        ],
        "theirstack": [
            {
                "title": "ML Engineer",
                "company": "TheirStack Inc",
                "location": "Seattle, WA", 
                "description": "Machine learning engineering position",
                "source": "TheirStack"
            }
        ]
    }


# ML Model Mocks
@pytest.fixture
def mock_ml_models():
    """Mock ML models to avoid loading actual models in tests."""
    with patch('backend.services.ml_service.CVClassifier') as mock_cv_classifier, \
         patch('backend.services.salary_prediction_service.SalaryPredictor') as mock_salary_predictor, \
         patch('backend.services.cover_letter_service.CoverLetterGenerator') as mock_cover_letter_gen, \
         patch('backend.services.interview_prep_service.InterviewPrepGenerator') as mock_interview_prep:
        
        # Mock CV Classifier
        mock_cv_instance = Mock()
        mock_cv_instance.model_loaded = True
        mock_cv_instance.classify_cv_category.return_value = "software_engineering"
        mock_cv_classifier.return_value = mock_cv_instance
        
        # Mock Salary Predictor
        mock_salary_instance = Mock()
        mock_salary_instance.model_loaded = True
        mock_salary_instance.predict.return_value = 85000.0
        mock_salary_predictor.return_value = mock_salary_instance
        
        # Mock Cover Letter Generator
        mock_cover_instance = Mock()
        mock_cover_instance.model_loaded = True
        mock_cover_instance.generate_cover_letter.return_value = "Dear Hiring Manager, I am writing to apply for..."
        mock_cover_letter_gen.return_value = mock_cover_instance
        
        # Mock Interview Prep Generator
        mock_interview_instance = Mock()
        mock_interview_instance.model_loaded = True
        mock_interview_instance.generate_questions.return_value = ["Tell me about yourself", "What are your strengths?"]
        mock_interview_prep.return_value = mock_interview_instance
        
        yield {
            "cv_classifier": mock_cv_instance,
            "salary_predictor": mock_salary_instance, 
            "cover_letter_generator": mock_cover_instance,
            "interview_prep": mock_interview_instance
        }


# File Upload Mocks
@pytest.fixture
def sample_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(b"Sample PDF content for testing")
        tmp_file.flush()
        yield tmp_file.name
    os.unlink(tmp_file.name)


@pytest.fixture
def mock_external_apis():
    """Mock external API calls."""
    with patch('backend.services.job_search_providers.adzuna_api.search_adzuna_jobs') as mock_adzuna, \
         patch('backend.services.job_search_providers.theirstack_api.search_theirstack_jobs') as mock_theirstack, \
         patch('backend.utils.geolocation.get_location_from_ip') as mock_geolocation:
        
        mock_adzuna.return_value = [
            {
                "title": "Python Developer",
                "company": "Test Company",
                "location": "Test City",
                "description": "Test job description",
                "source": "Adzuna"
            }
        ]
        
        mock_theirstack.return_value = [
            {
                "title": "Data Scientist", 
                "company": "Data Corp",
                "location": "Data City",
                "description": "Data science position",
                "source": "TheirStack"
            }
        ]
        
        mock_geolocation.return_value = {
            "city": "Test City",
            "country": "Test Country"
        }
        
        yield {
            "adzuna": mock_adzuna,
            "theirstack": mock_theirstack,
            "geolocation": mock_geolocation
        }


# Environment Variable Mocks
@pytest.fixture(autouse=True)
def mock_environment_variables():
    """Mock environment variables for testing."""
    test_env = {
        "SECRET_KEY": "test-secret-key-for-jwt-tokens-12345678901234567890",
        "ENCRYPTION_KEY": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env