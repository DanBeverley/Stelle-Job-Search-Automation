"""
Centralized configuration management for AI Job Search Application.
All environment variables, API keys, and configuration settings are managed here.
"""
import os
import secrets
from typing import Optional, List
from pydantic import validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation and type hints."""
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    app_name: str = "AI Job Search Application"
    app_version: str = "2.0.0"
    environment: str = "development"
    debug: bool = False
    testing: bool = False
    
    # =============================================================================
    # SECURITY SETTINGS
    # =============================================================================
    secret_key: str = secrets.token_urlsafe(32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30 * 24 * 60  # 30 days
    
    # Data encryption key (must be 64 hex characters for AES-256)
    encryption_key: str = secrets.token_hex(32)
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if len(v) != 64:
            raise ValueError('Encryption key must be 64 hex characters (32 bytes)')
        try:
            int(v, 16)  # Validate it's valid hex
        except ValueError:
            raise ValueError('Encryption key must be valid hexadecimal')
        return v
    
    # =============================================================================
    # DATABASE SETTINGS
    # =============================================================================
    database_url: str = "sqlite:///./ai_job_search.db"
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # =============================================================================
    # LOGGING SETTINGS
    # =============================================================================
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # =============================================================================
    # ML MODEL SETTINGS
    # =============================================================================
    models_directory: str = "ai_job_search_app/final_model"
    data_directory: str = "ai_job_search_app/data"
    
    # Model file paths
    bert_model_path: str = "ai_job_search_app/data/models/bert_resume_classifier"
    bert_quantized_model_path: str = "ai_job_search_app/data/models/bert_resume_classifier_quantized"
    
    # Salary prediction model
    salary_model_file: str = "salary_predictor_xgboost.json"
    salary_preprocessor_file: str = "salary_predictor_preprocessor.joblib"
    
    # Cover letter model settings
    cover_letter_base_model: str = "microsoft/DialoGPT-medium"
    cover_letter_adapter_path: str = "ai_job_search_app/final_model"
    
    # Interview prep model settings  
    interview_prep_base_model: str = "microsoft/DialoGPT-medium"
    interview_prep_adapter_path: str = "ai_job_search_app/final_model"
    
    # Model loading settings
    model_device: str = "auto"  # auto, cpu, cuda
    model_cache_dir: Optional[str] = None
    model_load_in_8bit: bool = False
    model_load_in_4bit: bool = False
    
    @property
    def salary_model_full_path(self) -> str:
        return os.path.join(self.models_directory, self.salary_model_file)
    
    @property
    def salary_preprocessor_full_path(self) -> str:
        return os.path.join(self.models_directory, self.salary_preprocessor_file)
    
    # =============================================================================
    # EXTERNAL API SETTINGS
    # =============================================================================
    
    # Job Search APIs
    adzuna_app_id: Optional[str] = None
    adzuna_app_key: Optional[str] = None
    adzuna_api_url: str = "http://api.adzuna.com/v1/api/jobs/gb/search/1"
    
    theirstack_api_key: Optional[str] = None
    theirstack_api_url: str = "https://api.theirstack.com/v1/jobs/search"
    
    # Geolocation API
    geolocation_api_url: str = "http://ip-api.com/json"
    geolocation_timeout: int = 5
    
    # Google Gemini API (for AI features)
    google_api_key: Optional[str] = None
    
    # Course recommendation APIs
    edx_api_base_url: str = "https://courses.edx.org/api"
    coursera_api_base_url: str = "https://api.coursera.org/api"
    
    # =============================================================================
    # API RATE LIMITING SETTINGS
    # =============================================================================
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    # External API rate limits
    adzuna_rate_limit: int = 200  # requests per hour
    theirstack_rate_limit: int = 1000  # requests per day
    
    # =============================================================================
    # FILE UPLOAD SETTINGS
    # =============================================================================
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_file_extensions: List[str] = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
    upload_directory: str = "uploads"
    
    # CV processing settings
    max_cv_text_length: int = 50000  # characters
    cv_processing_timeout: int = 30  # seconds
    
    # =============================================================================
    # CACHE SETTINGS
    # =============================================================================
    cache_ttl_seconds: int = 300  # 5 minutes
    skill_analysis_cache_ttl: int = 3600  # 1 hour
    job_search_cache_ttl: int = 900  # 15 minutes
    
    # =============================================================================
    # NOTIFICATION SETTINGS (Future use)
    # =============================================================================
    # Email settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # AWS SES settings
    aws_ses_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # SMS settings
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None
    
    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    
    # Job search settings
    max_jobs_per_search: int = 50
    job_search_timeout: int = 10
    
    # AI processing settings
    max_concurrent_ai_requests: int = 5
    ai_processing_timeout: int = 60
    
    # =============================================================================
    # DEVELOPMENT SETTINGS
    # =============================================================================
    reload_on_change: bool = False
    api_docs_enabled: bool = True
    cors_enabled: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # =============================================================================
    # MONITORING AND METRICS
    # =============================================================================
    metrics_enabled: bool = True
    health_check_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    
    # =============================================================================
    # CONFIGURATION LOADING
    # =============================================================================
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ('settings_',)  # Fix model_ field warnings
    }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.testing or self.environment.lower() == "testing"
    
    def get_database_url(self) -> str:
        """Get database URL with appropriate settings for environment."""
        if self.is_testing():
            return "sqlite:///:memory:"
        return self.database_url
    
    def validate_required_settings(self) -> List[str]:
        """Validate that all required settings are properly configured."""
        missing = []
        
        if self.is_production():
            # Production-specific validations
            if not self.secret_key or len(self.secret_key) < 32:
                missing.append("SECRET_KEY must be at least 32 characters in production")
            
            if not self.encryption_key:
                missing.append("ENCRYPTION_KEY is required in production")
            
            if self.debug:
                missing.append("DEBUG must be False in production")
        
        # General validations
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            missing.append(f"Invalid LOG_LEVEL: {self.log_level}")
        
        return missing


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached).
    This function is cached to avoid recreating the settings object multiple times.
    """
    return Settings()


# Convenience function for backwards compatibility
def get_config() -> Settings:
    """Alias for get_settings()."""
    return get_settings()


# Global settings instance
settings = get_settings()