"""
Test centralized configuration management.
"""
import pytest
import os
from unittest.mock import patch
from fastapi import status


class TestConfigurationManagement:
    """Test centralized configuration system."""
    
    def test_settings_loading_with_defaults(self):
        """Test that settings load with appropriate defaults."""
        from backend.config.settings import Settings
        
        settings = Settings()
        
        # Test defaults
        assert settings.app_name == "AI Job Search Application"
        assert settings.app_version == "2.0.0"
        assert settings.environment == "development"
        assert settings.algorithm == "HS256"
        assert settings.log_level == "INFO"
        assert settings.database_url == "sqlite:///./ai_job_search.db"
        assert len(settings.secret_key) >= 32
        assert len(settings.encryption_key) == 64
    
    def test_settings_with_environment_variables(self):
        """Test settings loading from environment variables."""
        from backend.config.settings import Settings
        
        test_env = {
            "SECRET_KEY": "test-secret-key-12345678901234567890123456789012",
            "ENCRYPTION_KEY": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "LOG_LEVEL": "DEBUG",
            "ENVIRONMENT": "testing",
            "DATABASE_URL": "sqlite:///:memory:",
            "ADZUNA_APP_ID": "test_app_id",
            "ADZUNA_APP_KEY": "test_app_key"
        }
        
        with patch.dict(os.environ, test_env):
            settings = Settings()
            
            assert settings.secret_key == test_env["SECRET_KEY"]
            assert settings.encryption_key == test_env["ENCRYPTION_KEY"]
            assert settings.log_level == "DEBUG"
            assert settings.environment == "testing"
            assert settings.database_url == "sqlite:///:memory:"
            assert settings.adzuna_app_id == "test_app_id"
            assert settings.adzuna_app_key == "test_app_key"
    
    def test_environment_detection_methods(self):
        """Test environment detection helper methods."""
        from backend.config.settings import Settings
        
        # Test development environment
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.is_development()
            assert not settings.is_production()
            assert not settings.is_testing()
        
        # Test production environment
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.is_production()
            assert not settings.is_development()
            assert not settings.is_testing()
        
        # Test testing environment
        with patch.dict(os.environ, {"TESTING": "true"}):
            settings = Settings()
            assert settings.is_testing()
    
    def test_database_url_for_testing(self):
        """Test that testing environment uses in-memory database."""
        from backend.config.settings import Settings
        
        with patch.dict(os.environ, {"TESTING": "true"}):
            settings = Settings()
            assert settings.get_database_url() == "sqlite:///:memory:"
    
    def test_configuration_validation_development(self):
        """Test configuration validation in development."""
        from backend.config.settings import Settings
        
        settings = Settings()
        issues = settings.validate_required_settings()
        
        # Development should have minimal issues
        assert isinstance(issues, list)
        # Should not fail on default settings in development
    
    def test_configuration_validation_production(self):
        """Test configuration validation in production."""
        from backend.config.settings import Settings
        
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "DEBUG": "true",  # This should cause an error
            "SECRET_KEY": "short"  # This should cause an error
        }):
            settings = Settings()
            issues = settings.validate_required_settings()
            
            assert len(issues) > 0
            assert any("DEBUG must be False" in issue for issue in issues)
            assert any("SECRET_KEY must be at least 32 characters" in issue for issue in issues)
    
    def test_encryption_key_validation(self):
        """Test encryption key validation."""
        from backend.config.settings import Settings
        from pydantic import ValidationError
        
        # Test invalid length
        with patch.dict(os.environ, {"ENCRYPTION_KEY": "too_short"}):
            with pytest.raises(ValidationError):
                Settings()
        
        # Test invalid hex
        with patch.dict(os.environ, {"ENCRYPTION_KEY": "g" * 64}):
            with pytest.raises(ValidationError):
                Settings()
        
        # Test valid key
        with patch.dict(os.environ, {"ENCRYPTION_KEY": "0" * 64}):
            settings = Settings()
            assert settings.encryption_key == "0" * 64
    
    def test_model_path_properties(self):
        """Test model path property methods."""
        from backend.config.settings import Settings
        
        settings = Settings()
        
        assert settings.salary_model_full_path.endswith("salary_predictor_xgboost.json")
        assert settings.salary_preprocessor_full_path.endswith("salary_predictor_preprocessor.joblib")
        assert settings.models_directory in settings.salary_model_full_path
    
    def test_settings_caching(self):
        """Test that settings are cached properly."""
        from backend.config.settings import get_settings
        
        # Get settings twice
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance due to caching
        assert settings1 is settings2


class TestHealthCheckEndpoints:
    """Test health check endpoints that show configuration status."""
    
    def test_basic_health_check(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/api/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "AI Job Search Application" in data["message"]
        assert "environment" in data
        assert "timestamp" in data
    
    def test_detailed_health_check(self, test_client):
        """Test detailed health check endpoint."""
        response = test_client.get("/api/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify structure
        required_sections = ["app_info", "configuration", "external_apis", "model_paths", "security"]
        for section in required_sections:
            assert section in data
        
        # Verify app info
        assert data["app_info"]["name"] == "AI Job Search Application"
        assert "version" in data["app_info"]
        assert "environment" in data["app_info"]
        
        # Verify configuration section
        assert "log_level" in data["configuration"]
        assert "database_configured" in data["configuration"]
        
        # Verify external APIs section
        assert "adzuna_configured" in data["external_apis"]
        assert "theirstack_configured" in data["external_apis"]
        
        # Verify model paths section
        assert "models_directory" in data["model_paths"]
        assert "salary_model_exists" in data["model_paths"]
        
        # Verify security section
        assert "encryption_key_configured" in data["security"]
        assert "secret_key_configured" in data["security"]
    
    def test_configuration_validation_endpoint(self, test_client):
        """Test configuration validation endpoint."""
        response = test_client.get("/api/config/validate")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "valid" in data
        assert "environment" in data
        assert "issues_count" in data
        assert "issues" in data
        assert isinstance(data["issues"], list)
    
    def test_configuration_overview_endpoint(self, test_client):
        """Test configuration overview endpoint."""
        response = test_client.get("/api/config/settings")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify safe configuration data is returned
        safe_fields = [
            "app_name", "app_version", "environment", "debug", "log_level",
            "cors_enabled", "api_docs_enabled", "max_file_size", 
            "allowed_file_extensions", "rate_limit_requests_per_minute"
        ]
        
        for field in safe_fields:
            assert field in data
        
        # Verify sensitive data is not exposed
        assert "secret_key" not in data
        assert "encryption_key" not in data
        assert "database_url" not in data
        
        # Verify external APIs status (boolean only, not actual keys)
        assert "external_apis" in data
        assert isinstance(data["external_apis"]["adzuna"], bool)
        assert isinstance(data["external_apis"]["theirstack"], bool)


class TestConfigurationIntegration:
    """Test integration of configuration with other components."""
    
    def test_security_uses_centralized_config(self):
        """Test that security module uses centralized configuration."""
        from backend.security import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
        from backend.config.settings import get_settings
        
        settings = get_settings()
        
        assert SECRET_KEY == settings.secret_key
        assert ALGORITHM == settings.algorithm
        assert ACCESS_TOKEN_EXPIRE_MINUTES == settings.access_token_expire_minutes
    
    def test_encryption_uses_centralized_config(self):
        """Test that encryption module uses centralized configuration."""
        from backend.utils.encryption import ENCRYPTION_KEY
        from backend.config.settings import get_settings
        
        settings = get_settings()
        
        assert ENCRYPTION_KEY == settings.encryption_key
    
    def test_job_providers_use_centralized_config(self):
        """Test that job search providers use centralized configuration."""
        from backend.services.job_search_providers.adzuna_api import ADZUNA_APP_ID, ADZUNA_API_URL
        from backend.services.job_search_providers.theirstack_api import THEIRSTACK_API_URL
        from backend.config.settings import get_settings
        
        settings = get_settings()
        
        assert ADZUNA_API_URL == settings.adzuna_api_url
        assert THEIRSTACK_API_URL == settings.theirstack_api_url
    
    def test_main_app_uses_centralized_config(self, test_client):
        """Test that main FastAPI app uses centralized configuration."""
        # Test that the app is configured with settings
        response = test_client.get("/docs")
        
        # If API docs are enabled (default), should get OpenAPI docs
        # If disabled in config, should get 404
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]