"""
Health check and system status API endpoints.
"""
import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, status
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/health", summary="Health Check")
def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    Returns system status and configuration info.
    """
    return {
        "status": "healthy",
        "message": f"Welcome to {settings.app_name} v{settings.app_version}",
        "environment": settings.environment,
        "timestamp": "2024-06-26"
    }


@router.get("/health/detailed", summary="Detailed Health Check")
def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with configuration and service status.
    """
    health_status = {
        "status": "healthy",
        "app_info": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
            "testing": settings.testing
        },
        "configuration": {
            "log_level": settings.log_level,
            "database_configured": bool(settings.database_url),
            "cors_enabled": settings.cors_enabled,
            "api_docs_enabled": settings.api_docs_enabled
        },
        "external_apis": {
            "adzuna_configured": bool(settings.adzuna_app_id and settings.adzuna_app_key),
            "theirstack_configured": bool(settings.theirstack_api_key),
            "google_api_configured": bool(settings.google_api_key)
        },
        "model_paths": {
            "models_directory": settings.models_directory,
            "models_directory_exists": os.path.exists(settings.models_directory),
            "salary_model_exists": os.path.exists(settings.salary_model_full_path),
            "salary_preprocessor_exists": os.path.exists(settings.salary_preprocessor_full_path)
        },
        "security": {
            "encryption_key_configured": bool(settings.encryption_key),
            "secret_key_configured": bool(settings.secret_key),
            "token_expiry_minutes": settings.access_token_expire_minutes
        }
    }
    
    # Check for configuration issues
    config_issues = settings.validate_required_settings()
    if config_issues:
        health_status["status"] = "degraded"
        health_status["configuration_issues"] = config_issues
        logger.warning("Configuration issues found: %s", config_issues)
    
    return health_status


@router.get("/config/validate", summary="Validate Configuration")
def validate_configuration() -> Dict[str, Any]:
    """
    Validate the current configuration and return any issues.
    """
    config_issues = settings.validate_required_settings()
    
    validation_result = {
        "valid": len(config_issues) == 0,
        "environment": settings.environment,
        "issues_count": len(config_issues),
        "issues": config_issues
    }
    
    if not validation_result["valid"]:
        logger.error("Configuration validation failed: %s", config_issues)
    
    return validation_result


@router.get("/config/settings", summary="Configuration Overview")
def get_configuration_overview() -> Dict[str, Any]:
    """
    Get a safe overview of current configuration (no sensitive data).
    """
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "database_type": "sqlite" if "sqlite" in settings.database_url else "other",
        "cors_enabled": settings.cors_enabled,
        "api_docs_enabled": settings.api_docs_enabled,
        "max_file_size": settings.max_file_size,
        "allowed_file_extensions": settings.allowed_file_extensions,
        "rate_limit_requests_per_minute": settings.rate_limit_requests_per_minute,
        "cache_ttl_seconds": settings.cache_ttl_seconds,
        "external_apis": {
            "adzuna": bool(settings.adzuna_app_id),
            "theirstack": bool(settings.theirstack_api_key),
            "google_api": bool(settings.google_api_key)
        },
        "notification_services": {
            "smtp": bool(settings.smtp_username),
            "aws_ses": bool(settings.aws_access_key_id),
            "twilio": bool(settings.twilio_account_sid)
        }
    }