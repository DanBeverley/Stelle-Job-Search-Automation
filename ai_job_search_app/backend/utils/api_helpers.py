"""
Common API utilities and helper functions to reduce code redundancy across API modules.
"""
import logging
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import schemas
from models.db import user as user_model

logger = logging.getLogger(__name__)


def get_user_with_cv_data(
    db: Session, 
    current_user: schemas.User, 
    require_cv: bool = True
) -> user_model.User:
    """
    Common function to retrieve user from database and optionally validate CV data exists.
    
    Args:
        db: Database session
        current_user: Current authenticated user
        require_cv: Whether to require CV data to be present
        
    Returns:
        User model instance
        
    Raises:
        HTTPException: If user not found or CV data missing when required
    """
    db_user = db.query(user_model.User).filter(
        user_model.User.id == current_user.id
    ).first()
    
    if not db_user:
        logger.warning("User not found in database: %s", current_user.email)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if require_cv and not db_user.parsed_cv_data:
        logger.warning("CV data not found for user: %s", current_user.email)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CV data not found. Please upload and parse a CV first."
        )
    
    return db_user


def validate_non_empty_string(value: Optional[str], field_name: str) -> None:
    """
    Validates that a string field is not None or empty.
    
    Args:
        value: The string value to validate
        field_name: Name of the field for error messages
        
    Raises:
        HTTPException: If value is None or empty
    """
    if not value or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot be empty."
        )


def handle_service_error(error: Exception, service_name: str) -> HTTPException:
    """
    Standardized error handling for service layer exceptions.
    
    Args:
        error: The exception that occurred
        service_name: Name of the service for logging/error messages
        
    Returns:
        HTTPException with appropriate status code and message
    """
    error_msg = str(error)
    logger.error("%s service error: %s", service_name, error_msg)
    
    if isinstance(error, ValueError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    elif isinstance(error, RuntimeError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{service_name} service is currently unavailable"
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred in {service_name}: {error_msg}"
        )


def check_resource_exists(resource: Optional[object], resource_type: str) -> None:
    """
    Generic function to check if a resource exists and raise appropriate error if not.
    
    Args:
        resource: The resource to check
        resource_type: Type of resource for error message
        
    Raises:
        HTTPException: If resource is None
    """
    if resource is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_type} not found"
        )