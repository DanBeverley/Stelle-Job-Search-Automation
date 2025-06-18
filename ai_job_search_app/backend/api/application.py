from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import schemas
from ..services import application_tracker as application_service
from ..models.db.database import get_db
from .auth import get_current_active_user

router = APIRouter()

@router.post("/", response_model=schemas.Application, status_code=status.HTTP_201_CREATED)
def create_application(
    application: schemas.ApplicationCreate, 
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Create a new job application entry for the current user.
    """
    return application_service.create_application_for_user(
        db=db, application=application, user_id=current_user.id
    )

@router.get("/", response_model=List[schemas.Application])
def read_applications(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Retrieve all job applications for the current user.
    """
    applications = application_service.get_applications_for_user(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    return applications

@router.get("/{application_id}", response_model=schemas.Application)
def read_application(
    application_id: int, 
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Retrieve a specific job application by its ID.
    """
    db_application = application_service.get_application_by_id(
        db, application_id=application_id, user_id=current_user.id
    )
    if db_application is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return db_application

@router.put("/{application_id}", response_model=schemas.Application)
def update_application(
    application_id: int, 
    application: schemas.ApplicationUpdate, 
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Update a job application's details.
    """
    db_application = application_service.update_application(
        db, application_id=application_id, application_update=application, user_id=current_user.id
    )
    if db_application is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return db_application

@router.delete("/{application_id}", response_model=schemas.Application)
def delete_application(
    application_id: int, 
    db: Session = Depends(get_db), 
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Delete a job application.
    """
    db_application = application_service.delete_application(
        db, application_id=application_id, user_id=current_user.id
    )
    if db_application is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return db_application 