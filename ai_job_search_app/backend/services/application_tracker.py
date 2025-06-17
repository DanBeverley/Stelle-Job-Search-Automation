from sqlalchemy.orm import Session
from ..models.db import application as application_model
from .. import schemas

def get_application_by_id(db: Session, application_id: int, user_id: int):
    return db.query(application_model.Application).filter(
        application_model.Application.id == application_id,
        application_model.Application.user_id == user_id
    ).first()

def get_applications_for_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(application_model.Application).filter(
        application_model.Application.user_id == user_id
    ).offset(skip).limit(limit).all()

def create_application_for_user(db: Session, application: schemas.ApplicationCreate, user_id: int):
    db_application = application_model.Application(**application.model_dump(), user_id=user_id)
    db.add(db_application)
    db.commit()
    db.refresh(db_application)
    return db_application

def update_application(db: Session, application_id: int, application_update: schemas.ApplicationUpdate, user_id: int):
    db_application = get_application_by_id(db=db, application_id=application_id, user_id=user_id)
    if db_application:
        update_data = application_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_application, key, value)
        db.commit()
        db.refresh(db_application)
    return db_application

def delete_application(db: Session, application_id: int, user_id: int):
    db_application = get_application_by_id(db=db, application_id=application_id, user_id=user_id)
    if db_application:
        db.delete(db_application)
        db.commit()
    return db_application 