import json
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

import schemas
from models.db.database import get_db
from utils.encryption import encrypt_data
from api.auth import get_current_active_user
from utils.api_helpers import get_user_with_cv_data

router = APIRouter(
    tags=["resume"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", status_code=status.HTTP_200_OK)
def save_resume(
    resume_data: schemas.ResumeData,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_active_user)
):
    """
    Saves or updates a user's resume data.

    The incoming resume data (JSON) is converted to a string, encrypted,
    and then stored in the database.
    """
    db_user = get_user_with_cv_data(db, current_user, require_cv=False)

    # Convert dict to JSON string before encrypting
    resume_json_string = json.dumps(resume_data.content)

    # Encrypt the data
    encrypted_resume = encrypt_data(resume_json_string)

    # Update user's resume data
    db_user.encrypted_resume_data = encrypted_resume
    db.commit()
    db.refresh(db_user)

    return {"message": "Resume saved successfully."} 