import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .. import schemas
from ..models.db.database import get_db
from ..models.db import user as user_model
from ..utils.encryption import encrypt_data
from .auth import get_current_active_user

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
    db_user = db.query(user_model.User).filter(user_model.User.id == current_user.id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Convert dict to JSON string before encrypting
    resume_json_string = json.dumps(resume_data.content)

    # Encrypt the data
    encrypted_resume = encrypt_data(resume_json_string)

    # Update user's resume data
    db_user.encrypted_resume_data = encrypted_resume
    db.commit()
    db.refresh(db_user)

    return {"message": "Resume saved successfully."} 