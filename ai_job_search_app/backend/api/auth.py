from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import schemas
from models.db import user as user_model, crud
from security import (
    verify_password, 
    create_access_token, 
    get_password_hash,
    SECRET_KEY,
    ALGORITHM
)
from models.db.database import get_db
from config.settings import get_settings


router = APIRouter()

# Password reset schemas
class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# In-memory storage for reset tokens (in production, use Redis or database)
reset_tokens = {}

# Email helper function
def send_reset_email(email: str, reset_token: str):
    """Send password reset email to user"""
    settings = get_settings()
    
    # Check if SMTP is configured
    if not settings.smtp_server or not settings.smtp_username or not settings.smtp_password:
        # For development, just log the token
        print(f"[DEV] Password reset token for {email}: {reset_token}")
        print(f"[DEV] Reset URL: http://localhost:3000/reset-password?token={reset_token}")
        return True
    
    try:
        msg = MIMEMultipart()
        msg['From'] = settings.smtp_username
        msg['To'] = email
        msg['Subject'] = "Password Reset - AI Job Search"
        
        body = f"""
        Hi,
        
        You requested a password reset for your AI Job Search account.
        
        Click the link below to reset your password:
        http://localhost:3000/reset-password?token={reset_token}
        
        This link will expire in 1 hour.
        
        If you didn't request this reset, please ignore this email.
        
        Best regards,
        AI Job Search Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(settings.smtp_server, settings.smtp_port)
        if settings.smtp_use_tls:
            server.starttls()
        server.login(settings.smtp_username, settings.smtp_password)
        text = msg.as_string()
        server.sendmail(settings.smtp_username, email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# This is the correct scheme for a simple "Bearer" token in the header.
oauth2_scheme = HTTPBearer(scheme_name="Bearer")

@router.post("/register", response_model=schemas.User)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = crud.create_user(db=db, user=user, hashed_password=hashed_password)
    return new_user

@router.post("/login", response_model=schemas.Token)
def login(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.email}
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(db: Session = Depends(get_db), credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: schemas.User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Send password reset email to user.
    Always returns success to avoid revealing if email exists.
    """
    user = crud.get_user_by_email(db, email=request.email)
    
    if user:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + timedelta(hours=1)
        
        # Store token with expiry
        reset_tokens[reset_token] = {
            "email": request.email,
            "expiry": expiry
        }
        
        # Send email
        email_sent = send_reset_email(request.email, reset_token)
        
        if not email_sent:
            raise HTTPException(
                status_code=500, 
                detail="Failed to send reset email. Please try again later."
            )
    
    # Always return success to avoid email enumeration
    return {"message": "If the email exists, a password reset link has been sent."}

@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset user password using the reset token.
    """
    # Check if token exists and is valid
    if request.token not in reset_tokens:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token."
        )
    
    token_data = reset_tokens[request.token]
    
    # Check if token has expired
    if datetime.utcnow() > token_data["expiry"]:
        # Remove expired token
        del reset_tokens[request.token]
        raise HTTPException(
            status_code=400,
            detail="Reset token has expired. Please request a new one."
        )
    
    # Get user by email
    user = crud.get_user_by_email(db, email=token_data["email"])
    if not user:
        raise HTTPException(
            status_code=400,
            detail="User not found."
        )
    
    # Update password
    hashed_password = get_password_hash(request.new_password)
    user.hashed_password = hashed_password
    db.commit()
    
    # Remove used token
    del reset_tokens[request.token]
    
    return {"message": "Password has been reset successfully."} 