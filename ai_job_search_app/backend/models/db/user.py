from sqlalchemy import Boolean, Column, Integer, String, Text, JSON
from .database import Base

class User(Base):
    __tablename__ = "users"
 
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    encrypted_resume_data = Column(Text, nullable=True)
    city = Column(String, nullable=True)
    country = Column(String, nullable=True) 
    parsed_cv_data = Column(JSON, nullable=True) 