from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey
from .database import Base

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String, index=True)
    company = Column(String, index=True)
    status = Column(String, default="Applied")
    date_applied = Column(String)
    notes = Column(Text, nullable=True)
    cover_letter_text = Column(Text, nullable=True)
    
    user_id = Column(Integer, ForeignKey("users.id")) 