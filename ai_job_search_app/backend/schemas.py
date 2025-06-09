from pydantic import BaseModel

class UserBase(BaseModel):
    email:str
class UserCreate(UserBase):
    password:str
class User(UserCreate):
    id:int
    is_active:bool
    class Config:
        from_attributes=True
class Token(BaseModel):
    access_token:str
    token_type:str
class TokenData(BaseModel):
    email:str|None = None