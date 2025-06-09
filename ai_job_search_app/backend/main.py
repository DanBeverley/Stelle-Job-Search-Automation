from fastapi import FastAPI
from .api import auth, cv_parser
from .models.db import user
from .models.db.database import engine

user.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(cv_parser.router, prefix="/cv", tags=["cv"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Job Search API"} 