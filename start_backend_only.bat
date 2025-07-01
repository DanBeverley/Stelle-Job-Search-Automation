@echo off
echo Starting Backend Only...
echo.

cd ai_job_search_app\backend
echo ✅ Starting FastAPI server on http://localhost:8000
echo ✅ API Documentation: http://localhost:8000/docs
echo.
python -m uvicorn main:app --reload --port 8000