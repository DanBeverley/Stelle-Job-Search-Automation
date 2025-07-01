@echo off
echo Starting AI Job Search Application...
echo.

echo [1/2] Starting Backend Server...
start "Backend" cmd /k "cd ai_job_search_app\backend && python -m uvicorn main:app --reload --port 8000"

echo [2/2] Starting Frontend Server...
timeout /t 3
start "Frontend" cmd /k "cd ai_job_search_app\frontend && npm start"

echo.
echo ✅ Both servers are starting!
echo ✅ Backend: http://localhost:8000
echo ✅ Frontend: http://localhost:3000  
echo ✅ API Docs: http://localhost:8000/docs
echo.
echo If frontend fails, run fix_frontend.bat first
echo Close this window when done.