@echo off
echo Fixing Frontend Dependencies...
echo.

cd ai_job_search_app\frontend

echo [1/3] Cleaning node_modules...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json

echo [2/3] Installing dependencies...
npm install

echo [3/3] Installing react-scripts specifically...
npm install react-scripts@5.0.1 --save-dev

echo.
echo ✅ Frontend dependencies fixed!
echo.
echo Now you can run: npm start
pause