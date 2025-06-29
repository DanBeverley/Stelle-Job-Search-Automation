@echo off
echo Checking Frontend Setup...
echo.

cd ai_job_search_app\frontend

echo [1] Node.js version:
node --version

echo.
echo [2] NPM version:
npm --version

echo.
echo [3] Checking if react-scripts exists:
if exist node_modules\.bin\react-scripts.cmd (
    echo ✅ react-scripts found
) else (
    echo ❌ react-scripts NOT found
    echo Need to run: npm install
)

echo.
echo [4] Package.json scripts:
type package.json | findstr "scripts" -A 5

echo.
echo [5] Testing direct path:
if exist node_modules\.bin\react-scripts.cmd (
    echo ✅ react-scripts.cmd exists
    node_modules\.bin\react-scripts.cmd --version
) else (
    echo ❌ react-scripts.cmd missing
)

echo.
pause