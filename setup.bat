@echo off
echo [SETUP] Checking Python availability...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.12.
    pause
    exit /b
)

echo [SETUP] Launching automation script...
python setup_env.py
if %errorlevel% neq 0 (
    echo [ERROR] Setup failed. Review errors above.
    pause
    exit /b
)

echo [SUCCESS] Environment is ready.
pause
