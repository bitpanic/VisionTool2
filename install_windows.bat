@echo off
REM VisionTool Windows Installation Script
REM This script sets up a Python virtual environment and installs dependencies.

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.9+ and rerun this script.
    pause
    exit /b 1
)

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo Installation complete!
pause 