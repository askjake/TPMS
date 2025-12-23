@echo off
REM TPMS Tracker - Windows Startup Script
REM ======================================

echo.
echo ========================================
echo TPMS Tracker - Starting...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo [ERROR] Virtual environment not found!
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Python startup script
python start_tracker.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo Press any key to exit...
    pause >nul
)
