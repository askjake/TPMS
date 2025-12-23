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

REM Check for HackRF
echo [1/3] Checking for HackRF One...
hackrf_info >nul 2>&1
if errorlevel 1 (
    echo [WARNING] HackRF not detected
    echo.
    echo Please ensure:
    echo 1. HackRF One is plugged in
    echo 2. WinUSB driver is installed (use Zadig)
    echo 3. Try a different USB port
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
) else (
    echo [OK] HackRF One detected
)

REM Check dependencies
echo.
echo [2/3] Checking dependencies...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [ERROR] Dependencies not installed
    echo Please run install.bat first
    pause
    exit /b 1
)
echo [OK] Dependencies found

REM Start the application
echo.
echo [3/3] Starting TPMS Tracker...
echo ========================================
echo.
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

python start_tracker.py

pause
