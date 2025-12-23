@echo off
REM TPMS Tracker - Windows Installation Script
REM ============================================

echo.
echo ========================================
echo TPMS Tracker - Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Python found
python --version

REM Check if git is installed (optional)
git --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git not found - skipping repository check
) else (
    echo [2/6] Git found
    git --version
)

REM Create virtual environment
echo.
echo [3/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install Python dependencies
echo.
echo [5/6] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully

REM Install HackRF tools
echo.
echo [6/6] Installing HackRF tools...
python install_hackrf.py
if errorlevel 1 (
    echo [WARNING] HackRF tools installation had issues
    echo You may need to install manually
)

REM Create desktop shortcut (optional)
echo.
echo Creating shortcuts...
call create_shortcuts.bat

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo NEXT STEPS:
echo 1. CLOSE THIS WINDOW and open a NEW terminal
echo 2. Plug in your HackRF One
echo 3. Install WinUSB driver using Zadig:
echo    - Download: https://zadig.akeo.ie/
echo    - Select 'HackRF One' device
echo    - Install WinUSB driver
echo 4. Run: start.bat
echo.
echo Press any key to exit...
pause >nul
