@echo off
REM Quick Zadig Driver Installer
REM =============================

echo.
echo ========================================
echo HackRF WinUSB Driver Installation
echo ========================================
echo.

REM Download Zadig if not present
if not exist zadig.exe (
    echo Downloading Zadig...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/pbatard/libwdi/releases/download/v1.5.0/zadig-2.8.exe' -OutFile 'zadig.exe'}"

    if not exist zadig.exe (
        echo Failed to download Zadig
        echo Please download manually from: https://zadig.akeo.ie/
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting Zadig...
echo ========================================
echo.
echo INSTRUCTIONS:
echo 1. In Zadig, click 'Options' -^> 'List All Devices'
echo 2. Select 'HackRF One' from dropdown
echo 3. Ensure 'WinUSB' is selected as target driver
echo 4. Click 'Replace Driver' or 'Install Driver'
echo 5. Wait for completion
echo 6. Close Zadig
echo 7. Unplug and replug HackRF
echo.
echo Press any key to launch Zadig...
pause >nul

REM Run Zadig as Administrator
powershell -Command "Start-Process -FilePath '%cd%\zadig.exe' -Verb RunAs"

echo.
echo After installing the driver:
echo 1. Unplug HackRF
echo 2. Plug it back in
echo 3. Run: start.bat
echo.
pause
