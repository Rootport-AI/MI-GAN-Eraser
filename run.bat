@echo off
setlocal

set "BASE_DIR=%~dp0"
set "APP_DIR=%BASE_DIR%appfiles"

if not exist "%APP_DIR%" (
    echo Error: Application directory not found at %APP_DIR%.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%APP_DIR%\venv\Scripts\activate"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

cd /d "%BASE_DIR%"

echo Starting MI-GAN-Eraser...
echo Open http://localhost:7859 in your browser.
echo note: add --listen to allow LAN access (debug forced OFF)

python app.py %*
if errorlevel 1 (
    echo Error: Application failed to start.
    pause
    exit /b 1
)

pause
endlocal
