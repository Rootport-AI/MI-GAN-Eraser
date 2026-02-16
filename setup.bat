@echo off
set PYTHONUTF8=1
echo Starting MI-GAN-Eraser setup...

set BASE_DIR=%~dp0
set APP_DIR=%BASE_DIR%appfiles
echo Setting up in: %APP_DIR%

if not exist "%APP_DIR%" mkdir "%APP_DIR%"

echo Creating virtual environment...
python -m venv "%APP_DIR%\venv"
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%APP_DIR%\venv\Scripts\activate"

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing requirements from require.txt...
pip install -r "%BASE_DIR%require.txt"
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo Downloading MI-GAN ONNX model from Hugging Face...
curl -L -o "%APP_DIR%\migan_pipeline_v2.onnx" ^
    https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx
if errorlevel 1 (
    echo Failed to download MI-GAN model. Check your internet connection.
    pause
    exit /b 1
)

REM Verify download (file should be approximately 29.5MB)
for %%A in ("%APP_DIR%\migan_pipeline_v2.onnx") do (
    if %%~zA LSS 1000000 (
        echo Error: Downloaded model file is too small. Download may have failed.
        del "%APP_DIR%\migan_pipeline_v2.onnx"
        pause
        exit /b 1
    )
)

echo.
echo Setup completed successfully.
echo Model installed in: %APP_DIR%\migan_pipeline_v2.onnx
echo.
echo To start the application, run: run.bat
pause
