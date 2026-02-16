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

echo Copying MI-GAN ONNX model to appfiles...
if exist "%APP_DIR%\migan_pipeline_v2.onnx" (
    echo Model already exists in %APP_DIR%, skipping copy.
) else (
    if not exist "%BASE_DIR%migan_pipeline_v2.onnx" (
        echo Error: migan_pipeline_v2.onnx not found in %BASE_DIR%.
        echo Please ensure the repository was cloned correctly.
        pause
        exit /b 1
    )
    copy "%BASE_DIR%migan_pipeline_v2.onnx" "%APP_DIR%\migan_pipeline_v2.onnx"
    if errorlevel 1 (
        echo Failed to copy model file.
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
