@echo off
REM Neural Network Steganography - Web Interface Launcher (Windows)
REM Quick start script for the web application

echo ========================================
echo Neural Network Steganography
echo Web Interface Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Flask not found
    echo Installing Flask...
    pip install flask werkzeug
)

echo Flask installed

REM Check if PyTorch is installed
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Error: PyTorch not found
    echo Please install PyTorch first:
    echo   pip install torch torchvision
    pause
    exit /b 1
)

echo PyTorch installed

REM Check CUDA availability
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: CUDA not available (using CPU)
) else (
    echo CUDA available
)

echo.
echo ========================================
echo Starting web server...
echo ========================================
echo.
echo Access the web interface at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Flask application
python app.py

pause
