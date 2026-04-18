@echo off
REM Install dependencies for Windows 11 + Python 3.10 + CUDA 12.9
REM Run this from the repo root: install_win.bat

echo ============================================================
echo  PINN-available - Windows 11 / CUDA 12.9 setup
echo ============================================================

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 and add to PATH.
    exit /b 1
)

REM Install PyTorch with CUDA 12.9 index
echo.
echo [1/2] Installing PyTorch with CUDA 12.9 support ...
pip install torch --index-url https://download.pytorch.org/whl/cu129

REM Install remaining dependencies
echo.
echo [2/2] Installing numpy, scipy, matplotlib ...
pip install "numpy>=1.24,<2.0" "scipy>=1.10" "matplotlib>=3.7"

echo.
echo ============================================================
echo  Done. Verify CUDA is visible:
echo    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo ============================================================
