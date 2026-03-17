@echo off
setlocal EnableExtensions
cd /d %~dp0

if exist model\\sam3.pt (
  echo Model already exists. Setup already completed.
  exit /b 0
)

echo [1/6] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
  echo Python 3.12 not found. Installing via winget...
  winget install -e --id Python.Python.3.12
)

set VENV_DIR=.venv
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo [2/6] Creating virtual environment...
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -3.12 -m venv "%VENV_DIR%"
  ) else (
    python -m venv "%VENV_DIR%"
  )
)
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo Failed to create virtual environment.
  exit /b 1
)

echo [3/6] Upgrading pip...
"%VENV_DIR%\\Scripts\\python.exe" -m pip install --upgrade pip

echo [4/6] Installing PyTorch (CUDA cu126 only)...
set TORCH_INDEX=https://download.pytorch.org/whl/cu126
"%VENV_DIR%\\Scripts\\python.exe" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url %TORCH_INDEX%

echo [5/6] Installing SAM3 and app dependencies...
"%VENV_DIR%\\Scripts\\python.exe" -m pip install git+https://github.com/facebookresearch/sam3.git
"%VENV_DIR%\\Scripts\\python.exe" -m pip install huggingface_hub
"%VENV_DIR%\\Scripts\\python.exe" -m pip install -r requirements.txt

echo [6/6] Downloading SAM3 checkpoint (Hugging Face)...
if not exist model mkdir model
echo If you haven't logged in, run: huggingface-cli login
"%VENV_DIR%\\Scripts\\huggingface-cli.exe" download facebook/sam3 sam3.pt --local-dir model --local-dir-use-symlinks False
if not exist model\\sam3.pt (
  echo Download failed. Please ensure Hugging Face access is granted and try again.
  exit /b 1
)

echo.
echo Setup complete. Run run.cmd to start the app.
endlocal
