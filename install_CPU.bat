@echo off
setlocal EnableExtensions
cd /d %~dp0
set "ROOT_DIR=%~dp0"
set "REPO_DIR=%ROOT_DIR%efficient-sam2"

echo [1/6] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
  echo Python not found. Installing Python 3.11 via winget...
  winget install -e --id Python.Python.3.11
)

set "VENV_DIR=%ROOT_DIR%.venv_cpu"
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo [2/6] Creating virtual environment...
  python -m venv "%VENV_DIR%"
)
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo Failed to create virtual environment.
  exit /b 1
)
set "VENV_PY=%VENV_DIR%\\Scripts\\python.exe"

echo [3/6] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel

echo [4/6] Installing PyTorch (CPU only)...
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
"%VENV_PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
"%VENV_PY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url %TORCH_INDEX%

echo [5/6] Installing Efficient-SAM2 and app dependencies...
if not exist "%REPO_DIR%\\.git" (
  git clone https://github.com/jingjing0419/Efficient-SAM2.git "%REPO_DIR%"
)
pushd "%REPO_DIR%"
"%VENV_PY%" -m pip install -e .
popd
"%VENV_PY%" -m pip install -r "%ROOT_DIR%requirements.txt"

echo [6/6] Downloading Efficient-SAM2 checkpoint...
set "MODEL_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
set "MODEL_URL_ALT=https://dl.fbaipublicfiles.com/segment_anything_2/sam2.1_hiera_large.pt"
set "MODEL_PATH=%ROOT_DIR%model\\sam2.1_hiera_large.pt"
if not exist "%ROOT_DIR%model" mkdir "%ROOT_DIR%model"
if exist "%MODEL_PATH%" (
  echo Checkpoint already exists.
) else (
  echo Trying curl...
  curl -fL -A "Mozilla/5.0" -o "%MODEL_PATH%" %MODEL_URL%
  if not exist "%MODEL_PATH%" (
    echo Primary download failed. Trying alternate URL...
    curl -fL -A "Mozilla/5.0" -o "%MODEL_PATH%" %MODEL_URL_ALT%
  )
  if not exist "%MODEL_PATH%" (
    echo Curl failed. Trying PowerShell...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri %MODEL_URL% -OutFile '%MODEL_PATH%' -Headers @{\"User-Agent\"=\"Mozilla/5.0\"}"
  )
)
set "MODEL_SIZE=0"
for %%F in ("%MODEL_PATH%") do set "MODEL_SIZE=%%~zF"
if "%MODEL_SIZE%"=="0" (
  echo Download failed (file is empty).
  exit /b 1
)
if %MODEL_SIZE% LSS 1000000 (
  echo Download failed (file too small: %MODEL_SIZE% bytes).
  exit /b 1
)
if not exist "%MODEL_PATH%" (
  echo Download failed. Please re-run install_CPU.bat or check network access.
  exit /b 1
)

echo.
echo Setup complete. Run run.cmd to start the app.
endlocal
