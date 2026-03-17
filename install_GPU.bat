@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d %~dp0
set "ROOT_DIR=%~dp0"
set "REPO_DIR=%ROOT_DIR%efficient-sam2"

echo [1/7] Checking Python 3.11...
set "PY_CMD="
where py >nul 2>&1
if errorlevel 0 set "PY_CMD=py -3.11"
set "PY311_EXE=%LocalAppData%\\Programs\\Python\\Python311\\python.exe"
if exist "%PY311_EXE%" set "PY_CMD=%PY311_EXE%"
if "%PY_CMD%"=="" (
  echo Python 3.11 not found. Installing...
  where winget >nul 2>&1
  if errorlevel 1 (
    echo winget is not available. Opening Python download page...
    start "" "https://www.python.org/downloads/windows/"
    echo Install Python 3.11, then re-run install_GPU.bat.
    pause
    exit /b 1
  )
  winget install -e --id Python.Python.3.11
  if errorlevel 1 (
    echo winget failed. Opening Python download page...
    start "" "https://www.python.org/downloads/windows/"
    echo Install Python 3.11, then re-run install_GPU.bat.
    pause
    exit /b 1
  )
  set "PY_CMD="
  where py >nul 2>&1
  if errorlevel 0 set "PY_CMD=py -3.11"
  if exist "%PY311_EXE%" set "PY_CMD=%PY311_EXE%"
)
if "%PY_CMD%"=="" (
  echo Python 3.11 not found after install. Please install Python 3.11 and re-run.
  exit /b 1
)
for /f "delims=" %%i in ('%PY_CMD% -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2^>nul') do set "PY_VER=%%i"
if "%PY_VER%"=="" (
  echo Python not found after install. Please install Python 3.11 and re-run.
  exit /b 1
)
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do set "PY_MAJ=%%a" & set "PY_MIN=%%b"
if %PY_MAJ% LSS 3 (
  echo Python %PY_VER% is too old. Install Python 3.11 and re-run.
  start "" "https://www.python.org/downloads/windows/"
  exit /b 1
)
if %PY_MAJ% EQU 3 if %PY_MIN% LSS 11 (
  echo Python %PY_VER% is too old. Installing Python 3.11...
  where winget >nul 2>&1
  if errorlevel 1 (
    start "" "https://www.python.org/downloads/windows/"
    exit /b 1
  )
  winget install -e --id Python.Python.3.11
)

set "VENV_DIR=%ROOT_DIR%.venv_gpu"
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo [2/7] Creating virtual environment...
  %PY_CMD% -m venv "%VENV_DIR%"
)
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo Failed to create virtual environment.
  exit /b 1
)
set "VENV_PY=%VENV_DIR%\\Scripts\\python.exe"

echo [3/7] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel

echo [4/7] Installing PyTorch (GPU)...
set "TORCH_CUDA=cu121"
set "CUDA_VER=unknown"
set "SMI_EXE=nvidia-smi"
if exist "%SystemRoot%\\System32\\nvidia-smi.exe" set "SMI_EXE=%SystemRoot%\\System32\\nvidia-smi.exe"
for /f "delims=" %%i in ('powershell -NoProfile -Command "$v='unknown'; try { $out=& '%SMI_EXE%' 2>$null; if ($out -match 'CUDA Version:\\s*([0-9]+)\\.([0-9]+)') { $v=$matches[1]+'.'+$matches[2] } } catch {} ; $v"') do set "CUDA_VER=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "$cuda='cu121'; try { $out=& '%SMI_EXE%' 2>$null; if ($out -match 'CUDA Version:\\s*([0-9]+)\\.([0-9]+)') { $maj=[int]$matches[1]; if ($maj -le 11) { $cuda='cu118' } } } catch {} ; $cuda"') do set "TORCH_CUDA=%%i"
echo Detected CUDA Version: %CUDA_VER% (using %TORCH_CUDA%)
set "TORCH_INDEX=https://download.pytorch.org/whl/%TORCH_CUDA%"
"%VENV_PY%" -c "import torch, torchvision; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
  echo PyTorch already installed (CUDA). Skipping.
) else (
  "%VENV_PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  "%VENV_PY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url %TORCH_INDEX%
)

echo [5/7] Installing Efficient-SAM2 and app dependencies...
if not exist "%REPO_DIR%\\sam2" (
  echo Cloning Efficient-SAM2...
  where git >nul 2>&1
  if errorlevel 1 (
    echo Git not found. Downloading zip with curl...
  ) else (
    git clone https://github.com/jingjing0419/Efficient-SAM2.git "%REPO_DIR%"
  )
  if errorlevel 1 (
    echo Git clone failed. Downloading zip with curl...
    set "ZIP_FILE=%TEMP%\\efficient-sam2.zip"
    set "ZIP_DIR=%TEMP%\\efficient-sam2-src"
    if exist "!ZIP_FILE!" del /f /q "!ZIP_FILE!"
    if exist "!ZIP_DIR!" rmdir /s /q "!ZIP_DIR!"
    curl -fL -o "!ZIP_FILE!" "https://github.com/jingjing0419/Efficient-SAM2/archive/refs/heads/main.zip"
    if not exist "!ZIP_FILE!" (
      echo curl download failed.
      exit /b 1
    )
    powershell -Command "Expand-Archive -Path '!ZIP_FILE!' -DestinationPath '!ZIP_DIR!' -Force; $src=Join-Path '!ZIP_DIR!' 'Efficient-SAM2-main'; if (Test-Path $src) {Move-Item $src '%REPO_DIR%' -Force}"
  )
)
if not exist "%REPO_DIR%\\sam2" (
  echo Efficient-SAM2 repo not found. Please check antivirus/Windows Defender.
  exit /b 1
)
pushd "%REPO_DIR%"
"%VENV_PY%" -m pip install -e .
popd
"%VENV_PY%" -m pip install -r "%ROOT_DIR%requirements.txt"

echo [6/7] Downloading Efficient-SAM2 checkpoint...
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

echo [7/7] Done.
echo Setup complete. Run run.cmd to start the app.
endlocal
