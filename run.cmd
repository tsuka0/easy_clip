@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d %~dp0

set "VENV_CPU=%~dp0.venv_cpu"
set "VENV_GPU=%~dp0.venv_gpu"
set "PY_EXE="
set "CUDA_OK="
set "GPU_SELECTED=0"
set "CUDA_CHECK_FILE=%TEMP%\\cuda_ok.txt"

if exist "%VENV_GPU%\\Scripts\\python.exe" (
  "%VENV_GPU%\\Scripts\\python.exe" -c "import torch; print(1 if torch.cuda.is_available() else 0)" > "%CUDA_CHECK_FILE%" 2>nul
  if exist "%CUDA_CHECK_FILE%" (
    set /p CUDA_OK=<"%CUDA_CHECK_FILE%"
    del /f /q "%CUDA_CHECK_FILE%" >nul 2>&1
    if "!CUDA_OK!"=="1" (
      set "PY_EXE=%VENV_GPU%\\Scripts\\python.exe"
      set "GPU_SELECTED=1"
    )
  )
)

if not defined PY_EXE (
  if exist "%VENV_CPU%\\Scripts\\python.exe" (
    set "PY_EXE=%VENV_CPU%\\Scripts\\python.exe"
  )
)

if not defined PY_EXE (
  echo No valid environment found.
  echo Run install_CPU.bat or install_GPU.bat first.
  pause
  exit /b 1
)

if not exist "%~dp0model\\sam2.1_hiera_large.pt" (
  echo Efficient-SAM2 model not found.
  echo Please run install_CPU.bat or install_GPU.bat to download the model.
  pause
  exit /b 1
)

if "%GPU_SELECTED%"=="1" (
  echo GPU mode
) else (
  echo CPU mode
)
"%PY_EXE%" "%~dp0main.py"
endlocal
