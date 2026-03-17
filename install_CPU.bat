@echo off
setlocal EnableExtensions
cd /d %~dp0

echo SAM3 requires an NVIDIA GPU with CUDA. CPU install is not supported.
echo Please run install_GPU(Nvidia only).bat
endlocal
