@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" run.py %*
) else if exist "%PY311%" (
    "%PY311%" run.py %*
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 run.py %*
    ) else (
        python run.py %*
    )
)
if errorlevel 1 pause
