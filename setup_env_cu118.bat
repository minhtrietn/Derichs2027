@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if exist "%PY311%" (
    "%PY311%" run.py --setup --setup-only %*
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 run.py --setup --setup-only %*
    ) else (
        python run.py --setup --setup-only %*
    )
)
pause
