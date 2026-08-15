@echo off
REM Convenience wrapper for Windows: runs the screener with the project's venv.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run this first:
    echo     py -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)
".venv\Scripts\python.exe" -m nse_cup_screener.cli %*
