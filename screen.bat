@echo off
REM Windows launcher. Finds a usable Python, builds the virtual environment on
REM first run, then hands every argument through to the screener.
REM
REM Structure note: the interpreter probe lives in a :findpython subroutine
REM rather than inside a for-loop block, because cmd.exe parses a whole
REM parenthesised block up front and mangles quoted parentheses inside it. The
REM version test also avoids ">" on purpose - it is a redirection operator.
setlocal
cd /d "%~dp0"

set "PY="
call :findpython py
call :findpython python
call :findpython python3
if not defined PY goto nopython

if not exist ".venv\Scripts\python.exe" goto setup
goto run

:setup
echo Setting up for first use with "%PY%" - this takes a minute.
echo.
%PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" goto novenv
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto nodeps
echo.
echo Setup complete.
echo.
goto run

:run
".venv\Scripts\python.exe" -m nse_cup_screener.cli %*
exit /b %errorlevel%

:findpython
if defined PY exit /b 0
%~1 -c "import sys;sys.exit(0 if max(sys.version_info[:2],(3,9))==sys.version_info[:2] else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PY=%~1"
exit /b 0

:nopython
echo.
echo   Could not find Python 3.9 or newer on this machine.
echo.
echo   Install it - easiest first:
echo.
echo       winget install -e --id Python.Python.3.12
echo.
echo   or download it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH" on the FIRST screen of the installer.
echo.
echo   IMPORTANT: after installing, close this window and open a NEW
echo   Command Prompt. An already-open terminal keeps the old PATH and
echo   will still say Python is missing. Then run screen.bat again.
echo.
exit /b 1

:novenv
echo.
echo   Failed to create the virtual environment.
echo   If Python came from the Microsoft Store, install it from
echo   python.org instead - the Store build is sandboxed and often
echo   cannot create one here.
echo.
exit /b 1

:nodeps
echo.
echo   Failed to install dependencies. Check your internet connection,
echo   then run screen.bat again.
echo.
exit /b 1
