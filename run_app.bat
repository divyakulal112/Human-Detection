@echo off
rem ============================================================
rem  Smart Human Detection System - one-click launcher
rem  Creates the virtual environment and installs dependencies
rem  on first run, then starts the Streamlit dashboard.
rem  Usage:  run_app.bat [port]     (default port 8501)
rem ============================================================
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8501"

if exist ".venv\Scripts\python.exe" goto deps_check
echo [1/3] Creating virtual environment...
py -3.12 -m venv .venv
if errorlevel 1 goto fail_venv
echo [1/3] Virtual environment created.
goto deps_check

:deps_check
if exist ".venv\Lib\site-packages\streamlit" goto start
echo [2/3] Installing dependencies (first run only, takes a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail_deps
goto start

:start
echo [3/3] Starting the dashboard on port %PORT%...
echo       Press Ctrl+C here or close this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% --browser.gatherUsageStats false
goto :eof

:fail_venv
echo ERROR: could not create the virtual environment.
echo        Install Python 3.12 from https://www.python.org/downloads/ and retry.
pause
exit /b 1

:fail_deps
echo ERROR: dependency installation failed. Check your internet connection and rerun.
pause
exit /b 1