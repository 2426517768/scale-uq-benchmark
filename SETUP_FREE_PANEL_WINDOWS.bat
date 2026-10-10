@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 py -3.12 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements_free_panel.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip freeze > environment_lock.txt
.venv\Scripts\python.exe run.py doctor --config config_free_panel.json
if errorlevel 1 goto fail
echo SETUP OK. Next: RUN_FREE_PANEL_WINDOWS.bat
pause
exit /b 0
:fail
echo Setup stopped. Read the error. Python 3.11/3.12 and a working NVIDIA driver are required.
pause
exit /b 1
