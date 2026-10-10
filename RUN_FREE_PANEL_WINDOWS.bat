@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
 echo First run SETUP_FREE_PANEL_WINDOWS.bat
 pause
 exit /b 1
)
.venv\Scripts\python.exe run.py doctor --config config_free_panel.json
if errorlevel 1 goto fail
.venv\Scripts\python.exe verify_free_panel.py
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m unittest test_free_panel_check
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py smoke --config config_free_panel.json
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py prepare --config config_free_panel.json --raw data\free_panel_raw --prepared data\free_panel_prepared
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py run --config config_free_panel.json --raw data\free_panel_raw --prepared data\free_panel_prepared --out results\free_panel_v2
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py verify --config config_free_panel.json --raw data\free_panel_raw --prepared data\free_panel_prepared --out results\free_panel_v2
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py summarize --config config_free_panel.json --raw data\free_panel_raw --prepared data\free_panel_prepared --out results\free_panel_v2
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py pack --config config_free_panel.json --raw data\free_panel_raw --prepared data\free_panel_prepared --out results\free_panel_v2
if errorlevel 1 goto fail
echo COMPLETE. Send results\free_panel_v2_RESULTS.zip
pause
exit /b 0
:fail
echo Stopped. Preserve all data/results. Fix the displayed error and rerun this file to resume.
pause
exit /b 1
