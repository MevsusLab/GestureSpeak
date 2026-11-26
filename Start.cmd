@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
if not exist ".venv\gesturespeak-installed" (
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto failed
    type nul > ".venv\gesturespeak-installed"
)
".venv\Scripts\python.exe" setup_assets.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" app.py --open
if errorlevel 1 goto failed
exit /b 0

:failed
echo GestureSpeak could not start. Check the message above and see README.md.
pause
exit /b 1
