@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting UI (default http://127.0.0.1:7860 , auto next port if busy)...
python app.py
pause
