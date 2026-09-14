@echo off
setlocal
set "UZNORM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%UZNORM_PYTHON%" (
  echo Python muhiti topilmadi. README bo'yicha .venv yarating.
  exit /b 2
)
"%UZNORM_PYTHON%" -B -X utf8 "%~dp0studio.py" %*
exit /b %errorlevel%
