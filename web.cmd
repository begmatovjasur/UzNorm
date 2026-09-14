@echo off
setlocal
cd /d "%~dp0"
echo UzNorm: http://127.0.0.1:8765/
echo Keep this window open while using the local website.
call "%~dp0run.cmd" web
if errorlevel 1 pause
