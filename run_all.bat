@echo off
rem Windows wrapper. The driver itself is cross-platform: python run_all.py [options]
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run_all.py %*
) else (
    python run_all.py %*
)
