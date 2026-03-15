@echo off
REM ====================================
REM ICT V7 MT5 Trading Bot - Windows Setup
REM ====================================

echo.
echo ====================================
echo ICT V7 - MT5 Trading Bot Setup
echo ====================================
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Please run as Administrator
    pause
    exit /b 1
)

REM Install NSSM if not exists
where nssm >nul 2>&1
if %errorLevel% neq 0 (
    echo [1/4] Installing NSSM...
    powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile 'nssm.zip'"
    powershell -Command "Expand-Archive -Path 'nssm.zip' -DestinationPath 'nssm_temp' -Force"
    copy "nssm_temp\nssm-2.24\win64\nssm.exe" "C:\Windows\System32\nssm.exe" >nul
    del nssm.zip
    rmdir /s /q nssm_temp
    echo [OK] NSSM installed
) else (
    echo [OK] NSSM already installed
)

REM Find Python
where py >nul 2>&1
if %errorLevel% equ 0 (
    set PYTHON=py
) else (
    where python >nul 2>&1
    if %errorLevel% equ 0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found
        pause
        exit /b 1
    )
)
echo [OK] Python found: %PYTHON%

REM Install dependencies
echo [2/4] Installing Python dependencies...
%PYTHON% -m pip install MetaTrader5 pandas numpy pytz python-telegram-bot requests >nul 2>&1
echo [OK] Dependencies installed

REM Git pull
echo [3/4] Pulling latest code...
if exist ".git" (
    git pull
) else (
    echo [WARNING] Not a git repository, skipping pull
)

REM Install and start service
echo [4/4] Installing Windows service...
nssm install ICTV7 "%PYTHON%" "C:\algot\ict_v7_mt5.py"
nssm set ICTV7 AppDirectory "C:\algot"
nssm set ICTV7 AppParameters "--symbols EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,USDCHF,NZDUSD,EURGBP,EURJPY,GBPJPY,EURAUD,EURCAD,GBPAUD,AUDJPY,CADJPY,CHFJPY,XAUUSD,XAGUSD,XTIUSD,XBRUSD,US30,USTEC,US500,UK100,AUS200 --mode live --login 298797826 --password Pokerx_007 --server Exness-MT5Trial9"
nssm set ICTV7 Start SERVICE_AUTO_START

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo Service commands:
echo   nssm start ICTV7    - Start service
echo   nssm stop ICTV7     - Stop service
echo   nssm restart ICTV7  - Restart service
echo   nssm remove ICTV7  - Remove service
echo.
echo Log file: C:\algot\ictv7.log
echo.

REM Start the service
echo Starting ICTV7 service...
nssm start ICTV7

echo Done!
pause
