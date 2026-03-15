@echo off
REM ICT V7 MT5 Trading Bot - Auto-restart script
REM Run this in a separate command window or with nssm/pm2

echo ============================================================
echo ICT V7 MT5 Trading Bot
echo ============================================================
echo.

:start
echo [%date% %time%] Starting bot...
py ict_v7_mt5.py --symbols "EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,USDCHF,NZDUSD,EURGBP,EURJPY,GBPJPY,EURAUD,EURCAD,GBPAUD,AUDJPY,CADJPY,CHFJPY,XAUUSD,XAGUSD,XTIUSD,XBRUSD,US30,USTEC,US500,UK100,AUS200" --mode live --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9" --risk 0.03 --rr 3.0

echo.
echo [%date% %time%] Bot stopped. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto start
