@echo off
REM ICT V7 MT5 Trading Bot - Auto-restart script
REM Run this in a separate command window or with nssm/pm2

echo ============================================================
echo ICT V7 MT5 Trading Bot - Best Performers
echo ============================================================
echo.

REM Fix any existing positions with wrong TP first
echo [%date% %time%] Checking and fixing existing positions...
py fix_existing_positions.py --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9" --rr 3.0
echo.

:start
echo [%date% %time%] Starting bot...
py ict_v7_mt5.py --symbols "GBPAUD,USDJPY,USDCHF,EURJPY,GBPJPY,NZDUSD" --mode live --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9" --risk 0.02 --rr 3.0 --confluence 65 --max-loss 500

echo.
echo [%date% %time%] Bot stopped. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto start
