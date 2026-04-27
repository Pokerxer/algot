@echo off
REM ICT V7 MT5 Trading Bot - Auto-restart script
REM Run this in a separate command window or with nssm/pm2

echo ============================================================
echo ICT V7 MT5 Trading Bot - Mixed Mode
echo Reverse: GBPAUD,USDCHF,EURJPY,GBPJPY (time-gated: off 08:00 + 19-22 UTC)
echo Normal:  USDJPY (signal-following, no reverse)
echo Stopped: NZDUSD,BTCUSD,ETHUSD,XRPUSD,SOLUSD
echo ============================================================
echo.

REM Close any existing ICT V7 positions first (keep V8 orders)
echo [%date% %time%] Closing ICT V7 positions, keeping V8 orders...
py close_v7_positions.py --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9"
echo.

REM Fix any existing positions with wrong TP
echo [%date% %time%] Checking and fixing existing positions...
py fix_existing_positions.py --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9" --rr 3.0
echo.

:start
echo [%date% %time%] Starting bot...
py ict_v7_mt5.py --symbols "GBPAUD,USDJPY,USDCHF,EURJPY,GBPJPY" --mode live --login 298797826 --password "Pokerx_007" --server "Exness-MT5Trial9" --risk 0.01 --rr 2.0 --confluence 75 --max-loss 500 --reverse "GBPAUD,USDCHF,EURJPY,GBPJPY" --max-positions 1

echo.
echo [%date% %time%] Bot stopped. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto start
