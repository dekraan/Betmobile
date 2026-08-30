@echo off
REM ===================================================================
REM run_collect_t1.bat
REM Wekelijkse verzameling van T1-meetpunten (Pinnacle vs Bet365)
REM
REM Het script is idempotent: fixtures die al verwerkt zijn worden
REM overgeslagen. Een gemiste week haalt zichzelf de keer erna in.
REM ===================================================================

set PYEXE=C:\Users\Gebruiker\AppData\Local\Programs\Python\Python313\python.exe
set WORKDIR=C:\Users\Gebruiker\Documents\Betmobile\eci_engine
set LOGDIR=%WORKDIR%\output\logs

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Datumstempel voor de lognaam (YYYY-MM-DD)

cd /d "%WORKDIR%"

echo ================================================== >> "%LOGDIR%\collect_t1.log"
echo Start: %date% %time% >> "%LOGDIR%\collect_t1.log"

"%PYEXE%" collect_t1_horizons.py >> "%LOGDIR%\collect_t1.log" 2>&1
set RC=%ERRORLEVEL%

echo Exit code: %RC% >> "%LOGDIR%\collect_t1.log"
echo Einde: %date% %time% >> "%LOGDIR%\collect_t1.log"

exit /b %RC%