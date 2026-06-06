@echo off
REM ============================================================================
REM install_libs.bat
REM FinBench Multi-Agent Business Analyst AI
REM
REM One-shot installer for the 6 FinBench support libraries.
REM Run from the finbench_agent project root with venv ACTIVATED.
REM
REM Usage:
REM   1. Open CMD in D:\projects\finbench_agent
REM   2. Activate venv:   venv\Scripts\activate
REM   3. Run:             install_libs.bat
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo  FinBench 8-lib installer
echo ================================================================
echo.

REM Check venv is activated
if "%VIRTUAL_ENV%"=="" (
    echo [ERROR] No virtual environment detected.
    echo Activate venv first:  venv\Scripts\activate
    exit /b 1
)

echo Using venv: %VIRTUAL_ENV%
echo.

REM ─── 1. maths_lib ──────────────────────────────────────────────────────────
echo [1/8] Installing maths_lib...
if exist "D:\projects\maths_lib_" (
    pip install -e D:\projects\maths_lib_ --quiet
    if !errorlevel! neq 0 echo   [WARN] maths_lib install failed
) else (
    echo   [SKIP] D:\projects\maths_lib_ not found
)

REM ─── 2. extract_lib ────────────────────────────────────────────────────────
echo [2/8] Installing extract_lib...
if exist "D:\projects\fina_extractor_lib" (
    pip install -e D:\projects\fina_extractor_lib --quiet
    if !errorlevel! neq 0 echo   [WARN] extract_lib install failed
) else (
    echo   [SKIP] D:\projects\fina_extractor_lib not found
)

REM ─── 3. pattern_lib ────────────────────────────────────────────────────────
echo [3/8] Installing pattern_lib...
if exist "D:\projects\fina_pattern_lib\fina_pattern_lib\pyproject.toml" (
    pip install -e D:\projects\fina_pattern_lib\fina_pattern_lib --quiet
    if !errorlevel! neq 0 echo   [WARN] pattern_lib install failed
) else (
    echo   [SKIP] D:\projects\fina_pattern_lib\fina_pattern_lib not found
)

REM ─── 4. format_lib ─────────────────────────────────────────────────────────
echo [4/8] Installing format_lib...
if exist "D:\projects\fina_format_lib" (
    pip install -e D:\projects\fina_format_lib --quiet
    if !errorlevel! neq 0 echo   [WARN] format_lib install failed
) else (
    echo   [SKIP] D:\projects\fina_format_lib not found
)

REM ─── 5. logic_lib ──────────────────────────────────────────────────────────
echo [5/8] Installing logic_lib...
if exist "D:\projects\Fina_Logic_lib\logic_lib_" (
    pip install -e D:\projects\Fina_Logic_lib\logic_lib_ --quiet
    if !errorlevel! neq 0 echo   [WARN] logic_lib install failed
) else (
    echo   [SKIP] D:\projects\Fina_Logic_lib\logic_lib_ not found
)

REM ─── 6. algo_lib ───────────────────────────────────────────────────────────
echo [6/8] Installing algo_lib...
if exist "D:\projects\fina_algo_lib" (
    pip install -e D:\projects\fina_algo_lib --quiet
    if !errorlevel! neq 0 echo   [WARN] algo_lib install failed
) else (
    echo   [SKIP] D:\projects\fina_algo_lib not found
)

REM ─── 7. verify_lib ─────────────────────────────────────────────────────────
echo [7/8] Installing verify_lib...
if exist "D:\projects\verify_lib_" (
    pip install -e D:\projects\verify_lib_ --quiet
    if !errorlevel! neq 0 echo   [WARN] verify_lib install failed
) else (
    echo   [SKIP] D:\projects\verify_lib_ not found
)

REM === 8. question_lib (JEE side-word decomposer) ===========================
echo [8/8] Installing question_lib...
if exist "D:\projects\fina_question_lib\pyproject.toml" (
    pip install -e D:\projects\fina_question_lib --quiet
    if !errorlevel! neq 0 echo   [WARN] question_lib install failed
) else (
    echo   [SKIP] D:\projects\fina_question_lib not found
)

echo.
echo ================================================================
echo  Verifying installation
echo ================================================================
echo.

python -c "from src.utils.lib_bridge import lib_summary, lib_status; print(lib_summary()); [print(f'  {k:<14}= {v}') for k,v in lib_status().items()]"

echo.
echo ================================================================
echo  Done. If any lib shows False above, check the [SKIP]/[WARN] log.
echo ================================================================
echo.

endlocal
