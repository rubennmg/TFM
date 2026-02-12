rem Execute benchmarks for all pipelines defined in the pipelines directory.

set "PIPELINES_DIR=pipelines"
set "PROFILE_COUNT=0"

for %%F in ("%PIPELINES_DIR%\*.json") do (
    set /A PROFILE_COUNT+=1
)

if %PROFILE_COUNT% equ 0 (
    echo Error: No pipelines found in %PIPELINES_DIR%
    exit /b 1
)

for %%F in ("%PIPELINES_DIR%\*.json") do (
    set "FILENAME=%%~nF"
    echo   - !FILENAME:~0,-5!
)

echo.
echo Executing benchmarks...
echo.

for %%F in ("%PIPELINES_DIR%\*.json") do (
    set "FILENAME=%%~nF"
    set "PROFILE=!FILENAME:~0,-5!"
    
    echo ================================
    echo Executing benchmark: !PROFILE!
    echo ================================
    
    pytest benchmark/run.py --bench-profile="!PIPELINES_DIR!/!PROFILE!"
    
    if errorlevel 1 (
        echo Benchmark '!PROFILE!' finished with error
    ) else (
        echo Benchmark '!PROFILE!' completed successfully
    )
    echo.
)

echo ================================
echo All benchmarks have finished
echo ================================
