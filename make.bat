@echo off
REM Build automation for ADE-FCM on Windows
REM Usage: make.bat [target]

if "%1"=="" (
    echo Available targets: install, test, lint, format, benchmark, docker-build, docker-run, clean
    exit /b 1
)

if "%1"=="install" (
    pip install -e .
    goto :eof
)

if "%1"=="install-all" (
    pip install -e .[all]
    goto :eof
)

if "%1"=="test" (
    pytest tests/ -v --cov=src --cov-report=term --cov-report=html
    goto :eof
)

if "%1"=="lint" (
    flake8 src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
    black --check src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
    isort --check-only src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
    goto :eof
)

if "%1"=="format" (
    black src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
    isort src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
    goto :eof
)

if "%1"=="benchmark" (
    python -m benchmarks.main
    goto :eof
)

if "%1"=="docker-build" (
    docker build -t ade-fcm:latest .
    goto :eof
)

if "%1"=="docker-run" (
    docker run --rm -v %CD%/results:/app/results ade-fcm:latest
    goto :eof
)

if "%1"=="clean" (
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist *.egg-info rmdir /s /q *.egg-info
    if exist .pytest_cache rmdir /s /q .pytest_cache
    if exist .coverage del /q .coverage
    if exist htmlcov rmdir /s /q htmlcov
    if exist results\plots (
        del /q results\plots\*.png 2>nul
        del /q results\plots\*.pdf 2>nul
    )
    for /r %%i in (*.pyc) do del /q "%%i"
    for /r %%i in (*.pyo) do del /q "%%i"
    for /r %%i in (*.so) do del /q "%%i"
    goto :eof
)

echo Unknown target: %1
echo Available targets: install, test, lint, format, benchmark, docker-build, docker-run, clean
exit /b 1
