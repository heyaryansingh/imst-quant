@echo off
REM Windows Makefile for IMST-Quant
if "%1"=="install" pip install -e ".[dev]" && goto :eof
if "%1"=="test" python -m pytest tests/ -v --tb=short && goto :eof
if "%1"=="reproduce" (
  python scripts/ingest_reddit.py
  python scripts/raw_to_bronze.py
  python scripts/bronze_to_silver.py
  python scripts/silver_to_sentiment.py
  python scripts/build_features.py
  echo Reproduce complete.
) && goto :eof
echo Usage: make.bat [install|test|reproduce]
