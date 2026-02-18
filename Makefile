# IMST-Quant Makefile (REPR-01)
.PHONY: install reproduce test clean

install:
	pip install -e ".[dev]"

reproduce:
	python scripts/ingest_reddit.py || true
	python scripts/raw_to_bronze.py
	python scripts/bronze_to_silver.py
	python scripts/silver_to_sentiment.py
	python scripts/build_features.py
	@echo "Reproduce complete."

test:
	python -m pytest tests/ -v --tb=short

clean:
	rm -rf data/raw/* data/bronze/* data/silver/* data/sentiment/* data/gold/* data/influence/*
	rm -rf .pytest_cache build dist *.egg-info
