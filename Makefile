# Makefile for PCB Defect Detection System

.PHONY: help install install-dev test lint format clean train validate

help:
	@echo "PCB Defect Detection System - Makefile Commands"
	@echo "================================================"
	@echo "install        - Install production dependencies"
	@echo "install-dev    - Install development dependencies"
	@echo "test           - Run unit tests"
	@echo "test-cov       - Run tests with coverage"
	@echo "lint           - Run code linters"
	@echo "format         - Format code with black and isort"
	@echo "validate       - Validate setup"
	@echo "train          - Train model"
	@echo "train-fast     - Train model (10 epochs for testing)"
	@echo "clean          - Clean generated files"
	@echo "clean-all      - Clean everything including data"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	@echo "Running pylint..."
	pylint src/ --max-line-length=100 --disable=C0103,R0913,R0914
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting with black..."
	black src/ tests/ main.py test_setup.py --line-length=100
	@echo "Sorting imports with isort..."
	isort src/ tests/ main.py test_setup.py --profile black

validate:
	python test_setup.py

train:
	python main.py

train-fast:
	python main.py --epochs 10

train-gpu:
	python main.py --epochs 50 --batch-size 64

clean:
	@echo "Cleaning generated files..."
	rm -rf output/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean
	@echo "Cleaning data..."
	rm -rf data/

# Docker commands
docker-build:
	docker build -t pcb-defect-detector .

docker-run:
	docker run -it --gpus all -v $(PWD)/data:/app/data pcb-defect-detector

# Kaggle commands
kaggle-download:
	python -c "from src.kaggle_setup import KaggleSetup; KaggleSetup().download_dataset('akhatova/pcb-defects')"

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Profiling
profile:
	python -m cProfile -o profile.stats main.py --epochs 1
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

memory-profile:
	python -m memory_profiler main.py --epochs 1

# Model optimization
optimize-model:
	python scripts/optimize_model.py

convert-tflite:
	python scripts/convert_to_tflite.py

# Git commands
git-setup:
	git init
	git add .
	git commit -m "Initial commit: PCB Defect Detection System"

# CI/CD
ci: lint test

# All checks before commit
pre-commit: format lint test
	@echo "✓ All checks passed! Ready to commit."
