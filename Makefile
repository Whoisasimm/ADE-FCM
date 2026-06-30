.PHONY: install test lint benchmark docker-build docker-run clean

install:
	pip install -e .

install-all:
	pip install -e .[all]

test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

lint:
	flake8 src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
	black --check src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
	isort --check-only src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/

format:
	black src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/
	isort src/ benchmarks/ streaming/ xai/ ablation/ gpu/ tests/

benchmark:
	python -m benchmarks.main

docker-build:
	docker build -t ade-fcm:latest .

docker-run:
	docker run --rm -v $(PWD)/results:/app/results ade-fcm:latest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf results/plots/*.png results/plots/*.pdf
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete
