.PHONY: setup test evaluate run dashboard docker-up lint seed

setup:
	pip install -e ".[dev,dashboard]"
	pre-commit install || true

test:
	pytest tests/ -v --cov=medflow --cov=api --cov-report=html --cov-report=term-missing

evaluate:
	python scripts/run_evaluation.py --config configs/default.yaml

lint:
	ruff check src/ tests/ scripts/
	mypy src/

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

docker-up:
	docker compose up -d --build

seed:
	python scripts/seed_synthetic_data.py --config configs/default.yaml
