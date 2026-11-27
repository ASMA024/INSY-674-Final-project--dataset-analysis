.PHONY: setup install format lint test integration-test check train register mlflow-ui serve drift docker

IMAGE ?= churn-api
PORT ?= 8000

setup: install
	pre-commit install
install:
	pip install -r requirements-dev.txt && pip install -e .

format:
	ruff format .
	ruff check . --fix
lint:
	ruff format --check .
	ruff check .
test:
	python -m pytest --cov=churn
integration-test:
	docker build -t $(IMAGE) .
	docker rm -f churn-it 2>/dev/null || true
	docker run -d --name churn-it -p $(PORT):8000 $(IMAGE)
	for i in $$(seq 1 30); do curl -fs localhost:$(PORT)/health >/dev/null && break; sleep 1; done
	API_URL=http://localhost:$(PORT) python -m pytest -m integration -v; \
		status=$$?; docker rm -f churn-it >/dev/null; exit $$status
check: lint test integration-test

train:
	python -m churn.train --data data/raw/telco.csv
register:
	python -m churn.train --data data/raw/telco.csv --register
mlflow-ui:
	python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
serve:
	python -m uvicorn churn.api:app --reload --port $(PORT)
drift:
	for s in none price_hike fiber_rollout new_customers; do \
		python -m churn.drift --data data/raw/telco.csv --scenario $$s; done
docker:
	docker build -t $(IMAGE) . && docker run --rm -p $(PORT):8000 $(IMAGE)
