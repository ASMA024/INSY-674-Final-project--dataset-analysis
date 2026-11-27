.PHONY: install test lint train register serve drift docker mlflow-ui

install:
	pip install -r requirements-dev.txt && pip install -e .
lint:
	ruff check .
test:
	pytest --cov=churn
train:
	python -m churn.train --data data/raw/telco.csv
register:
	python -m churn.train --data data/raw/telco.csv --register
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
serve:
	uvicorn churn.api:app --reload
drift:
	for s in none price_hike fiber_rollout new_customers; do \
		python -m churn.drift --data data/raw/telco.csv --scenario $$s; done
docker:
	docker build -t churn-api . && docker run --rm -p 8000:8000 churn-api
