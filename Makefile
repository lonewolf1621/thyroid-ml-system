.PHONY: setup data train test lint serve docker run drift

setup:            ## install dev dependencies
	pip install -r requirements-dev.txt

data:             ## download the public dataset
	python -m thyroid.data download --out data/hypothyroid.csv

train: data       ## train, evaluate and package the model into models/
	python -m thyroid.train --data data/hypothyroid.csv --out models

test:             ## run the test suite (offline, synthetic data)
	pytest --cov=thyroid --cov=api --cov-report=term-missing

lint:
	ruff check .

serve:            ## run the API + UI on http://localhost:8000
	PREDICTION_LOG_PATH=logs/predictions.jsonl uvicorn api.main:app --reload

docker:           ## build the self-contained image
	docker build -t thyroid-ml-system:local .

run: docker       ## run the image
	docker run --rm -p 8000:8000 thyroid-ml-system:local

drift:            ## check logged requests for drift against the training data
	python -m thyroid.drift --model-dir models --current logs/predictions.jsonl
