.PHONY: help setup backend frontend dev test lint build docker up down logs clean

PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

help:
	@echo "make setup     - create the virtualenv, install backend + frontend deps"
	@echo "make backend   - run the API on :8000 (reload)"
	@echo "make frontend  - run the Vite dev server on :5173"
	@echo "make test      - backend test suite"
	@echo "make lint      - ruff"
	@echo "make build     - build the frontend into frontend/dist"
	@echo "make up        - docker compose up (app + ollama + postgres)"
	@echo "make down      - docker compose down"

setup:
	python3 -m venv backend/.venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements-ml.txt -r backend/requirements-dev.txt
	cd frontend && npm install
	@test -f .env || (cp .env.example .env && echo "Created .env — fill in SECRET_KEY and SMTP_PASSWORD")

backend:
	cd backend && .venv/bin/python -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/python -m pytest

lint:
	cd backend && .venv/bin/python -m ruff check app tests

build:
	cd frontend && npm run build

docker:
	docker build -t ai-interview-analyzer:latest .

up:
	docker compose up -d --build
	@echo "Pull the model once with: docker compose run --rm model-init"

down:
	docker compose down

logs:
	docker compose logs -f app

clean:
	rm -rf frontend/dist backend/.pytest_cache backend/data/*.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
