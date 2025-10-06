.PHONY: bootstrap lint test smoke

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
BLACK := $(VENV)/bin/black
IMPORTLINTER := $(VENV)/bin/lint-imports

bootstrap:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements-lock.txt

lint:
	$(RUFF) check cli feedflipnets tests
	$(BLACK) --check cli feedflipnets tests
	$(IMPORTLINTER)

test:
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(PYTEST) -m "not network" --cov=feedflipnets --cov-report=term-missing --cov-fail-under=75

smoke:
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(VENV)/bin/python -m cli.main --preset basic_dfa_cpu
