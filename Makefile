.PHONY: bootstrap lint test smoke perf bundle release-rc

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
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(PYTEST) -m "not network and not perf" --cov=feedflipnets --cov-report=term-missing --cov-fail-under=75

smoke:
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(VENV)/bin/python -m cli.main --preset basic_dfa_cpu

perf:
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(PYTEST) -m "perf" --maxfail=1

bundle:
	@latest=$$(ls -td .artifacts/* 2>/dev/null | head -n 1); \
	if [ -z "$$latest" ]; then echo "No artifacts found in .artifacts"; exit 1; fi; \
	PYTHONPATH=. FEEDFLIP_DATA_OFFLINE=1 $(VENV)/bin/python scripts/build_paper_bundle.py --run-dir "$$latest" --out paper_bundle --include-plots

release-rc:
	@echo "FeedFlipNets v1.0.0-rc1 candidate ready. Run make lint test perf bundle before tagging."
